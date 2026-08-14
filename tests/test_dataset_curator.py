"""Dataset curation: registry integrity and the merge/sanitisation logic.

Network fetches are not exercised — they need credentials and live third-party
services. What *is* tested is everything that runs on the client's machine
afterwards: folder→class resolution, cross-source deduplication, splitting, and
the manifest that makes the paper's dataset section reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.dataset import CANONICAL_CLASSES
from tools.dataset_curator import (
    DATASET_REGISTRY,
    build_parser,
    collect_labelled_images,
    merge_datasets,
    write_manifest,
)


def _write_images(folder: Path, names, content_prefix: str = "img") -> None:  # noqa: ANN001
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        # Content differs per name so the hash-dedup has something to work with.
        (folder / f"{name}.jpg").write_bytes(f"{content_prefix}-{name}".encode())


# ---------------------------------------------------------------- registry --
def test_registry_entries_are_complete():
    for key, source in DATASET_REGISTRY.items():
        assert source.key == key
        assert source.pipeline in {"freshness", "detection"}
        assert source.provider in {"roboflow", "kaggle", "manual"}
        assert source.url.startswith("http"), key
        assert source.licence, key
        assert source.description, key


def test_roboflow_entries_pin_an_exact_version():
    """A floating version silently changes the dataset under a published result."""
    for source in DATASET_REGISTRY.values():
        if source.provider == "roboflow":
            assert source.workspace and source.project
            assert isinstance(source.version, int)


def test_registry_covers_the_ripening_gap():
    """Phase 2 found no public binary set supplies `ripening`; one source must."""
    providers = [s for s in DATASET_REGISTRY.values() if "ripening" in s.provides]
    assert providers, "no registry entry provides the ripening class"


def test_cli_parses_every_subcommand():
    parser = build_parser()
    assert parser.parse_args(["list"]).command == "list"
    assert parser.parse_args(["fetch", "freshness", "--dry-run"]).dry_run is True
    merged = parser.parse_args(["merge", "--sources", "a", "b", "--val-split", "0.3"])
    assert merged.sources == ["a", "b"] and merged.val_split == 0.3


# -------------------------------------------------------------- collection --
def test_collects_kaggle_style_folders(tmp_path: Path):
    _write_images(tmp_path / "train" / "freshapples", ["a", "b"])
    _write_images(tmp_path / "train" / "rottenbanana", ["c"])

    grouped, unmapped = collect_labelled_images(tmp_path)
    assert len(grouped["fresh"]) == 2
    assert len(grouped["spoiled"]) == 1
    assert unmapped == []


def test_reports_unmapped_folders_instead_of_absorbing_them(tmp_path: Path):
    _write_images(tmp_path / "fresh", ["a"])
    _write_images(tmp_path / "mystery", ["b"])

    grouped, unmapped = collect_labelled_images(tmp_path)
    assert "mystery" in unmapped
    assert sum(len(v) for v in grouped.values()) == 1


def test_class_map_override_is_honoured(tmp_path: Path):
    _write_images(tmp_path / "stage_two", ["a", "b"])
    grouped, _ = collect_labelled_images(tmp_path, {"stage_two": "ripening"})
    assert len(grouped["ripening"]) == 2


def test_missing_directory_is_not_an_error(tmp_path: Path):
    grouped, unmapped = collect_labelled_images(tmp_path / "absent")
    assert unmapped == []
    assert all(not images for images in grouped.values())


# ------------------------------------------------------------------ merge --
def test_merges_binary_and_ripening_sources_into_three_classes(tmp_path: Path):
    binary = tmp_path / "binary"
    _write_images(binary / "train" / "freshapples", [f"f{i}" for i in range(6)])
    _write_images(binary / "train" / "rottenapples", [f"r{i}" for i in range(4)])

    ripeness = tmp_path / "ripeness"
    _write_images(ripeness / "semiripe", [f"s{i}" for i in range(5)], content_prefix="rip")

    out = tmp_path / "merged"
    stats = merge_datasets([binary, ripeness], out, val_split=0.2, seed=1)

    assert stats.per_class == {"fresh": 6, "ripening": 5, "spoiled": 4}
    assert stats.missing_classes == []
    for split in ("train", "val"):
        for label in CANONICAL_CLASSES:
            assert (out / split / label).is_dir()
    assert stats.train_count + stats.val_count == 15


def test_duplicate_images_across_sources_are_removed(tmp_path: Path):
    """Public fruit datasets overlap; the same photo in train and val inflates accuracy."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    _write_images(first / "fresh", ["shared", "unique_a"])
    _write_images(second / "fresh", ["shared", "unique_b"])  # identical content for "shared"

    stats = merge_datasets([first, second], tmp_path / "out", val_split=0.0, seed=1)
    assert stats.duplicates_removed == 1
    assert stats.per_class["fresh"] == 3


def test_no_image_appears_in_both_splits(tmp_path: Path):
    source = tmp_path / "src"
    _write_images(source / "fresh", [f"f{i}" for i in range(10)])

    out = tmp_path / "out"
    merge_datasets([source], out, val_split=0.3, seed=5)

    train = {p.read_bytes() for p in (out / "train" / "fresh").glob("*.jpg")}
    val = {p.read_bytes() for p in (out / "val" / "fresh").glob("*.jpg")}
    assert train and val
    assert not train & val


def test_split_is_deterministic_for_a_seed(tmp_path: Path):
    source = tmp_path / "src"
    _write_images(source / "fresh", [f"f{i}" for i in range(10)])

    first = merge_datasets([source], tmp_path / "a", val_split=0.3, seed=7)
    second = merge_datasets([source], tmp_path / "b", val_split=0.3, seed=7)
    assert first.train_count == second.train_count
    assert first.val_count == second.val_count


def test_missing_class_is_reported_not_silently_empty(tmp_path: Path):
    source = tmp_path / "src"
    _write_images(source / "fresh", ["a", "b"])
    _write_images(source / "rotten", ["c"])

    stats = merge_datasets([source], tmp_path / "out", val_split=0.0)
    assert stats.missing_classes == ["ripening"]


def test_manifest_records_provenance(tmp_path: Path):
    source = tmp_path / "src"
    _write_images(source / "fresh", ["a", "b"])

    out = tmp_path / "out"
    stats = merge_datasets([source], out, val_split=0.5, seed=3)
    path = write_manifest(out, stats, extra={"val_split": 0.5, "seed": 3})

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["classes"] == list(CANONICAL_CLASSES)
    assert manifest["seed"] == 3
    assert manifest["sources"] == [str(source)]
    assert "per_source" in manifest and "duplicates_removed" in manifest
    assert manifest["curated_at"]


@pytest.mark.parametrize("val_split", [0.0, 0.5, 1.0])
def test_extreme_split_ratios_do_not_crash(tmp_path: Path, val_split):  # noqa: ANN001
    source = tmp_path / "src"
    _write_images(source / "fresh", ["a", "b", "c", "d"])
    stats = merge_datasets([source], tmp_path / f"out{val_split}", val_split=val_split)
    assert stats.train_count + stats.val_count == 4
