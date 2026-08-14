"""SKU-110K subsetting: official split boundaries and label resolvability.

Two properties matter here and both fail silently if broken.

The first is that a subset never mixes the published train/validation/test
partitions — the reason for moving to this dataset at all was that the previous
corpus leaked across them.

The second is that Ultralytics can actually find the label file for every image
listed. It locates labels by swapping the literal substring `os.sep + "images" +
os.sep`, so a list written with the wrong separator resolves every label to a
path that does not exist. Ultralytics then treats each image as unlabelled
background and trains a detector that never sees a box, without raising
anything. `test_write_list_uses_native_separators` guards precisely that.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from tools.subset_sku110k import (
    list_split,
    main,
    stride_sample,
    verify_labels,
    write_list,
)


def _dataset(root: Path, counts=None) -> Path:
    """A miniature SKU-110K: images/<split>/ and labels/<split>/ side by side."""
    counts = counts or {"train": 40, "val": 12, "test": 15}
    for split, n in counts.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (root / "images" / split / f"{split}_{i}.jpg").write_bytes(b"\xff\xd8jpeg")
            (root / "labels" / split / f"{split}_{i}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )
    return root


# ---------------------------------------------------------- stride_sample --
def test_stride_sample_returns_the_requested_count():
    items = [Path(f"{i}.jpg") for i in range(100)]
    assert len(stride_sample(items, 10)) == 10


def test_stride_sample_spans_the_whole_range():
    """A prefix would over-represent whichever stores appear first."""
    items = [Path(f"{i:03d}.jpg") for i in range(100)]
    picked = stride_sample(items, 10)
    positions = [items.index(p) for p in picked]
    assert positions[0] < 10, "should start near the beginning"
    assert positions[-1] > 80, f"should reach the end, stopped at {positions[-1]}"


def test_stride_sample_returns_everything_when_count_exceeds_total():
    items = [Path(f"{i}.jpg") for i in range(5)]
    assert len(stride_sample(items, 50)) == 5


def test_stride_sample_returns_nothing_for_non_positive_count():
    items = [Path(f"{i}.jpg") for i in range(5)]
    assert stride_sample(items, 0) == []
    assert stride_sample(items, -3) == []


def test_stride_sample_never_repeats_an_item():
    items = [Path(f"{i}.jpg") for i in range(30)]
    picked = stride_sample(items, 29)
    assert len(picked) == len(set(picked))


# ------------------------------------------------------------ list_split --
def test_list_split_reads_only_images(tmp_path: Path):
    root = _dataset(tmp_path, {"train": 3, "val": 1, "test": 1})
    (root / "images" / "train" / "notes.txt").write_text("ignore me", encoding="utf-8")

    found = list_split(root, "train")
    assert len(found) == 3
    assert all(p.suffix == ".jpg" for p in found)


def test_list_split_raises_on_a_missing_split(tmp_path: Path):
    root = _dataset(tmp_path, {"train": 2, "val": 1, "test": 1})
    with pytest.raises(SystemExit):
        list_split(root, "nonexistent")


# --------------------------------------------------------- verify_labels --
def test_verify_labels_counts_missing_and_empty(tmp_path: Path):
    root = _dataset(tmp_path, {"train": 4, "val": 1, "test": 1})
    images = list_split(root, "train")
    (root / "labels" / "train" / "train_0.txt").unlink()
    (root / "labels" / "train" / "train_1.txt").write_text("", encoding="utf-8")

    missing, empty = verify_labels(images)
    assert missing == 1
    assert empty == 1


# ------------------------------------------------------------ write_list --
def test_write_list_uses_native_separators(tmp_path: Path):
    """Regression: forward slashes make Ultralytics resolve zero labels.

    img2label_paths swaps f"{os.sep}images{os.sep}" for the labels equivalent.
    On Windows that is a backslash form, so a forward-slash list matches nothing,
    every label silently resolves to a non-existent path, and training proceeds
    on images the loader believes contain no objects.
    """
    root = _dataset(tmp_path, {"train": 2, "val": 1, "test": 1})
    images = list_split(root, "train")
    target = tmp_path / "out" / "train.txt"

    write_list(target, images)
    lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln]

    assert len(lines) == 2
    for line in lines:
        assert f"{os.sep}images{os.sep}" in line, (
            f"line does not contain the native images separator: {line}"
        )


def test_written_list_resolves_to_real_label_files(tmp_path: Path):
    """End-to-end against Ultralytics' own resolver, when it is installed."""
    ultra = pytest.importorskip("ultralytics.data.utils")
    root = _dataset(tmp_path, {"train": 6, "val": 2, "test": 2})
    images = list_split(root, "train")
    target = tmp_path / "out" / "train.txt"
    write_list(target, images)

    listed = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln]
    resolved = ultra.img2label_paths(listed)
    assert all(Path(p).is_file() for p in resolved), "labels must resolve on disk"


# ------------------------------------------------------------------ main --
def test_main_preserves_official_split_boundaries(tmp_path: Path):
    root = _dataset(tmp_path, {"train": 40, "val": 12, "test": 15})
    out = tmp_path / "subset"
    config = tmp_path / "sku.yaml"

    code = main(
        [
            "--root",
            str(root),
            "--out",
            str(out),
            "--yaml",
            str(config),
            "--train",
            "10",
            "--val",
            "5",
            "--test",
            "0",
        ]
    )
    assert code == 0

    for split in ("train", "val", "test"):
        listed = [
            ln for ln in (out / f"{split}.txt").read_text(encoding="utf-8").splitlines() if ln
        ]
        # Filenames encode their split, so a boundary crossing is detectable.
        for line in listed:
            assert f"{os.sep}{split}{os.sep}" in line, (
                f"{split}.txt contains an image from another split: {line}"
            )


def test_main_keeps_the_full_test_split_when_asked_for_zero(tmp_path: Path):
    root = _dataset(tmp_path, {"train": 20, "val": 6, "test": 15})
    out = tmp_path / "subset"

    main(
        [
            "--root",
            str(root),
            "--out",
            str(out),
            "--yaml",
            str(tmp_path / "s.yaml"),
            "--train",
            "5",
            "--val",
            "3",
            "--test",
            "0",
        ]
    )

    listed = [ln for ln in (out / "test.txt").read_text(encoding="utf-8").splitlines() if ln]
    assert len(listed) == 15, "test split must be retained whole for comparability"


def test_main_writes_a_usable_yaml_and_manifest(tmp_path: Path):
    root = _dataset(tmp_path, {"train": 20, "val": 6, "test": 8})
    out = tmp_path / "subset"
    config = tmp_path / "sku.yaml"

    main(
        [
            "--root",
            str(root),
            "--out",
            str(out),
            "--yaml",
            str(config),
            "--train",
            "8",
            "--val",
            "4",
            "--test",
            "0",
        ]
    )

    parsed = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert parsed["nc"] == 1
    assert parsed["names"] == ["product"]
    for key in ("train", "val", "test"):
        assert Path(parsed[key]).is_file(), f"{key} list file must exist"

    manifest = json.loads((out / "subset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["selected"] == 8
    assert manifest["splits"]["train"]["available"] == 20
    assert "official" in manifest["split_policy"]


def test_main_aborts_when_labels_are_largely_missing(tmp_path: Path):
    """A wrong layout assumption must fail loudly, not train on nothing."""
    root = _dataset(tmp_path, {"train": 20, "val": 4, "test": 4})
    for path in (root / "labels" / "train").glob("*.txt"):
        path.unlink()

    with pytest.raises(SystemExit):
        main(
            [
                "--root",
                str(root),
                "--out",
                str(tmp_path / "subset"),
                "--yaml",
                str(tmp_path / "s.yaml"),
                "--train",
                "10",
                "--val",
                "2",
                "--test",
                "0",
            ]
        )
