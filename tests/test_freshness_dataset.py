"""Dataset discovery: the folder→label mapping that makes public sets usable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.dataset import (
    DatasetError,
    FreshnessDataset,
    map_folder_to_label,
)


def _make_images(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (folder / f"img_{i:03d}.jpg").write_bytes(b"stub")


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("fresh", "fresh"),
        ("freshapples", "fresh"),
        ("Fresh_Bananas", "fresh"),
        ("good_quality", "fresh"),
        ("unripe", "fresh"),
        ("ripening", "ripening"),
        ("semi-ripe", "ripening"),
        ("Turning", "ripening"),
        ("rotten", "spoiled"),
        ("rottenbanana", "spoiled"),
        ("Overripe", "spoiled"),
        ("decayed", "spoiled"),
        ("stale_bread", "spoiled"),
    ],
)
def test_keyword_mapping(folder, expected):  # noqa: ANN001
    assert map_folder_to_label(folder) == expected


def test_negation_beats_the_bare_keyword():
    # "notfresh" contains "fresh"; naive substring matching would call it fresh.
    assert map_folder_to_label("notfresh") == "spoiled"
    assert map_folder_to_label("not_fresh_apples") == "spoiled"


def test_overripe_is_spoiled_not_ripening():
    # "overripe" contains "ripe"; ordering must resolve it to spoiled.
    assert map_folder_to_label("overripe") == "spoiled"


def test_unknown_folder_returns_none():
    assert map_folder_to_label("miscellaneous") is None


def test_explicit_override_wins_over_keywords():
    overrides = {"freshapples": "spoiled"}
    assert map_folder_to_label("freshapples", overrides) == "spoiled"


def test_discovers_kaggle_layout(tmp_path: Path):
    _make_images(tmp_path / "train" / "freshapples", 6)
    _make_images(tmp_path / "train" / "rottenbanana", 4)
    _make_images(tmp_path / "test" / "freshoranges", 2)

    dataset = FreshnessDataset.discover(tmp_path)
    assert len(dataset.train) == 10
    assert len(dataset.val) == 2
    assert dataset.distribution("train") == {"fresh": 6, "ripening": 0, "spoiled": 4}


def test_discovers_roboflow_layout(tmp_path: Path):
    _make_images(tmp_path / "train" / "Fresh", 5)
    _make_images(tmp_path / "train" / "Overripe", 5)
    _make_images(tmp_path / "valid" / "Fresh", 2)

    dataset = FreshnessDataset.discover(tmp_path)
    assert len(dataset.train) == 10 and len(dataset.val) == 2


def test_flat_layout_gets_a_held_out_split(tmp_path: Path):
    _make_images(tmp_path / "fresh", 10)
    _make_images(tmp_path / "spoiled", 10)

    dataset = FreshnessDataset.discover(tmp_path, val_split=0.2, seed=1)
    assert len(dataset.val) == 4  # stratified: 2 per class
    assert len(dataset.train) == 16
    # No image may appear in both splits.
    assert not {p for p, _ in dataset.train} & {p for p, _ in dataset.val}


def test_split_is_deterministic_for_a_seed(tmp_path: Path):
    _make_images(tmp_path / "fresh", 10)
    _make_images(tmp_path / "spoiled", 10)
    first = FreshnessDataset.discover(tmp_path, val_split=0.2, seed=7).val
    second = FreshnessDataset.discover(tmp_path, val_split=0.2, seed=7).val
    assert first == second


def test_unmapped_folders_are_reported_not_absorbed(tmp_path: Path):
    _make_images(tmp_path / "fresh", 3)
    _make_images(tmp_path / "mystery_category", 3)

    dataset = FreshnessDataset.discover(tmp_path, val_split=0.0)
    assert "mystery_category" in dataset.unmapped
    assert len(dataset.train) == 3  # the mystery images were not silently included
    assert "UNMAPPED" in dataset.describe()


def test_class_map_file_is_honoured(tmp_path: Path):
    _make_images(tmp_path / "category_a", 4)
    mapping = tmp_path / "classes.json"
    mapping.write_text(json.dumps({"mapping": {"category_a": "ripening"}}), encoding="utf-8")

    dataset = FreshnessDataset.discover(tmp_path, class_map_file=mapping, val_split=0.0)
    assert dataset.distribution("train")["ripening"] == 4


def test_binary_dataset_warns_about_the_missing_class(tmp_path: Path):
    _make_images(tmp_path / "fresh", 5)
    _make_images(tmp_path / "rotten", 5)

    description = FreshnessDataset.discover(tmp_path, val_split=0.0).describe()
    assert "ripening" in description
    assert "WARNING" in description  # empty class is surfaced, not hidden


def test_empty_directory_raises_with_guidance(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError, match="No labelled images"):
        FreshnessDataset.discover(tmp_path)


def test_missing_directory_raises(tmp_path: Path):
    with pytest.raises(DatasetError, match="not found"):
        FreshnessDataset.discover(tmp_path / "nope")
