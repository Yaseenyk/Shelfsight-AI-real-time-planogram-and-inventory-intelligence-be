"""Ground-truth loading for the OCR benchmark (CSV and JSON)."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.ocr_runner import find_ground_truth, load_ground_truth


def test_loads_csv_with_canonical_columns(tmp_path: Path):
    path = tmp_path / "ground_truth.csv"
    path.write_text(
        "image,truth_text,truth_date\n"
        "exp-001.jpg,EXP 12/09/2026,2026-09-12\n"
        "exp-002.jpg,BB 20260818,2026-08-18\n",
        encoding="utf-8",
    )
    rows = load_ground_truth(path)["by_image"]
    assert rows["exp-001.jpg"]["truth_date"] == "2026-09-12"
    assert rows["exp-002.jpg"]["truth_text"] == "BB 20260818"


def test_accepts_alternative_column_spellings(tmp_path: Path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "filename,text,expiry_date\nexp-003.jpg,USE BY 2026-08-10,2026-08-10\n",
        encoding="utf-8",
    )
    rows = load_ground_truth(path)["by_image"]
    assert rows["exp-003.jpg"]["truth_date"] == "2026-08-10"


def test_blank_truth_date_is_kept_as_unreadable(tmp_path: Path):
    # These rows are the denominator for the read rate — dropping them would
    # flatter the metric by discarding the hard samples.
    path = tmp_path / "ground_truth.csv"
    path.write_text("image,truth_text,truth_date\nexp-006.jpg,,\n", encoding="utf-8")
    rows = load_ground_truth(path)["by_image"]
    assert "exp-006.jpg" in rows
    assert rows["exp-006.jpg"]["truth_date"] is None


def test_csv_without_an_image_column_is_reported_not_crashing(tmp_path: Path):
    path = tmp_path / "ground_truth.csv"
    path.write_text("text,date\nfoo,2026-01-01\n", encoding="utf-8")
    assert load_ground_truth(path)["by_image"] == {}


def test_loads_json_with_reference_date(tmp_path: Path):
    path = tmp_path / "ground_truth.json"
    path.write_text(
        json.dumps(
            {
                "reference_date": "2026-08-14",
                "samples": [
                    {"image": "exp-001.jpg", "truth_text": "EXP", "truth_date": "2026-09-12"}
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = load_ground_truth(path)
    assert payload["reference_date"] == "2026-08-14"
    assert payload["by_image"]["exp-001.jpg"]["truth_date"] == "2026-09-12"


def test_json_paths_are_reduced_to_basenames(tmp_path: Path):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps({"samples": [{"image": "crops/exp-009.jpg", "truth_date": "2026-01-01"}]}),
        encoding="utf-8",
    )
    assert "exp-009.jpg" in load_ground_truth(path)["by_image"]


def test_invalid_json_degrades_to_empty(tmp_path: Path):
    path = tmp_path / "ground_truth.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_ground_truth(path)["by_image"] == {}


def test_missing_file_degrades_to_empty(tmp_path: Path):
    assert load_ground_truth(tmp_path / "absent.csv")["by_image"] == {}


def test_find_ground_truth_looks_beside_and_above_the_images(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    assert find_ground_truth(images) is None

    (tmp_path / "ground_truth.csv").write_text("image\n", encoding="utf-8")
    assert find_ground_truth(images) == tmp_path / "ground_truth.csv"

    (images / "labels.json").write_text("{}", encoding="utf-8")
    assert find_ground_truth(images) == images / "labels.json"
