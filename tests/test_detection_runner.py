from __future__ import annotations

from pathlib import Path

from evaluation.detection_runner import (
    load_labels,
    resolve_label_path,
    yolo_line_to_xyxy,
)


def test_yolo_line_converts_center_format_to_xyxy():
    parsed = yolo_line_to_xyxy(["0", "0.5", "0.5", "0.2", "0.4"])
    assert parsed is not None
    class_id, box = parsed
    assert class_id == 0
    assert box == [0.4, 0.3, 0.6, 0.7]


def test_yolo_line_clips_boxes_crossing_the_frame_edge():
    parsed = yolo_line_to_xyxy(["3", "0.05", "0.05", "0.4", "0.4"])
    assert parsed is not None
    _, box = parsed
    assert box[0] == 0.0 and box[1] == 0.0


def test_yolo_line_rejects_malformed_rows():
    assert yolo_line_to_xyxy(["0", "0.5", "0.5"]) is None
    assert yolo_line_to_xyxy(["x", "0.5", "0.5", "0.2", "0.2"]) is None
    assert yolo_line_to_xyxy(["0", "0.5", "0.5", "0", "0.2"]) is None


def test_load_labels_skips_bad_lines_and_comments(tmp_path: Path):
    label = tmp_path / "frame.txt"
    label.write_text(
        "\n".join(
            [
                "# exported by CVAT",
                "0 0.5 0.5 0.2 0.4",
                "garbage row",
                "",
                "1 0.25 0.25 0.1 0.1",
            ]
        ),
        encoding="utf-8",
    )
    parsed = load_labels(label)
    assert parsed["labels"] == [0, 1]
    assert len(parsed["boxes"]) == 2


def test_load_labels_missing_file_is_empty_not_an_error(tmp_path: Path):
    assert load_labels(tmp_path / "absent.txt") == {"boxes": [], "labels": []}


def test_resolve_label_path_prefers_ultralytics_layout(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    image = images / "frame-001.jpg"
    image.write_bytes(b"stub")
    (labels / "frame-001.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")

    assert resolve_label_path(image, None) == labels / "frame-001.txt"


def test_resolve_label_path_honours_explicit_directory(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    override = tmp_path / "gt"
    assert resolve_label_path(image, override) == override / "frame.txt"
