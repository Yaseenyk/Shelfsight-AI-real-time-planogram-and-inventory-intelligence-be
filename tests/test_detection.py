"""Detector parsing and NMS.

These run **without torch**: the Ultralytics result object is duck-typed by
`_FakeBoxes`/`_FakeResult`, which is exactly what `_parse` consumes. That keeps
the box-validation logic — the part most likely to break silently — under test
on any machine.
"""

from __future__ import annotations

from typing import List, Sequence

import pytest

from app.schemas.common import BoundingBox, Detection
from app.services.detection import (
    DetectorUnavailableError,
    YoloDetector,
    non_max_suppression,
)


class _Scalar:
    """Stands in for a 0-dim torch tensor."""

    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _Row:
    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)

    def tolist(self) -> List[float]:
        return list(self._values)


class _FakeBoxes:
    def __init__(self, rows, classes, scores) -> None:  # noqa: ANN001
        self.xyxyn = [_Row(r) for r in rows]
        self.cls = [_Scalar(c) for c in classes]
        self.conf = [_Scalar(s) for s in scores]

    def __len__(self) -> int:
        return len(self.xyxyn)


class _FakeResult:
    def __init__(self, boxes, names) -> None:  # noqa: ANN001
        self.boxes = boxes
        self.names = names
        self.speed = {"preprocess": 1.0, "inference": 20.0, "postprocess": 0.5}


def _detector() -> YoloDetector:
    return YoloDetector(weights="does-not-exist.pt")


def _detection(name: str, box, conf: float = 0.9, class_id: int = 0) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=name,
        confidence=conf,
        bbox=BoundingBox.from_xyxy(box),
    )


# ------------------------------------------------------------------ parsing --
def test_parse_builds_validated_detections():
    result = _FakeResult(
        _FakeBoxes(
            rows=[[0.1, 0.1, 0.3, 0.4], [0.5, 0.5, 0.8, 0.9]],
            classes=[0, 39],
            scores=[0.91, 0.62],
        ),
        names={0: "person", 39: "bottle"},
    )
    detections, dropped = _detector()._parse([result])

    assert dropped == 0
    assert [d.class_name for d in detections] == ["person", "bottle"]
    assert detections[0].bbox.as_tuple() == (0.1, 0.1, 0.3, 0.4)
    assert detections[1].confidence == pytest.approx(0.62)


def test_parse_clips_boxes_that_exceed_the_frame():
    result = _FakeResult(
        _FakeBoxes(rows=[[-0.05, -0.02, 1.04, 1.10]], classes=[0], scores=[0.8]),
        names={0: "bottle"},
    )
    detections, dropped = _detector()._parse([result])
    assert dropped == 0
    assert detections[0].bbox.as_tuple() == (0.0, 0.0, 1.0, 1.0)


def test_parse_drops_degenerate_and_nan_boxes():
    result = _FakeResult(
        _FakeBoxes(
            rows=[
                [0.5, 0.5, 0.5, 0.5],            # zero area
                [float("nan"), 0.1, 0.2, 0.3],   # NaN coordinate
                [0.1, 0.1, 0.4, 0.4],            # the only good row
            ],
            classes=[0, 0, 0],
            scores=[0.9, 0.9, 0.9],
        ),
        names={0: "bottle"},
    )
    detections, dropped = _detector()._parse([result])
    assert len(detections) == 1
    assert dropped == 2


def test_parse_drops_out_of_range_confidence():
    result = _FakeResult(
        _FakeBoxes(rows=[[0.1, 0.1, 0.4, 0.4]], classes=[0], scores=[1.4]),
        names={0: "bottle"},
    )
    detections, dropped = _detector()._parse([result])
    assert detections == [] and dropped == 1


def test_parse_falls_back_to_class_id_when_name_missing():
    result = _FakeResult(
        _FakeBoxes(rows=[[0.1, 0.1, 0.4, 0.4]], classes=[77], scores=[0.5]), names={}
    )
    detections, _ = _detector()._parse([result])
    assert detections[0].class_name == "77"


def test_parse_tolerates_empty_and_boxless_results():
    empty = _FakeResult(_FakeBoxes([], [], []), names={0: "bottle"})
    boxless = _FakeResult(None, names={})
    detections, dropped = _detector()._parse([empty, boxless])
    assert detections == [] and dropped == 0


# ---------------------------------------------------------------------- NMS --
def test_nms_suppresses_cross_class_duplicate():
    # One physical bottle detected as both `bottle` and `cup`.
    detections = [
        _detection("bottle", (0.10, 0.10, 0.30, 0.40), conf=0.90, class_id=39),
        _detection("cup", (0.11, 0.11, 0.31, 0.41), conf=0.55, class_id=41),
    ]
    kept, suppressed = non_max_suppression(detections, iou_threshold=0.45)
    assert len(kept) == 1 and suppressed == 1
    assert kept[0].class_name == "bottle"  # the higher-confidence box wins


def test_nms_class_aware_keeps_both():
    detections = [
        _detection("bottle", (0.10, 0.10, 0.30, 0.40), conf=0.90, class_id=39),
        _detection("cup", (0.11, 0.11, 0.31, 0.41), conf=0.55, class_id=41),
    ]
    kept, suppressed = non_max_suppression(detections, class_agnostic=False)
    assert len(kept) == 2 and suppressed == 0


def test_nms_keeps_adjacent_facings():
    # Neighbouring facings of the same SKU barely overlap — both must survive,
    # or the shelf count collapses.
    detections = [
        _detection("bottle", (0.10, 0.10, 0.20, 0.40), conf=0.9),
        _detection("bottle", (0.21, 0.10, 0.31, 0.40), conf=0.88),
    ]
    kept, suppressed = non_max_suppression(detections)
    assert len(kept) == 2 and suppressed == 0


def test_nms_on_empty_input():
    assert non_max_suppression([]) == ([], 0)


# ------------------------------------------------------------------ loading --
def test_predict_raises_when_model_unavailable(monkeypatch):  # noqa: ANN001
    from app.core.config import settings

    monkeypatch.setattr(settings, "DETECTION_ALLOW_DOWNLOAD", False)
    detector = YoloDetector(weights="missing-weights.pt")

    with pytest.raises(DetectorUnavailableError):
        detector.predict_with_metrics(object())
    assert detector.is_ready is False
    assert "missing-weights.pt" in (detector.load_failure or "")


def test_version_string_carries_thresholds():
    detector = YoloDetector(weights="w.pt", conf=0.4, iou=0.5, imgsz=640)
    assert detector.version == "yolov8:w.pt@conf0.4:iou0.5:imgsz640"
