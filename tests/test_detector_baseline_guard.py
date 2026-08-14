"""Detecting that the loaded detector is the stock COCO model.

Every route into this state is silent. models/weights is gitignored, so a fresh
clone has no checkpoint; Ultralytics then downloads yolov8n.pt on demand; the
service loads it without error; the health endpoint reports detector_loaded
true. Nothing anywhere says the model recognises people and cars rather than
shelf products, and the first sign of trouble is a demo returning nonsense.

Identification is by class vocabulary rather than filename, so a renamed or
relocated copy is caught too.
"""

from __future__ import annotations

from app.services.detection import DetectionService

COCO_SUBSET = {
    0: "person",
    1: "bicycle",
    2: "car",
    9: "traffic light",
    10: "fire hydrant",
    39: "bottle",
}

SHELF_CLASSES = {0: "product"}

BRANDED_SHELF_CLASSES = {
    0: "Bisleri 1L",
    1: "Thums Up 750ml",
    2: "Lay's Classic 52g",
    3: "Amul Taaza 500ml",
}


def _service_with(names: dict[int, str]) -> DetectionService:
    service = DetectionService()
    service._names = names  # noqa: SLF001 - exercising the guard without a real model
    return service


def test_flags_the_coco_baseline():
    assert _service_with(COCO_SUBSET).is_generic_baseline is True


def test_single_class_shelf_detector_is_not_flagged():
    """SKU-110K is single-class; it must not trip the guard."""
    assert _service_with(SHELF_CLASSES).is_generic_baseline is False


def test_branded_multi_class_detector_is_not_flagged():
    assert _service_with(BRANDED_SHELF_CLASSES).is_generic_baseline is False


def test_unloaded_service_is_not_flagged():
    """Absent a model there is nothing to judge; failure is reported elsewhere."""
    assert _service_with({}).is_generic_baseline is False


def test_requires_several_tells_not_just_one():
    """A shelf dataset may legitimately contain one overlapping word.

    "bottle" is a plausible product class, so a single match must not condemn an
    otherwise valid detector.
    """
    assert _service_with({0: "bottle", 1: "can", 2: "carton"}).is_generic_baseline is False


def test_detection_is_case_insensitive():
    shouty = {0: "PERSON", 1: "Bicycle", 2: "CAR", 3: "Traffic Light"}
    assert _service_with(shouty).is_generic_baseline is True


def test_health_schema_carries_the_flag():
    from app.schemas.common import HealthResponse

    assert "detector_is_generic_baseline" in HealthResponse.model_fields
    # Defaults false so an unrelated construction never raises a false alarm.
    assert HealthResponse.model_fields["detector_is_generic_baseline"].default is False
