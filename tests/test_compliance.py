from app.models.enums import ComplianceStatus
from app.schemas.common import BoundingBox, Detection
from app.services.compliance import ComplianceEngine


def _detection(sku: str, box, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=0,
        class_name=sku,
        confidence=confidence,
        bbox=BoundingBox.from_xyxy(box),
        sku=sku,
    )


def test_perfect_shelf_is_fully_compliant(default_planogram):
    detections = [
        _detection(slot.sku, slot.bbox.as_tuple())
        for _shelf, _row, slot in default_planogram.iter_slots()
    ]
    result = ComplianceEngine().evaluate(default_planogram, detections)

    assert result.total_slots == default_planogram.slot_count
    assert result.compliant_slots == result.total_slots
    assert result.extra_detections == 0
    assert result.compliance_score == 1.0
    assert result.spatial_alignment_accuracy == 1.0


def test_empty_shelf_reports_every_slot_missing(default_planogram):
    result = ComplianceEngine().evaluate(default_planogram, [])
    assert result.missing_slots == result.total_slots
    assert result.compliance_score == 0.0
    # No slot is filled, so alignment is undefined -> reported as 0, not 1.
    assert result.spatial_alignment_accuracy == 0.0


def test_wrong_sku_in_slot_is_misplaced_not_missing(default_planogram):
    shelf = default_planogram.shelves[0]
    slot = shelf.rows[0].slots[0]
    intruder = _detection("SKU-WRONG-999", slot.bbox.as_tuple())

    result = ComplianceEngine().evaluate(default_planogram, [intruder])
    verdicts = {s.slot_id: s.status for s in result.slot_results}
    assert verdicts[slot.slot_id] is ComplianceStatus.MISPLACED
    assert result.extra_detections == 0


def test_detection_outside_every_slot_counts_as_extra(default_planogram):
    stray = _detection("SKU-COLA-330", (0.86, 0.08, 0.96, 0.28))
    result = ComplianceEngine().evaluate(default_planogram, [stray])
    assert result.extra_detections == 1
    assert result.false_positive_rate > 0.0


def test_low_confidence_detections_are_discarded(default_planogram):
    _shelf, _row, slot = next(default_planogram.iter_slots())
    weak = _detection(slot.sku, slot.bbox.as_tuple(), confidence=0.05)
    result = ComplianceEngine().evaluate(default_planogram, [weak])
    verdicts = {s.slot_id: s.status for s in result.slot_results}
    assert verdicts[slot.slot_id] is ComplianceStatus.MISSING
