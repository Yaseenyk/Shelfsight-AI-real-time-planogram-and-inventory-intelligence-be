"""Planogram compliance engine.

Algorithm (documented in docs/ARCHITECTURE.md, reported in the paper):

1. Restrict detections to the shelf's `y_range` band.
2. Spatially sort detections into rows (y-centre clustering) then left-to-right.
3. For each expected slot, consider only detections whose SKU matches, and match
   greedily by IoU with a Euclidean centre-distance rescue.
4. Classify each slot: COMPLIANT / MISPLACED / MISSING; leftover detections that
   belong to no slot are EXTRA.
5. Roll up compliance score, spatial alignment accuracy, mean IoU / distance and
   the discrepancy false-positive rate.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.models.enums import ComplianceStatus
from app.schemas.common import Detection
from app.schemas.planogram import (
    PlanogramDocument,
    PlanogramTolerances,
    SlotResult,
)
from app.utils.geometry import (
    Box,
    center_distance,
    greedy_match,
    iou_xyxy,
    mean,
    sort_detections_by_shelf_row,
)


class ComplianceResult:
    """Container for the engine output (mapped 1:1 onto ComplianceAudit)."""

    def __init__(
        self,
        slot_results: List[SlotResult],
        extra_detections: int,
        latency_ms: float,
    ) -> None:
        self.slot_results = slot_results
        self.extra_detections = extra_detections
        self.latency_ms = latency_ms

    def _count(self, status: ComplianceStatus) -> int:
        return sum(1 for s in self.slot_results if s.status is status)

    @property
    def total_slots(self) -> int:
        return len(self.slot_results)

    @property
    def compliant_slots(self) -> int:
        return self._count(ComplianceStatus.COMPLIANT)

    @property
    def misplaced_slots(self) -> int:
        return self._count(ComplianceStatus.MISPLACED)

    @property
    def missing_slots(self) -> int:
        return self._count(ComplianceStatus.MISSING)

    @property
    def compliance_score(self) -> float:
        return self.compliant_slots / self.total_slots if self.total_slots else 0.0

    @property
    def spatial_alignment_accuracy(self) -> float:
        """Share of *filled* slots whose product sits within tolerance.

        Missing slots are excluded: they measure availability, not alignment.
        """
        filled = [s for s in self.slot_results if s.status is not ComplianceStatus.MISSING]
        if not filled:
            return 0.0
        aligned = sum(1 for s in filled if s.status is ComplianceStatus.COMPLIANT)
        return aligned / len(filled)

    @property
    def mean_iou(self) -> float:
        return mean(s.iou for s in self.slot_results if s.observed_sku is not None)

    @property
    def mean_center_distance(self) -> float:
        return mean(
            s.center_distance for s in self.slot_results if s.observed_sku is not None
        )

    @property
    def false_positive_rate(self) -> float:
        """Extra detections as a share of all reported non-compliances.

        Detections outside every slot are the dominant false-positive source for
        the discrepancy alerting layer, so this is the metric the paper reports.
        """
        flagged = self.misplaced_slots + self.missing_slots + self.extra_detections
        return self.extra_detections / flagged if flagged else 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_slots": self.total_slots,
            "compliant_slots": self.compliant_slots,
            "misplaced_slots": self.misplaced_slots,
            "missing_slots": self.missing_slots,
            "extra_detections": self.extra_detections,
            "compliance_score": round(self.compliance_score, 4),
            "spatial_alignment_accuracy": round(self.spatial_alignment_accuracy, 4),
            "mean_iou": round(self.mean_iou, 4),
            "mean_center_distance": round(self.mean_center_distance, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "latency_ms": round(self.latency_ms, 3),
        }


class ComplianceEngine:
    def __init__(self, tolerances: Optional[PlanogramTolerances] = None) -> None:
        self.tolerances = tolerances or PlanogramTolerances(
            iou_threshold=settings.COMPLIANCE_IOU_THRESHOLD,
            center_distance_threshold=settings.COMPLIANCE_CENTER_DISTANCE_THRESHOLD,
            row_band_tolerance=settings.COMPLIANCE_ROW_BAND_TOLERANCE,
            min_detection_confidence=settings.DETECTION_CONF_THRESHOLD,
        )

    def evaluate(
        self,
        planogram: PlanogramDocument,
        detections: Sequence[Detection],
        shelf_id: Optional[str] = None,
    ) -> ComplianceResult:
        started = time.perf_counter()
        tol = planogram.tolerances or self.tolerances

        kept = [d for d in detections if d.confidence >= tol.min_detection_confidence]
        # Reading-order sort; keeps slot_results deterministic across runs.
        boxes: List[Box] = [d.bbox.as_tuple() for d in kept]
        ordered = sort_detections_by_shelf_row(boxes, tol.row_band_tolerance)
        kept = [kept[i] for i in ordered]

        slot_results: List[SlotResult] = []
        consumed: set[int] = set()

        for shelf in planogram.shelves:
            if shelf_id and shelf.shelf_id != shelf_id:
                continue
            low, high = shelf.y_range
            shelf_idx = [
                i
                for i, d in enumerate(kept)
                if i not in consumed and low <= d.bbox.center[1] <= high
            ]
            slots = [
                (row.row_id, slot)
                for row in shelf.rows
                for slot in row.slots
            ]
            if not slots:
                continue

            expected_boxes: List[Box] = [slot.bbox.as_tuple() for _, slot in slots]
            observed_boxes: List[Box] = [kept[i].bbox.as_tuple() for i in shelf_idx]

            # SKU identity gate: a slot may only match a detection of its own SKU.
            candidate_filter: Dict[int, set[int]] = {}
            for ei, (_row_id, slot) in enumerate(slots):
                candidate_filter[ei] = {
                    local
                    for local, det_i in enumerate(shelf_idx)
                    if _sku_of(kept[det_i]) == slot.sku
                }

            matches, unmatched_expected, unmatched_observed = greedy_match(
                expected_boxes,
                observed_boxes,
                iou_threshold=tol.iou_threshold,
                center_threshold=tol.center_distance_threshold,
                candidate_filter=candidate_filter,
            )

            for ei, (row_id, slot) in enumerate(slots):
                if ei in matches:
                    local = matches[ei]
                    det = kept[shelf_idx[local]]
                    consumed.add(shelf_idx[local])
                    slot_results.append(
                        SlotResult(
                            slot_id=slot.slot_id,
                            shelf_id=shelf.shelf_id,
                            row_id=row_id,
                            expected_sku=slot.sku,
                            observed_sku=_sku_of(det),
                            status=ComplianceStatus.COMPLIANT,
                            iou=round(iou_xyxy(expected_boxes[ei], det.bbox.as_tuple()), 4),
                            center_distance=round(
                                center_distance(expected_boxes[ei], det.bbox.as_tuple()), 4
                            ),
                            confidence=det.confidence,
                            expected_bbox=slot.bbox,
                            observed_bbox=det.bbox,
                            expected_facings=slot.expected_facings,
                            observed_facings=1,
                        )
                    )
                    continue

                # No same-SKU detection fits the slot. Is *something else* there?
                intruder = _best_overlap(expected_boxes[ei], kept, shelf_idx, consumed, tol)
                if intruder is None:
                    slot_results.append(
                        SlotResult(
                            slot_id=slot.slot_id,
                            shelf_id=shelf.shelf_id,
                            row_id=row_id,
                            expected_sku=slot.sku,
                            status=ComplianceStatus.MISSING,
                            iou=0.0,
                            center_distance=1.0,
                            expected_bbox=slot.bbox,
                            expected_facings=slot.expected_facings,
                            observed_facings=0,
                        )
                    )
                else:
                    det_i, iou_value = intruder
                    det = kept[det_i]
                    consumed.add(det_i)
                    slot_results.append(
                        SlotResult(
                            slot_id=slot.slot_id,
                            shelf_id=shelf.shelf_id,
                            row_id=row_id,
                            expected_sku=slot.sku,
                            observed_sku=_sku_of(det),
                            status=ComplianceStatus.MISPLACED,
                            iou=round(iou_value, 4),
                            center_distance=round(
                                center_distance(expected_boxes[ei], det.bbox.as_tuple()), 4
                            ),
                            confidence=det.confidence,
                            expected_bbox=slot.bbox,
                            observed_bbox=det.bbox,
                            expected_facings=slot.expected_facings,
                            observed_facings=1,
                        )
                    )

            _ = unmatched_expected, unmatched_observed  # accounted for above

        extra = sum(1 for i in range(len(kept)) if i not in consumed)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ComplianceResult(slot_results, extra, latency_ms)


def _sku_of(detection: Detection) -> str:
    """Prefer a catalogue-resolved SKU, fall back to the raw detector class."""
    return detection.sku or detection.class_name


def _best_overlap(
    slot_box: Box,
    detections: Sequence[Detection],
    shelf_idx: Sequence[int],
    consumed: set,
    tol: PlanogramTolerances,
) -> Optional[Tuple[int, float]]:
    """Find the strongest unconsumed detection overlapping a slot (any SKU)."""
    best: Optional[Tuple[int, float]] = None
    for det_i in shelf_idx:
        if det_i in consumed:
            continue
        value = iou_xyxy(slot_box, detections[det_i].bbox.as_tuple())
        if value >= tol.iou_threshold and (best is None or value > best[1]):
            best = (det_i, value)
    return best


def get_compliance_engine() -> ComplianceEngine:
    return ComplianceEngine()
