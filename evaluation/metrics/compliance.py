"""Planogram-compliance metrics: spatial alignment accuracy + discrepancy FPR.

The engine's per-slot verdict is compared against human slot labels:

- **Spatial alignment accuracy** = slots whose verdict matches the human label,
  over all labelled slots.
- **Discrepancy false-positive rate** = slots the engine flagged as non-compliant
  that the human labelled compliant, over all slots the engine flagged. This is
  the number a store manager actually feels — every false alarm is a wasted walk.
- **Discrepancy recall** = true non-compliances caught, over all real ones.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.schemas.common import BoundingBox, Detection
from app.schemas.planogram import PlanogramDocument
from app.services.compliance import ComplianceEngine

COMPLIANT = "compliant"


def evaluate_frames(
    document: PlanogramDocument, frames: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run the engine over labelled frames and score it against `slot_truth`."""
    engine = ComplianceEngine()

    total = matched = 0
    flagged = false_positives = 0
    real_issues = caught_issues = 0
    ious: List[float] = []
    latencies: List[float] = []
    confusion: Dict[str, Dict[str, int]] = {}
    per_frame: List[Dict[str, Any]] = []

    for frame in frames:
        detections = [
            Detection(
                class_id=int(d.get("class_id", 0)),
                class_name=str(d.get("class_name", "")),
                confidence=float(d.get("confidence", 1.0)),
                bbox=BoundingBox.from_xyxy(d["bbox"]),
                sku=d.get("sku"),
            )
            for d in frame.get("detections", [])
        ]
        result = engine.evaluate(document, detections)
        latencies.append(result.latency_ms)

        truth: Dict[str, str] = frame.get("slot_truth", {}) or {}
        frame_matched = 0
        for slot in result.slot_results:
            expected = truth.get(slot.slot_id)
            if expected is None:
                continue
            predicted = slot.status.value
            total += 1
            confusion.setdefault(expected, {}).setdefault(predicted, 0)
            confusion[expected][predicted] += 1

            if predicted == expected:
                matched += 1
                frame_matched += 1
            if predicted != COMPLIANT:
                flagged += 1
                if expected == COMPLIANT:
                    false_positives += 1
            if expected != COMPLIANT:
                real_issues += 1
                if predicted != COMPLIANT:
                    caught_issues += 1
            if slot.observed_sku:
                ious.append(slot.iou)

        per_frame.append(
            {
                "frame_id": frame.get("frame_id"),
                "engine": result.as_dict(),
                "labelled_slots": len(truth),
                "correct_verdicts": frame_matched,
            }
        )

    return {
        "frames": len(frames),
        "labelled_slots": total,
        "spatial_alignment_accuracy": round(matched / total, 4) if total else 0.0,
        "discrepancy_false_positive_rate": round(false_positives / flagged, 4) if flagged else 0.0,
        "discrepancy_recall": round(caught_issues / real_issues, 4) if real_issues else 0.0,
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "verdict_confusion": confusion,
        "per_frame": per_frame,
    }
