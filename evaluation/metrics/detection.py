"""Object-detection metrics: mAP@0.5, mAP@0.5:0.95, precision, recall, latency.

Primary path uses `torchmetrics.detection.MeanAveragePrecision` (the reference
implementation reviewers expect). A dependency-free fallback computes IoU-matched
precision/recall/AP@0.5 so the harness still reports numbers on a machine without
torch installed — the report records which path produced the figures.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Sequence, Tuple

from app.utils.geometry import iou_xyxy


def compute_map(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute detection metrics for parallel prediction/target lists.

    Each entry is `{"boxes": [[x1,y1,x2,y2], ...], "labels": [int, ...],
    "scores": [float, ...]}` (scores only on predictions), in normalised xyxy.
    """
    try:
        return _torchmetrics_map(predictions, targets)
    except ImportError as exc:
        # torchmetrics' MeanAveragePrecision needs a COCO eval backend
        # (pycocotools or faster-coco-eval). Falling back is fine for a smoke
        # run, but the reason must land in the report — a silent fallback is how
        # a paper ends up quoting AP@0.5 as if it were mAP@0.5:0.95.
        result = _fallback_map(predictions, targets, iou_threshold)
        result["backend"] = "fallback"
        result["backend_error"] = (
            f"{exc} — install pycocotools (or faster-coco-eval) for mAP@0.5:0.95"
        )
        return result


def _torchmetrics_map(
    predictions: Sequence[Dict[str, Any]], targets: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    import torch  # noqa: PLC0415
    from torchmetrics.detection import MeanAveragePrecision  # noqa: PLC0415

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(
        [
            {
                "boxes": torch.tensor(p["boxes"], dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor(p.get("scores", []), dtype=torch.float32),
                "labels": torch.tensor(p["labels"], dtype=torch.int64),
            }
            for p in predictions
        ],
        [
            {
                "boxes": torch.tensor(t["boxes"], dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(t["labels"], dtype=torch.int64),
            }
            for t in targets
        ],
    )
    computed = {k: float(v) for k, v in metric.compute().items() if v.numel() == 1}
    pr = _fallback_map(predictions, targets, 0.5)
    return {
        "backend": "torchmetrics",
        "map_50": computed.get("map_50"),
        "map_50_95": computed.get("map"),
        "map_small": computed.get("map_small"),
        "map_medium": computed.get("map_medium"),
        "map_large": computed.get("map_large"),
        "precision": pr["precision"],
        "recall": pr["recall"],
        "f1": pr["f1"],
        "true_positives": pr["true_positives"],
        "false_positives": pr["false_positives"],
        "false_negatives": pr["false_negatives"],
    }


def _fallback_map(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    iou_threshold: float,
) -> Dict[str, Any]:
    """Greedy score-ordered matching → precision, recall, F1 and AP@IoU."""
    tp = fp = fn = 0
    records: List[tuple] = []  # (score, is_tp)
    total_gt = 0

    for pred, truth in zip(predictions, targets):
        boxes = list(pred.get("boxes", []))
        labels = list(pred.get("labels", []))
        scores = list(pred.get("scores", [1.0] * len(boxes)))
        gt_boxes = list(truth.get("boxes", []))
        gt_labels = list(truth.get("labels", []))
        total_gt += len(gt_boxes)

        matched: set[int] = set()
        for _score, box, label in sorted(
            zip(scores, boxes, labels), key=lambda row: row[0], reverse=True
        ):
            best_iou, best_idx = 0.0, -1
            for gi, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gi in matched or gt_label != label:
                    continue
                value = iou_xyxy(tuple(box), tuple(gt_box))
                if value > best_iou:
                    best_iou, best_idx = value, gi
            hit = best_iou >= iou_threshold and best_idx >= 0
            if hit:
                matched.add(best_idx)
                tp += 1
            else:
                fp += 1
            records.append((_score, hit))
        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "map_50": _average_precision(records, total_gt),
        "map_50_95": None,  # single-threshold fallback cannot produce the sweep
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def pr_curve(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, List[Tuple[float, float]]]:
    """Score-ranked precision-recall points, per class.

    Built from the same greedy IoU matching the fallback AP uses, so the curve
    and the reported AP@0.5 can never disagree — a figure that contradicts its
    own table is worse than no figure.
    """
    per_class_records: Dict[int, List[tuple]] = {}
    per_class_gt: Dict[int, int] = {}

    for pred, truth in zip(predictions, targets):
        boxes = list(pred.get("boxes", []))
        labels = list(pred.get("labels", []))
        scores = list(pred.get("scores", [1.0] * len(boxes)))
        gt_boxes = list(truth.get("boxes", []))
        gt_labels = list(truth.get("labels", []))

        for label in set(gt_labels):
            per_class_gt[label] = per_class_gt.get(label, 0) + gt_labels.count(label)

        matched: set[int] = set()
        for score, box, label in sorted(
            zip(scores, boxes, labels), key=lambda row: row[0], reverse=True
        ):
            best_iou, best_index = 0.0, -1
            for gi, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gi in matched or gt_label != label:
                    continue
                value = iou_xyxy(tuple(box), tuple(gt_box))
                if value > best_iou:
                    best_iou, best_index = value, gi
            hit = best_iou >= iou_threshold and best_index >= 0
            if hit:
                matched.add(best_index)
            per_class_records.setdefault(label, []).append((score, hit))

    curves: Dict[str, List[Tuple[float, float]]] = {}
    for label, records in per_class_records.items():
        total = per_class_gt.get(label, 0)
        if not total:
            continue
        tp = fp = 0
        points: List[Tuple[float, float]] = []
        for _score, hit in sorted(records, key=lambda r: r[0], reverse=True):
            tp, fp = (tp + 1, fp) if hit else (tp, fp + 1)
            points.append((tp / total, tp / (tp + fp)))
        curves[f"class {label}"] = points
    return curves


def _average_precision(records: Sequence[tuple], total_gt: int) -> float:
    """11-point interpolated AP over the score-ranked detection list."""
    if not records or total_gt == 0:
        return 0.0
    ranked = sorted(records, key=lambda r: r[0], reverse=True)
    tp = fp = 0
    curve: List[tuple] = []
    for _score, hit in ranked:
        tp, fp = (tp + 1, fp) if hit else (tp, fp + 1)
        curve.append((tp / total_gt, tp / (tp + fp)))

    ap = 0.0
    for point in [i / 10 for i in range(11)]:
        candidates = [p for r, p in curve if r >= point]
        ap += max(candidates) if candidates else 0.0
    return round(ap / 11, 4)


def latency_stats(samples_ms: Sequence[float]) -> Dict[str, float]:
    """Inference-latency distribution — reported alongside accuracy in the paper."""
    values = [float(v) for v in samples_ms if v is not None]
    if not values:
        return {}
    ordered = sorted(values)
    # Nearest-rank percentile: ceil(p·n) - 1. Truncating instead would report
    # the *minimum* as p95 for a two-frame run, which is worse than useless.
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "count": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "fps": round(1000.0 / statistics.fmean(ordered), 2) if statistics.fmean(ordered) else 0.0,
    }
