"""Spatial primitives behind the planogram compliance engine.

Everything here operates on **normalised xyxy** tuples so the same thresholds
apply to any camera resolution. Pure Python (no numpy) keeps the module import
cheap and unit-testable without the ML stack installed.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

Box = Tuple[float, float, float, float]


def area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou_xyxy(a: Box, b: Box) -> float:
    """Intersection-over-Union of two axis-aligned boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Box, b: Box) -> float:
    """Euclidean distance between box centres, in normalised units."""
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner` covered by `outer` — robust to facing-count mismatch."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = area(inner)
    return inter / denom if denom > 0 else 0.0


def cluster_rows(boxes: Sequence[Box], band_tolerance: float = 0.05) -> List[List[int]]:
    """Group box indices into shelf rows by y-centre proximity.

    Detections are sorted top-to-bottom; a new row starts whenever the vertical
    gap to the running row centroid exceeds `band_tolerance`. This is the
    "bounding-box spatial sorting" step: rows first, then left-to-right ordering.
    """
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda i: center(boxes[i])[1])
    rows: List[List[int]] = []
    current: List[int] = [order[0]]
    running_y = center(boxes[order[0]])[1]

    for idx in order[1:]:
        cy = center(boxes[idx])[1]
        if abs(cy - running_y) <= band_tolerance:
            current.append(idx)
            running_y = sum(center(boxes[i])[1] for i in current) / len(current)
        else:
            rows.append(current)
            current = [idx]
            running_y = cy
    rows.append(current)
    return rows


def sort_detections_by_shelf_row(
    boxes: Sequence[Box], band_tolerance: float = 0.05
) -> List[int]:
    """Return indices in reading order: top row left→right, then next row."""
    ordered: List[int] = []
    for row in cluster_rows(boxes, band_tolerance):
        ordered.extend(sorted(row, key=lambda i: center(boxes[i])[0]))
    return ordered


def greedy_match(
    expected: Sequence[Box],
    observed: Sequence[Box],
    iou_threshold: float = 0.5,
    center_threshold: float = 0.08,
    candidate_filter: "Dict[int, set[int]] | None" = None,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Match planogram slots to detections, highest IoU first.

    A pair is admissible when IoU >= `iou_threshold` **or** the centres are
    within `center_threshold` (the latter rescues correctly-placed products whose
    detected box is tighter than the slot rectangle).

    `candidate_filter` optionally restricts which observed indices may match a
    given expected index (used to enforce SKU identity).

    Returns `(matches, unmatched_expected, unmatched_observed)` where `matches`
    maps expected index -> observed index.
    """
    pairs: List[Tuple[float, float, int, int]] = []
    for ei, ebox in enumerate(expected):
        allowed = candidate_filter.get(ei) if candidate_filter is not None else None
        for oi, obox in enumerate(observed):
            if allowed is not None and oi not in allowed:
                continue
            iou = iou_xyxy(ebox, obox)
            dist = center_distance(ebox, obox)
            if iou >= iou_threshold or dist <= center_threshold:
                pairs.append((iou, -dist, ei, oi))

    # Best IoU wins; ties broken by the closer centre (stored negated).
    pairs.sort(key=lambda p: (p[0], p[1]), reverse=True)

    matches: Dict[int, int] = {}
    used_expected: set[int] = set()
    used_observed: set[int] = set()
    for _iou, _neg_dist, ei, oi in pairs:
        if ei in used_expected or oi in used_observed:
            continue
        matches[ei] = oi
        used_expected.add(ei)
        used_observed.add(oi)

    unmatched_expected = [i for i in range(len(expected)) if i not in used_expected]
    unmatched_observed = [i for i in range(len(observed)) if i not in used_observed]
    return matches, unmatched_expected, unmatched_observed


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
