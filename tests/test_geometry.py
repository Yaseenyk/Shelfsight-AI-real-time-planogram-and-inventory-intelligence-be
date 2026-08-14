from app.utils.geometry import (
    center_distance,
    greedy_match,
    iou_xyxy,
    sort_detections_by_shelf_row,
)


def test_iou_identical_boxes_is_one():
    box = (0.1, 0.1, 0.3, 0.3)
    assert iou_xyxy(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou_xyxy((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_half_overlap():
    a = (0.0, 0.0, 0.2, 0.1)
    b = (0.1, 0.0, 0.3, 0.1)
    assert round(iou_xyxy(a, b), 4) == round(1 / 3, 4)


def test_center_distance():
    assert round(center_distance((0.0, 0.0, 0.2, 0.2), (0.3, 0.0, 0.5, 0.2)), 4) == 0.3


def test_reading_order_groups_rows_then_columns():
    boxes = [
        (0.60, 0.05, 0.75, 0.20),  # top-right
        (0.05, 0.50, 0.20, 0.65),  # bottom-left
        (0.05, 0.06, 0.20, 0.21),  # top-left (slightly offset row)
        (0.60, 0.51, 0.75, 0.66),  # bottom-right
    ]
    assert sort_detections_by_shelf_row(boxes, band_tolerance=0.05) == [2, 0, 1, 3]


def test_greedy_match_prefers_highest_iou():
    expected = [(0.0, 0.0, 0.2, 0.2)]
    observed = [(0.15, 0.0, 0.35, 0.2), (0.01, 0.01, 0.21, 0.21)]
    matches, unmatched_e, unmatched_o = greedy_match(expected, observed)
    assert matches == {0: 1}
    assert unmatched_e == []
    assert unmatched_o == [0]


def test_greedy_match_respects_candidate_filter():
    expected = [(0.0, 0.0, 0.2, 0.2)]
    observed = [(0.0, 0.0, 0.2, 0.2)]
    matches, unmatched_e, _ = greedy_match(expected, observed, candidate_filter={0: set()})
    assert matches == {}
    assert unmatched_e == [0]
