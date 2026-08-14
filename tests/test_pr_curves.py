"""Precision-recall curve construction for the publication figures."""

from __future__ import annotations

import pytest

from evaluation.metrics.classification import pr_curve_from_probabilities
from evaluation.metrics.detection import pr_curve


# ------------------------------------------------------------- detection --
def test_perfect_detector_curve_holds_precision_at_one():
    predictions = [
        {"boxes": [[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]],
         "labels": [0, 0], "scores": [0.9, 0.8]}
    ]
    targets = [{"boxes": [[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]], "labels": [0, 0]}]

    curves = pr_curve(predictions, targets)
    points = curves["class 0"]
    assert [round(p, 3) for _r, p in points] == [1.0, 1.0]
    assert points[-1][0] == pytest.approx(1.0)  # full recall reached


def test_false_positive_drags_precision_down():
    predictions = [
        {"boxes": [[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 0.9, 0.9]],
         "labels": [0, 0], "scores": [0.9, 0.7]}
    ]
    targets = [{"boxes": [[0.0, 0.0, 0.2, 0.2]], "labels": [0]}]

    points = pr_curve(predictions, targets)["class 0"]
    assert points[0][1] == pytest.approx(1.0)   # the confident hit
    assert points[1][1] == pytest.approx(0.5)   # then the false positive


def test_curve_is_ordered_by_descending_score():
    predictions = [
        {"boxes": [[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]],
         "labels": [0, 0], "scores": [0.3, 0.95]}
    ]
    targets = [{"boxes": [[0.5, 0.5, 0.7, 0.7]], "labels": [0]}]

    points = pr_curve(predictions, targets)["class 0"]
    # The 0.95 detection is the true positive, so precision starts at 1.0.
    assert points[0][1] == pytest.approx(1.0)


def test_separate_curve_per_class():
    predictions = [
        {"boxes": [[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]],
         "labels": [0, 1], "scores": [0.9, 0.8]}
    ]
    targets = [{"boxes": [[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]], "labels": [0, 1]}]

    curves = pr_curve(predictions, targets)
    assert set(curves) == {"class 0", "class 1"}


def test_class_with_no_ground_truth_is_omitted():
    """A class with no positives has no meaningful curve — drawing one implies data."""
    predictions = [{"boxes": [[0.0, 0.0, 0.2, 0.2]], "labels": [7], "scores": [0.9]}]
    targets = [{"boxes": [], "labels": []}]
    assert pr_curve(predictions, targets) == {}


def test_empty_input_yields_no_curves():
    assert pr_curve([], []) == {}


# --------------------------------------------------------- classification --
def test_perfect_classifier_reaches_average_precision_one():
    y_true = ["fresh", "ripening", "spoiled"]
    probabilities = [
        {"fresh": 0.98, "ripening": 0.01, "spoiled": 0.01},
        {"fresh": 0.01, "ripening": 0.97, "spoiled": 0.02},
        {"fresh": 0.01, "ripening": 0.02, "spoiled": 0.97},
    ]
    curves, average_precision = pr_curve_from_probabilities(
        y_true, probabilities, ["fresh", "ripening", "spoiled"]
    )
    assert set(curves) == {"fresh", "ripening", "spoiled"}
    for value in average_precision.values():
        assert value == pytest.approx(1.0)


def test_confused_classifier_scores_below_one():
    y_true = ["fresh", "spoiled"]
    probabilities = [
        {"fresh": 0.4, "ripening": 0.3, "spoiled": 0.3},
        {"fresh": 0.6, "ripening": 0.2, "spoiled": 0.2},  # wrong, and confident
    ]
    _curves, average_precision = pr_curve_from_probabilities(
        y_true, probabilities, ["fresh", "ripening", "spoiled"]
    )
    assert average_precision["fresh"] < 1.0


def test_absent_class_produces_no_curve():
    """The empty `ripening` row in a binary dataset must not draw a flat line."""
    y_true = ["fresh", "spoiled"]
    probabilities = [
        {"fresh": 0.9, "ripening": 0.05, "spoiled": 0.05},
        {"fresh": 0.1, "ripening": 0.05, "spoiled": 0.85},
    ]
    curves, average_precision = pr_curve_from_probabilities(
        y_true, probabilities, ["fresh", "ripening", "spoiled"]
    )
    assert "ripening" not in curves
    assert "ripening" not in average_precision


def test_recall_is_monotonic_and_bounded():
    y_true = ["fresh"] * 3 + ["spoiled"] * 2
    probabilities = [
        {"fresh": 0.9, "spoiled": 0.1},
        {"fresh": 0.7, "spoiled": 0.3},
        {"fresh": 0.4, "spoiled": 0.6},
        {"fresh": 0.3, "spoiled": 0.7},
        {"fresh": 0.2, "spoiled": 0.8},
    ]
    curves, _ = pr_curve_from_probabilities(y_true, probabilities, ["fresh", "spoiled"])
    recalls = [r for r, _p in curves["fresh"]]
    assert recalls == sorted(recalls)
    assert recalls[0] >= 0.0 and recalls[-1] <= 1.0


def test_figure_is_written(tmp_path):  # noqa: ANN001
    pytest.importorskip("matplotlib")
    from evaluation.metrics.plotting import plot_pr_curve

    out = plot_pr_curve(
        {"fresh": [(0.5, 1.0), (1.0, 0.8)], "spoiled": [(0.5, 0.9), (1.0, 0.7)]},
        tmp_path / "pr.png",
        title="test",
        average_precision={"fresh": 0.9, "spoiled": 0.8},
    )
    assert out.exists()
    assert out.with_suffix(".pdf").exists()  # vector copy for LaTeX
