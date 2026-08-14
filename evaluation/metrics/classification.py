"""Freshness-classification metrics: Top-1 accuracy, macro/micro F1, confusion matrix.

Uses scikit-learn when available and falls back to a pure-Python implementation
so the harness never dies mid-experiment; the report records which backend ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from evaluation.metrics.plotting import plot_confusion_matrix


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    report_dir: Optional[Path] = None,
    prefix: str = "freshness",
) -> Dict[str, object]:
    """Return Top-1 accuracy, F1 scores, per-class report and confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        return {"support": 0}

    label_list = list(labels) if labels else sorted({*y_true, *y_pred})
    try:
        result = _sklearn_metrics(y_true, y_pred, label_list)
    except ImportError:
        result = _manual_metrics(y_true, y_pred, label_list)

    result["labels"] = label_list
    result["support"] = len(y_true)

    if report_dir is not None:
        figure = plot_confusion_matrix(
            result["confusion_matrix"],
            label_list,
            Path(report_dir) / f"{prefix}_confusion_matrix.png",
            title="Freshness classification — confusion matrix",
        )
        normalized = plot_confusion_matrix(
            result["confusion_matrix"],
            label_list,
            Path(report_dir) / f"{prefix}_confusion_matrix_normalized.png",
            title="Freshness classification — normalised",
            normalize=True,
        )
        result["figures"] = [str(figure), str(normalized)]

    return result


def _sklearn_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]
) -> Dict[str, object]:
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    return {
        "backend": "sklearn",
        "top1_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)), 4),
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)), 4),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _manual_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]
) -> Dict[str, object]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred):
        if truth in index and pred in index:
            matrix[index[truth]][index[pred]] += 1

    correct = sum(matrix[i][i] for i in range(len(labels)))
    total = sum(sum(row) for row in matrix)

    per_class: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(labels))) - tp
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": sum(matrix[i]),
        }

    accuracy = correct / total if total else 0.0
    return {
        "backend": "manual",
        "top1_accuracy": round(accuracy, 4),
        "f1_macro": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "f1_micro": round(accuracy, 4),  # micro-F1 == accuracy in single-label settings
        "f1_weighted": round(
            sum(
                per_class[label]["f1-score"] * per_class[label]["support"] for label in labels
            )
            / total,
            4,
        )
        if total
        else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
