"""Publication artefact generation: LaTeX correctness and missing-input handling.

The failure mode that matters here is a *silent* one — a figure set that looks
complete but was built from a stale or absent metrics file. These tests pin the
loud-skip behaviour and the LaTeX escaping that would otherwise break a build
the day before a deadline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.generate_publication_figures import (
    escape_latex,
    generate,
    load_json,
    table_detection,
    table_freshness,
    table_latency,
)

DETECTION = {
    "map_50": 0.9123,
    "map_50_95": 0.8412,
    "precision": 0.9011,
    "recall": 0.8834,
    "imgsz": 480,
    "images": 60,
    "per_class": {
        "Cheetoz 30g & 90g": {
            "precision": 0.95, "recall": 0.93, "map_50": 0.97, "map_50_95": 0.90
        },
        "Maz_Maz chips": {
            "precision": 0.80, "recall": 0.73, "map_50": 0.72, "map_50_95": 0.65
        },
    },
    "pr_curve": {
        "Cheetoz 30g & 90g": [[0.0, 1.0], [0.5, 0.95], [1.0, 0.80]],
        "Maz_Maz chips": [[0.0, 1.0], [0.5, 0.70], [1.0, 0.40]],
    },
}

FRESHNESS = {
    "labels": ["fresh", "ripening", "spoiled"],
    "top1_accuracy": 0.8871,
    "f1_macro": 0.8654,
    "f1_micro": 0.8871,
    "confusion_matrix": [[1050, 120, 42], [88, 430, 84], [60, 95, 2091]],
    "per_class": {
        "fresh": {"precision": 0.876, "recall": 0.866, "f1-score": 0.871, "support": 1212},
        "ripening": {"precision": 0.667, "recall": 0.714, "f1-score": 0.690, "support": 602},
        "spoiled": {"precision": 0.943, "recall": 0.931, "f1-score": 0.937, "support": 2246},
    },
}

EXPORT = {
    "latency": {"runs": 20, "pytorch_ms": 24.27, "onnx_ms": 3.41, "speedup": 7.12},
    "components": {"YOLOv8n": {"pytorch_ms": 138.4, "onnx_ms": 61.2, "speedup": 2.26}},
}


def _seed(directory: Path, **payloads) -> None:  # noqa: ANN003
    names = {
        "detection": "detection_test_metrics.json",
        "freshness": "freshness_test_metrics.json",
        "export": "export_manifest.json",
    }
    for key, payload in payloads.items():
        (directory / names[key]).write_text(json.dumps(payload), encoding="utf-8")


# ------------------------------------------------------------------ LaTeX --
def test_latex_escapes_characters_that_break_a_build():
    assert escape_latex("Cheetoz 30g & 90g") == r"Cheetoz 30g \& 90g"
    assert escape_latex("Maz_Maz") == r"Maz\_Maz"
    assert escape_latex("100%") == r"100\%"


def test_detection_table_lists_classes_and_a_bold_mean():
    latex = table_detection(DETECTION)
    assert r"\begin{table}" in latex and r"\end{table}" in latex
    assert r"Cheetoz 30g \& 90g" in latex          # escaped
    assert r"\textbf{Overall (mean)}" in latex
    assert r"\textbf{0.912}" in latex              # the mAP@0.5 mean
    assert "mAP@0.5:0.95" in latex
    assert latex.count(r"\\") >= 4                 # header + 2 classes + mean


def test_detection_table_orders_classes_by_map():
    rows = table_detection(DETECTION).splitlines()
    first = next(i for i, line in enumerate(rows) if "Cheetoz" in line)
    second = next(i for i, line in enumerate(rows) if "Maz" in line)
    assert first < second  # 0.97 before 0.72


def test_freshness_table_has_all_three_classes_and_support():
    latex = table_freshness(FRESHNESS)
    for label in ("Fresh", "Ripening", "Spoiled"):
        assert label in latex
    assert "1212" in latex and "602" in latex and "2246" in latex
    assert r"\textbf{4060}" in latex          # total support
    assert "Top-1 accuracy 0.887" in latex


def test_latency_table_reports_speedup_multipliers():
    latex = table_latency(EXPORT)
    assert r"7.12$\times$" in latex
    assert r"2.26$\times$" in latex
    assert "24.27" in latex and "3.41" in latex


# ---------------------------------------------------- missing-input policy --
def test_missing_inputs_are_reported_not_silently_skipped(tmp_path: Path):
    result = generate(tmp_path)
    assert set(result["missing"]) == {
        "detection_test_metrics.json",
        "freshness_test_metrics.json",
        "export_manifest.json",
    }
    assert result["tables"] == []
    assert result["figures"] == []


def test_partial_inputs_still_produce_what_they_can(tmp_path: Path):
    _seed(tmp_path, freshness=FRESHNESS)
    result = generate(tmp_path, tables_only=True)

    assert "table_ii_freshness" in result["tables"]
    assert "detection_test_metrics.json" in result["missing"]
    assert (tmp_path / "ieee_tables.tex").exists()


def test_malformed_json_is_reported_not_crashing(tmp_path: Path):
    (tmp_path / "detection_test_metrics.json").write_text("{not json", encoding="utf-8")
    assert load_json("detection_test_metrics.json", tmp_path) is None
    assert generate(tmp_path)["tables"] == []


# --------------------------------------------------------------- figures --
def test_figures_are_written_as_png_and_vector_pdf(tmp_path: Path):
    pytest.importorskip("matplotlib")
    _seed(tmp_path, detection=DETECTION, freshness=FRESHNESS, export=EXPORT)

    result = generate(tmp_path)
    assert result["missing"] == []

    for name in (
        "freshness_confusion_matrix",
        "detection_pr_curve",
        "latency_benchmark",
    ):
        assert (tmp_path / f"{name}.png").exists(), name
        assert (tmp_path / f"{name}.pdf").exists(), name  # vector copy for LaTeX


def test_combined_tex_file_holds_all_three_tables(tmp_path: Path):
    _seed(tmp_path, detection=DETECTION, freshness=FRESHNESS, export=EXPORT)
    generate(tmp_path, tables_only=True)

    latex = (tmp_path / "ieee_tables.tex").read_text(encoding="utf-8")
    assert latex.count(r"\begin{table}") == 3
    assert r"\label{tab:detection}" in latex
    assert r"\label{tab:freshness}" in latex
    assert r"\label{tab:latency}" in latex
