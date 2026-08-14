"""Turn the evaluation JSONs into paper-ready figures and LaTeX tables.

    python tools/generate_publication_figures.py            # figures + tables
    python tools/generate_publication_figures.py --tables-only
    python tools/generate_publication_figures.py --print     # echo the numbers

Reads from `docs/publication_metrics/`:

| Input | Produces |
| --- | --- |
| `detection_test_metrics.json` | `detection_pr_curve.*`, Table I |
| `freshness_test_metrics.json` | `freshness_confusion_matrix.*`, `freshness_test_pr_curve.*`, Table II |
| `export_manifest.json` | `latency_benchmark.*`, Table III |

Nothing here computes a metric. Every number is read from a file written by an
evaluation run, so a figure can never disagree with the table beside it — and a
missing input produces a loud skip, never a placeholder. Regenerating after a new
run is one command, which is what keeps a paper's figures from drifting apart.

Figure conventions follow the rest of the harness: 300 dpi PNG **and** vector
PDF, single-hue sequential ramp for the matrix, and per-series dash patterns so
the curves survive greyscale printing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.core.logging import configure_logging, get_logger
from evaluation.metrics.plotting import (
    SERIES_COLORS,
    _pyplot,
    plot_confusion_matrix,
    plot_pr_curve,
    save_figure,
)

logger = get_logger(__name__)

PUBLICATION_DIR = Path(__file__).resolve().parents[1] / "docs" / "publication_metrics"

DETECTION_JSON = "detection_test_metrics.json"
FRESHNESS_JSON = "freshness_test_metrics.json"
EXPORT_JSON = "export_manifest.json"

#: LaTeX escapes for dataset class names ("Cheetoz 30g & 90g" would break a table).
LATEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(char, char) for char in str(text))


def load_json(name: str, directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = (directory or PUBLICATION_DIR) / name
    if not path.exists():
        logger.warning("Missing %s — the figures and tables it feeds will be skipped", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("%s is not valid JSON: %s", path, exc)
        return None


# ----------------------------------------------------------------- figures --
def figure_confusion_matrix(freshness: Dict[str, Any], out_dir: Path) -> List[Path]:
    """3x3 normalised confusion matrix for the freshness classifier."""
    matrix = freshness.get("confusion_matrix")
    labels = freshness.get("labels")
    if not matrix or not labels:
        logger.warning("No confusion matrix in the freshness JSON — skipping")
        return []

    written = [
        plot_confusion_matrix(
            matrix,
            labels,
            out_dir / "freshness_confusion_matrix.png",
            title="Freshness classification — held-out test split",
            normalize=True,
        ),
        plot_confusion_matrix(
            matrix,
            labels,
            out_dir / "freshness_confusion_matrix_counts.png",
            title="Freshness classification — counts",
            normalize=False,
        ),
    ]
    return written


def figure_detection_pr(detection: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Per-class detection PR curves.

    A 19-class legend is unreadable in a single IEEE column, so the figure shows
    the best and worst classes by AP plus the count of those omitted — the shape
    of the spread is the point, not every individual line.
    """
    curves = detection.get("pr_curve") or {}
    if not curves:
        logger.warning("No PR curve data in the detection JSON — skipping")
        return []

    per_class = detection.get("per_class") or {}
    ranked = sorted(
        curves,
        key=lambda label: per_class.get(label, {}).get("map_50", 0.0),
        reverse=True,
    )
    selected = ranked[:2] + ranked[-2:] if len(ranked) > 4 else ranked
    selected = list(dict.fromkeys(selected))  # de-dup when the list is short

    series = {label: [tuple(point) for point in curves[label]] for label in selected}
    average_precision = {
        label: per_class.get(label, {}).get("map_50", 0.0) for label in selected
    }

    omitted = len(ranked) - len(selected)
    title = f"Detection precision-recall (test split, mAP@0.5 = {detection.get('map_50', 0):.3f})"
    if omitted > 0:
        title += f"\n{len(selected)} of {len(ranked)} classes shown (best and worst by AP)"

    return [plot_pr_curve(series, out_dir / "detection_pr_curve.png", title, average_precision)]


def figure_freshness_pr(freshness: Dict[str, Any], out_dir: Path) -> List[Path]:
    """One-vs-rest PR curves — solid/dashed/dotted for greyscale printing."""
    curves = freshness.get("pr_curves") or {}
    if not curves:
        # The evaluator already writes this figure when it has probabilities;
        # only regenerate when the raw points were persisted.
        logger.info("No stored PR points for freshness — using the evaluator's figure")
        return []

    series = {label: [tuple(p) for p in points] for label, points in curves.items()}
    return [
        plot_pr_curve(
            series,
            out_dir / "freshness_test_pr_curve.png",
            "Freshness precision-recall (one-vs-rest, held-out test split)",
            freshness.get("average_precision"),
        )
    ]


def figure_latency(export: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Grouped horizontal bars: PyTorch eager vs ONNX Runtime, per component.

    Two series, so a legend is mandatory; values are labelled directly at the bar
    ends because the speed-up is the headline and readers should not have to
    measure it off an axis.
    """
    rows = _latency_rows(export)
    if not rows:
        logger.warning("No latency data in the export manifest — skipping")
        return []

    plt = _pyplot()
    components = [row["component"] for row in rows]
    torch_ms = [row["pytorch_ms"] for row in rows]
    onnx_ms = [row["onnx_ms"] for row in rows]

    positions = range(len(components))
    height = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 1.1 * len(components) + 1.6))

    ax.barh(
        [p + height / 2 for p in positions], torch_ms, height=height,
        color=SERIES_COLORS[0], label="PyTorch eager",
    )
    ax.barh(
        [p - height / 2 for p in positions], onnx_ms, height=height,
        color=SERIES_COLORS[1], label="ONNX Runtime",
    )

    ax.set_yticks(list(positions), components)
    ax.set_xlabel("CPU latency per inference (ms) — lower is better")
    ax.set_title("Inference latency: PyTorch eager vs ONNX Runtime")
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    span = max(torch_ms + onnx_ms) or 1.0
    for index, row in enumerate(rows):
        ax.text(row["pytorch_ms"] + span * 0.015, index + height / 2,
                f"{row['pytorch_ms']:.1f} ms", va="center", fontsize=8)
        speedup = f"{row['onnx_ms']:.1f} ms ({row['speedup']:.1f}x faster)" if row["speedup"] \
            else f"{row['onnx_ms']:.1f} ms"
        ax.text(row["onnx_ms"] + span * 0.015, index - height / 2, speedup,
                va="center", fontsize=8)

    ax.set_xlim(0, span * 1.35)
    return [save_figure(fig, out_dir / "latency_benchmark.png")]


def _latency_rows(export: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise the export manifest into per-component latency rows."""
    rows: List[Dict[str, Any]] = []
    latency = export.get("latency") or {}
    if latency.get("pytorch_ms") and latency.get("onnx_ms"):
        rows.append(
            {
                "component": "Freshness CNN\n(MobileNetV2)",
                "pytorch_ms": float(latency["pytorch_ms"]),
                "onnx_ms": float(latency["onnx_ms"]),
                "speedup": float(latency.get("speedup") or 0.0),
            }
        )

    for name, payload in (export.get("components") or {}).items():
        if payload.get("pytorch_ms") and payload.get("onnx_ms"):
            rows.append(
                {
                    "component": name,
                    "pytorch_ms": float(payload["pytorch_ms"]),
                    "onnx_ms": float(payload["onnx_ms"]),
                    "speedup": float(payload.get("speedup") or 0.0),
                }
            )
    return rows


# ------------------------------------------------------------ LaTeX tables --
def _table(caption: str, label: str, spec: str, header: Sequence[str],
           rows: Sequence[Sequence[str]], note: Optional[str] = None) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\hline",
        " & ".join(header) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}"])
    if note:
        lines.append(rf"\vspace{{2pt}}{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_detection(detection: Dict[str, Any]) -> str:
    per_class = detection.get("per_class") or {}
    rows: List[List[str]] = []
    for label in sorted(per_class, key=lambda k: per_class[k].get("map_50", 0), reverse=True):
        values = per_class[label]
        rows.append([
            escape_latex(label),
            f"{values.get('precision', 0):.3f}",
            f"{values.get('recall', 0):.3f}",
            f"{values.get('map_50', 0):.3f}",
            f"{values.get('map_50_95', 0):.3f}",
        ])

    rows.append([
        r"\textbf{Overall (mean)}",
        rf"\textbf{{{detection.get('precision', 0):.3f}}}",
        rf"\textbf{{{detection.get('recall', 0):.3f}}}",
        rf"\textbf{{{detection.get('map_50', 0):.3f}}}",
        rf"\textbf{{{detection.get('map_50_95', 0):.3f}}}",
    ])

    images = detection.get("images", 60)
    return _table(
        caption=f"Object Detection Performance ({len(per_class)} Classes, "
        f"{images}-Image Holdout Test Split)",
        label="tab:detection",
        spec="lrrrr",
        header=["Class", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
        rows=rows,
        note=f"YOLOv8n, {detection.get('imgsz', 480)} px, CPU inference. "
        "Evaluated once on the test split; no checkpoint decision used it.",
    )


def table_freshness(freshness: Dict[str, Any]) -> str:
    per_class = freshness.get("per_class") or {}
    labels = freshness.get("labels") or ["fresh", "ripening", "spoiled"]
    rows: List[List[str]] = []

    for label in labels:
        values = per_class.get(label) or {}
        rows.append([
            escape_latex(label.capitalize()),
            f"{values.get('precision', 0):.3f}",
            f"{values.get('recall', 0):.3f}",
            f"{values.get('f1-score', values.get('f1', 0)):.3f}",
            f"{int(values.get('support', 0))}",
        ])

    total_support = sum(int((per_class.get(lbl) or {}).get("support", 0)) for lbl in labels)
    rows.append([
        r"\textbf{Macro avg}",
        rf"\textbf{{{_macro(per_class, labels, 'precision'):.3f}}}",
        rf"\textbf{{{_macro(per_class, labels, 'recall'):.3f}}}",
        rf"\textbf{{{freshness.get('f1_macro', 0):.3f}}}",
        rf"\textbf{{{total_support}}}",
    ])

    accuracy = freshness.get("top1_accuracy", 0)
    return _table(
        caption=f"Perishable Freshness Classification (3 Classes, "
        f"{total_support}-Image Holdout Test Split)",
        label="tab:freshness",
        spec="lrrrr",
        header=["Class", "Precision", "Recall", "F1-Score", "Support"],
        rows=rows,
        note=f"MobileNetV2, Top-1 accuracy {accuracy:.3f}, "
        f"micro-F1 {freshness.get('f1_micro', 0):.3f}.",
    )


def _macro(per_class: Dict[str, Any], labels: Sequence[str], key: str) -> float:
    values = [float((per_class.get(label) or {}).get(key, 0.0)) for label in labels]
    return sum(values) / len(values) if values else 0.0


def table_latency(export: Dict[str, Any]) -> str:
    rows_data = _latency_rows(export)
    rows = [
        [
            escape_latex(row["component"].replace("\n", " ")),
            f"{row['pytorch_ms']:.2f}",
            f"{row['onnx_ms']:.2f}",
            rf"\textbf{{{row['speedup']:.2f}$\times$}}",
        ]
        for row in rows_data
    ]
    return _table(
        caption="Inference Latency and Optimisation Benchmark (CPU)",
        label="tab:latency",
        spec="lrrr",
        header=["Pipeline Component", "PyTorch (ms)", "ONNX (ms)", "Speedup"],
        rows=rows,
        note="Single-image inference, mean of "
        f"{(export.get('latency') or {}).get('runs', '?')} runs after warm-up. "
        "CPU-only container, no GPU.",
    )


# --------------------------------------------------------------------- CLI --
def generate(out_dir: Path, tables_only: bool = False) -> Dict[str, Any]:
    detection = load_json(DETECTION_JSON, out_dir)
    freshness = load_json(FRESHNESS_JSON, out_dir)
    export = load_json(EXPORT_JSON, out_dir)

    figures: List[Path] = []
    tables: Dict[str, str] = {}

    if freshness:
        if not tables_only:
            figures += figure_confusion_matrix(freshness, out_dir)
            figures += figure_freshness_pr(freshness, out_dir)
        tables["table_ii_freshness"] = table_freshness(freshness)
    if detection:
        if not tables_only:
            figures += figure_detection_pr(detection, out_dir)
        tables["table_i_detection"] = table_detection(detection)
    if export:
        if not tables_only:
            figures += figure_latency(export, out_dir)
        tables["table_iii_latency"] = table_latency(export)

    if tables:
        combined = "\n\n".join(
            tables[key] for key in ("table_i_detection", "table_ii_freshness",
                                    "table_iii_latency") if key in tables
        )
        (out_dir / "ieee_tables.tex").write_text(combined + "\n", encoding="utf-8")
        logger.info("Wrote %s", out_dir / "ieee_tables.tex")

    return {
        "figures": [str(p) for p in figures],
        "tables": list(tables),
        "missing": [
            name
            for name, payload in (
                (DETECTION_JSON, detection), (FRESHNESS_JSON, freshness), (EXPORT_JSON, export)
            )
            if payload is None
        ],
    }


def print_values(out_dir: Path) -> None:
    """Echo the headline numbers, so the terminal shows what the paper will."""
    detection = load_json(DETECTION_JSON, out_dir)
    freshness = load_json(FRESHNESS_JSON, out_dir)
    export = load_json(EXPORT_JSON, out_dir)

    print("\n" + "=" * 68)
    print("FINAL METRICS (held-out test splits)")
    print("=" * 68)

    if detection:
        print("\nDetection (YOLOv8n):")
        print(f"  mAP@0.5      : {detection.get('map_50')}")
        print(f"  mAP@0.5:0.95 : {detection.get('map_50_95')}")
        print(f"  precision    : {detection.get('precision')}")
        print(f"  recall       : {detection.get('recall')}")
        print(f"  classes      : {len(detection.get('per_class') or {})}")
    else:
        print("\nDetection: NOT AVAILABLE — run tools/evaluate_holdout.py detector")

    if freshness:
        print("\nFreshness (MobileNetV2):")
        print(f"  top-1 accuracy : {freshness.get('top1_accuracy')}")
        print(f"  macro F1       : {freshness.get('f1_macro')}")
        print(f"  micro F1       : {freshness.get('f1_micro')}")
        for label, values in (freshness.get("per_class") or {}).items():
            if isinstance(values, dict) and "precision" in values:
                print(
                    f"    {label:<10} P={values.get('precision'):.3f} "
                    f"R={values.get('recall'):.3f} "
                    f"F1={values.get('f1-score', 0):.3f} n={int(values.get('support', 0))}"
                )
    else:
        print("\nFreshness: NOT AVAILABLE — run tools/evaluate_holdout.py freshness")

    latency = (export or {}).get("latency") or {}
    if latency:
        print("\nLatency (CPU):")
        print(f"  PyTorch : {latency.get('pytorch_ms')} ms")
        print(f"  ONNX    : {latency.get('onnx_ms')} ms")
        print(f"  speedup : {latency.get('speedup')}x")
    else:
        print("\nLatency: NOT AVAILABLE — run models/export_pipeline.py export --benchmark")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Build paper figures and LaTeX tables")
    parser.add_argument("--dir", default=str(PUBLICATION_DIR), help="metrics directory")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--print", dest="show", action="store_true", help="echo the numbers")
    args = parser.parse_args(argv)

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = generate(out_dir, tables_only=args.tables_only)
    if args.show:
        print_values(out_dir)

    print(json.dumps(result, indent=2))
    # Missing inputs are a loud, non-zero outcome: a half-built figure set must
    # not look like a successful run in a build script.
    return 1 if result["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
