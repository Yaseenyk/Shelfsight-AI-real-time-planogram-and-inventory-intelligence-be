"""Figure helpers shared by the metric modules.

Figure conventions (IEEE-friendly, print-safe):
- Single-hue sequential colormap for the confusion matrix — never a rainbow;
  a one-hue light→dark ramp survives greyscale printing and CVD.
- One measure per axis, one series per bar chart, so no legend is needed.
- Every cell/bar carries its value as a direct label; the grid stays recessive.
- 300 dpi PNG + vector PDF, because reviewers zoom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# Sequential single hue, light -> dark. Perceptually ordered in greyscale.
SEQUENTIAL_CMAP = "Blues"
INK = "#1f2933"
MUTED = "#6b7280"
BAR_FILL = "#2c6fb5"


def _pyplot():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless: no display needed on the lab box
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def save_figure(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")  # vector for LaTeX
    return out_path


def plot_confusion_matrix(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    out_path: Path,
    title: str = "Confusion matrix",
    normalize: bool = False,
) -> Path:
    """Row-labelled confusion matrix with in-cell counts."""
    plt = _pyplot()
    import numpy as np  # noqa: PLC0415

    data = np.asarray(matrix, dtype=float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 1.6, 1.2 * len(labels) + 1.4))
    image = ax.imshow(data, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=data.max() or 1.0)

    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)

    # Direct labels: flip ink to white once the fill is dark enough to need it.
    threshold = (data.max() or 1.0) * 0.6
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}" if normalize else f"{int(value)}",
                ha="center",
                va="center",
                color="white" if value > threshold else INK,
                fontsize=9,
            )

    bar = fig.colorbar(image, ax=ax, shrink=0.82)
    bar.outline.set_edgecolor(MUTED)
    bar.set_label("Proportion" if normalize else "Samples")
    ax.grid(False)
    return save_figure(fig, out_path)


#: Categorical hues for multi-series curves, in fixed assignment order. Chosen to
#: stay separable in greyscale (distinct lightness) and under common CVD, since
#: these figures are printed. Never cycled — a 5th class folds into "other".
SERIES_COLORS = ("#2c6fb5", "#c2571a", "#3f7d4e", "#7a4fa3")
SERIES_DASHES = ((), (5, 2), (1, 1.5), (6, 2, 1, 2))


def plot_pr_curve(
    series: Dict[str, Sequence[Tuple[float, float]]],
    out_path: Path,
    title: str = "Precision-recall",
    average_precision: Optional[Dict[str, float]] = None,
) -> Path:
    """Precision-recall curves, one line per class.

    `series` maps a label to an ordered list of `(recall, precision)` points.
    Identity is carried by colour **and** dash pattern plus a legend, so the
    figure survives greyscale printing — a colour-only legend is unreadable in a
    printed IEEE column.
    """
    plt = _pyplot()

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for index, (label, points) in enumerate(sorted(series.items())):
        if not points:
            continue
        recalls = [float(r) for r, _p in points]
        precisions = [float(p) for _r, p in points]
        legend_label = label
        if average_precision and label in average_precision:
            legend_label = f"{label} (AP={average_precision[label]:.3f})"
        ax.plot(
            recalls,
            precisions,
            linewidth=2.0,
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            dashes=SERIES_DASHES[index % len(SERIES_DASHES)],
            label=legend_label,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    if series:
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    return save_figure(fig, out_path)


def plot_metric_bars(
    metrics: Dict[str, float],
    out_path: Path,
    title: str,
    xlabel: str = "Score",
    value_format: str = "{:.3f}",
    upper: Optional[float] = 1.0,
) -> Path:
    """Horizontal single-series bar chart — one hue, values labelled at the ends."""
    plt = _pyplot()

    names = list(metrics.keys())
    values = [metrics[name] for name in names]
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(names) + 1.3))

    positions = range(len(names))
    ax.barh(list(positions), values, height=0.55, color=BAR_FILL)
    ax.set_yticks(list(positions), names)
    ax.invert_yaxis()  # first metric on top
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if upper is not None:
        ax.set_xlim(0, max(upper, max(values) * 1.15 if values else upper))
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)

    span = ax.get_xlim()[1] or 1.0
    for pos, value in zip(positions, values):
        ax.text(value + span * 0.012, pos, value_format.format(value), va="center", fontsize=9)

    return save_figure(fig, out_path)
