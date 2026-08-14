"""Final evaluation on held-out test splits, for the paper's results table.

    python tools/evaluate_holdout.py detector --data models/datasets/shelf.yaml
    python tools/evaluate_holdout.py freshness --data-dir data/freshness

Why this exists separately from training: the numbers printed during a training
run are computed on the **validation** split, which drives checkpoint selection.
Quoting them as final results reports the best of N attempts rather than
generalisation. These commands evaluate the chosen checkpoint once, on data that
influenced nothing.

Both write JSON into `docs/publication_metrics/` alongside the figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

PUBLICATION_DIR = Path(__file__).resolve().parents[1] / "docs" / "publication_metrics"


def evaluate_detector(
    data_yaml: Path, weights: Optional[Path] = None, imgsz: int = 480
) -> Dict[str, Any]:
    """Run YOLOv8 validation on the `test` split."""
    from ultralytics import YOLO  # noqa: PLC0415

    weights_path = Path(weights or settings.DETECTION_WEIGHTS)
    if not weights_path.exists():
        raise FileNotFoundError(f"Detector weights not found: {weights_path}")

    logger.info("Evaluating %s on the TEST split of %s", weights_path.name, data_yaml)
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",  # the split no checkpoint decision ever saw
        imgsz=imgsz,
        device=settings.DETECTION_DEVICE,
        verbose=False,
    )

    box = metrics.box
    result: Dict[str, Any] = {
        "weights": str(weights_path),
        "data": str(data_yaml),
        "split": "test",
        "imgsz": imgsz,
        "map_50": round(float(box.map50), 4),
        "map_50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "f1": round(2 * float(box.mp) * float(box.mr) / (float(box.mp) + float(box.mr)), 4)
        if (float(box.mp) + float(box.mr)) > 0
        else 0.0,
        "speed_ms": {k: round(float(v), 3) for k, v in (metrics.speed or {}).items()},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Record how many images the figures describe. Without it the table caption
    # downstream fell back to a hard-coded count left over from an earlier,
    # much smaller dataset -- a specific wrong number, which reads as
    # authoritative in a way that a missing one does not.
    images = _count_split_images(data_yaml, "test")
    if images is not None:
        result["images"] = images

    # Per-class rows for Table I: the mean alone hides which SKUs the detector
    # actually fails on, which is the interesting part of a 19-class result.
    try:
        names = model.names
        per_class: Dict[str, Dict[str, float]] = {}
        for position, cls in enumerate(box.ap_class_index):
            label = str(names[int(cls)])
            per_class[label] = {
                "precision": round(float(box.p[position]), 4),
                "recall": round(float(box.r[position]), 4),
                "map_50": round(float(box.ap50[position]), 4),
                "map_50_95": round(float(box.ap[position]), 4),
            }
        result["per_class"] = per_class
    except Exception as exc:  # noqa: BLE001 - attribute shapes vary across versions
        logger.warning("Per-class detection metrics unavailable: %s", exc)

    result["pr_curve"] = _extract_pr_curve(box, model)
    return result


def _extract_pr_curve(box: Any, model: Any) -> Dict[str, List[List[float]]]:
    """Pull per-class precision-recall points out of the Ultralytics metrics.

    Returned as `{label: [[recall, precision], ...]}` so the figure script needs
    no Ultralytics import. Curves are subsampled to ~200 points: a 1000-point
    polyline bloats the vector PDF for no visible gain in an IEEE column.
    """
    curves: Dict[str, List[List[float]]] = {}
    try:
        recalls = [float(v) for v in box.px]  # x axis shared by all classes
        precision_matrix = box.prec_values  # (nc, 1000)
        names = model.names
        step = max(1, len(recalls) // 200)

        for position, cls in enumerate(box.ap_class_index):
            label = str(names[int(cls)])
            precisions = [float(v) for v in precision_matrix[position]]
            curves[label] = [
                [round(recalls[i], 4), round(precisions[i], 4)]
                for i in range(0, len(recalls), step)
            ]
    except Exception as exc:  # noqa: BLE001 - optional decoration, never fatal
        logger.warning("PR curve data unavailable: %s", exc)
    return curves


def evaluate_freshness(
    data_dir: Path, weights: Optional[Path] = None, batch_size: int = 32
) -> Dict[str, Any]:
    """Classify the held-out `test` folder and score it."""
    from app.services.freshness import FreshnessService  # noqa: PLC0415
    from evaluation.freshness_runner import run_classifier_over_directory  # noqa: PLC0415
    from evaluation.metrics import classification  # noqa: PLC0415
    from evaluation.metrics.plotting import plot_pr_curve  # noqa: PLC0415

    test_dir = Path(data_dir) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"No test split at {test_dir}. Re-run the curator with --test-split."
        )

    service = FreshnessService(weights=weights) if weights else None
    run = run_classifier_over_directory(test_dir, service=service, batch_size=batch_size)
    samples = run["samples"]
    if not samples:
        raise RuntimeError(f"No images classified under {test_dir}")

    y_true = [s["truth_label"] for s in samples]
    y_pred = [s["predicted_label"] for s in samples]
    labels = list(settings.FRESHNESS_CLASSES)

    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    scored = classification.evaluate(
        y_true, y_pred, labels=labels, report_dir=PUBLICATION_DIR, prefix="freshness_test"
    )

    probabilities = [s.get("probabilities") for s in samples if s.get("probabilities")]
    if len(probabilities) == len(samples):
        curves, average_precision = classification.pr_curve_from_probabilities(
            y_true, probabilities, labels
        )
        if curves:
            scored["average_precision"] = average_precision
            scored["pr_curve_figure"] = str(
                plot_pr_curve(
                    curves,
                    PUBLICATION_DIR / "freshness_test_pr_curve.png",
                    title="Freshness precision-recall (held-out test split)",
                    average_precision=average_precision,
                )
            )

    latencies = run.get("latencies_ms") or []
    if latencies:
        from evaluation.metrics import detection as detection_metrics  # noqa: PLC0415

        scored["latency"] = detection_metrics.latency_stats(latencies)

    scored["split"] = "test"
    scored["images"] = run["images"]
    scored["model_version"] = run.get("model_version")
    scored["distribution"] = run.get("distribution")
    scored["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    return scored


def _count_split_images(data_yaml: Path, split: str) -> Optional[int]:
    """How many images a split holds, for the figure captions.

    Ultralytics accepts either a directory or a text file listing image paths,
    so both forms are handled. Returns None rather than guessing when the config
    cannot be read: an absent count is honest, a wrong one is not.
    """
    try:
        import yaml  # noqa: PLC0415

        config = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        target = config.get(split)
        if not target:
            return None
        path = Path(target)
        if not path.is_absolute():
            path = Path(config.get("path", data_yaml.parent)) / target
        if path.is_file():
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if path.is_dir():
            return sum(
                1
                for entry in path.iterdir()
                if entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
    except Exception:  # noqa: BLE001 - a caption detail must not fail evaluation
        return None
    return None


def save(name: str, payload: Dict[str, Any]) -> Path:
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    path = PUBLICATION_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate on held-out test splits")
    sub = parser.add_subparsers(dest="command", required=True)

    det = sub.add_parser("detector", help="YOLOv8 mAP on the test split")
    det.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    det.add_argument("--weights", default=None)
    det.add_argument("--imgsz", type=int, default=480)

    fresh = sub.add_parser("freshness", help="classifier metrics on the test split")
    fresh.add_argument("--data-dir", default="data/freshness")
    fresh.add_argument("--weights", default=None)

    both = sub.add_parser("all", help="run both")
    both.add_argument("--data", required=True)
    both.add_argument("--data-dir", default="data/freshness")
    both.add_argument("--imgsz", type=int, default=480)

    args = parser.parse_args(argv)
    results: Dict[str, Any] = {}

    try:
        if args.command in ("detector", "all"):
            results["detection"] = evaluate_detector(
                Path(args.data), getattr(args, "weights", None), args.imgsz
            )
            save("detection_test_metrics.json", results["detection"])

        if args.command in ("freshness", "all"):
            results["freshness"] = evaluate_freshness(
                Path(args.data_dir), getattr(args, "weights", None)
            )
            save("freshness_test_metrics.json", results["freshness"])
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    print(json.dumps(results, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
