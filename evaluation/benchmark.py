"""ShelfSight AI — publication benchmark harness.

Runs one or all four evaluation suites and writes a timestamped, reproducible
report bundle (JSON + figures) under `evaluation/reports/<run_id>/`.

    python -m evaluation.benchmark all
    python -m evaluation.benchmark detection --predictions runs/preds.json --targets runs/gt.json
    python -m evaluation.benchmark freshness --labels runs/freshness_labels.json
    python -m evaluation.benchmark ocr --ground-truth data/ground_truth/expiry_ground_truth.json
    python -m evaluation.benchmark compliance \
        --planogram data/planograms/default_planogram.json \
        --ground-truth data/ground_truth/compliance_ground_truth.json

Every report embeds an environment block (package versions, device, thresholds)
so a table in the paper can be traced back to the exact run that produced it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.schemas.planogram import PlanogramDocument
from app.utils.vision import list_images
from evaluation.detection_runner import run_detector_over_directory
from evaluation.freshness_runner import run_classifier_over_directory
from evaluation.metrics import classification, compliance, detection, ocr
from evaluation.metrics.plotting import plot_metric_bars, plot_pr_curve
from evaluation.ocr_runner import run_ocr_over_directory

logger = get_logger(__name__)

GT_DIR = settings.DATA_DIR / "ground_truth"
DEFAULT_EXPIRY_GT = GT_DIR / "expiry_ground_truth.json"
DEFAULT_COMPLIANCE_GT = GT_DIR / "compliance_ground_truth.json"
DEFAULT_PLANOGRAM = settings.PLANOGRAM_DIR / "default_planogram.json"

# Toy fixtures so `benchmark all` exercises every suite on a fresh clone.
# Replace with real runs before quoting any number in the paper.
DEFAULT_DETECTION_PREDICTIONS = GT_DIR / "detection_predictions.example.json"
DEFAULT_DETECTION_TARGETS = GT_DIR / "detection_targets.example.json"
DEFAULT_FRESHNESS_LABELS = GT_DIR / "freshness_labels.example.json"

#: Live-inference directories. When one holds images, that suite runs the real
#: model instead of replaying the JSON fixtures.
DEFAULT_TEST_IMAGES = settings.DATA_DIR / "test_images"  # Phase 1: detection
DEFAULT_FRESHNESS_IMAGES = settings.DATA_DIR / "test_freshness"  # Phase 2: classifier
DEFAULT_EXPIRY_IMAGES = settings.DATA_DIR / "test_expiry"  # Phase 2: OCR


# --- helpers ---------------------------------------------------------------
def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment_block() -> Dict[str, Any]:
    """Version + threshold provenance, embedded in every report."""
    versions: Dict[str, str] = {}
    for package in ("torch", "torchvision", "torchmetrics", "ultralytics", "sklearn", "easyocr"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = "not-installed"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "device": settings.DETECTION_DEVICE,
        "weights": {
            "detector": str(settings.DETECTION_WEIGHTS),
            "detector_present": settings.DETECTION_WEIGHTS.exists(),
            "freshness": str(settings.FRESHNESS_WEIGHTS),
            "freshness_present": settings.FRESHNESS_WEIGHTS.exists(),
        },
        "thresholds": {
            "detection_conf": settings.DETECTION_CONF_THRESHOLD,
            "detection_nms_iou": settings.DETECTION_IOU_NMS,
            "detection_imgsz": settings.DETECTION_IMG_SIZE,
            "detection_agnostic_nms": settings.DETECTION_AGNOSTIC_NMS,
            "compliance_iou": settings.COMPLIANCE_IOU_THRESHOLD,
            "compliance_center_distance": settings.COMPLIANCE_CENTER_DISTANCE_THRESHOLD,
            "expiry_near_days": settings.EXPIRY_NEAR_THRESHOLD_DAYS,
        },
    }


def new_run_dir(run_id: Optional[str] = None) -> Path:
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = settings.REPORTS_DIR / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report(run_dir: Path, name: str, payload: Dict[str, Any]) -> Path:
    target = run_dir / f"{name}.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", target)
    return target


# --- suites ----------------------------------------------------------------
def run_detection(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    """mAP@0.5, mAP@0.5:0.95, precision, recall and inference latency.

    Two modes:
    - **live** (default): run the real YOLOv8 detector over `--images-dir`.
    - **replay**: score pre-computed `--predictions`/`--targets` JSON.
    """
    images_dir = Path(args.images_dir or DEFAULT_TEST_IMAGES)
    use_live = not args.predictions and images_dir.exists() and list_images(images_dir)

    source: Dict[str, Any] = {}
    if use_live:
        logger.info("detection: live inference over %s", images_dir)
        run = run_detector_over_directory(
            images_dir,
            labels_dir=Path(args.labels_dir) if args.labels_dir else None,
            conf=args.conf,
            limit=args.limit,
        )
        predictions = run["predictions"]
        targets = run["targets"]
        source = {
            "mode": "live",
            "images_dir": str(images_dir),
            "frames": run["images"],
            "labelled_frames": run["labelled_frames"],
            "skipped": run["skipped"],
            "model_version": run.get("model_version"),
        }
        if run["skipped"]:
            logger.warning("detection: %d frame(s) skipped — see report", len(run["skipped"]))
        if not predictions:
            return {"skipped": run.get("note", "no frames processed"), "source": source}
        if run["labelled_frames"] == 0:
            # Latency is still real and worth reporting; accuracy is not.
            logger.warning("detection: no labels found — reporting latency only")
            latencies = [p["latency_ms"] for p in predictions]
            return {
                "source": source,
                "latency": detection.latency_stats(latencies),
                "note": (
                    "No YOLO-format labels found; mAP/precision/recall omitted. "
                    f"Place <stem>.txt files in {images_dir}/labels/ to score accuracy."
                ),
            }
    else:
        predictions_path = Path(args.predictions or DEFAULT_DETECTION_PREDICTIONS)
        targets_path = Path(args.targets or DEFAULT_DETECTION_TARGETS)
        if not predictions_path.exists() or not targets_path.exists():
            logger.warning("detection: %s or %s missing, skipping", predictions_path, targets_path)
            return {"skipped": "missing predictions/targets"}
        if not args.predictions:
            logger.warning("detection: using example fixtures — not publication numbers")

        predictions = load_json(predictions_path)
        targets = load_json(targets_path)
        source = {
            "mode": "replay",
            "predictions": str(predictions_path),
            "targets": str(targets_path),
        }

    if len(predictions) != len(targets):
        raise ValueError("prediction and target files must contain the same number of frames")

    result = detection.compute_map(predictions, targets, iou_threshold=args.iou)
    result["source"] = source
    latencies = [p.get("latency_ms") for p in predictions if p.get("latency_ms") is not None]
    result["latency"] = detection.latency_stats(latencies)

    # Per-frame detail keeps a bad frame identifiable after the run.
    write_report(
        run_dir,
        "detection_frames",
        {
            "source": source,
            "frames": [
                {
                    "frame_id": p.get("frame_id"),
                    "detections": len(p.get("boxes", [])),
                    "ground_truth": len(t.get("boxes", [])),
                    "latency_ms": p.get("latency_ms"),
                }
                for p, t in zip(predictions, targets)
            ],
        },
    )

    bars = {
        key: value
        for key, value in (
            ("mAP@0.5", result.get("map_50")),
            ("mAP@0.5:0.95", result.get("map_50_95")),
            ("Precision", result.get("precision")),
            ("Recall", result.get("recall")),
            ("F1", result.get("f1")),
        )
        if value is not None
    }
    if bars:
        result["figure"] = str(
            plot_metric_bars(bars, run_dir / "detection_metrics.png", "Detection performance")
        )

    curves = detection.pr_curve(predictions, targets, iou_threshold=args.iou)
    if curves:
        result["pr_curve_figure"] = str(
            plot_pr_curve(
                curves,
                run_dir / "detection_pr_curve.png",
                title=f"Detection precision-recall (IoU {args.iou})",
            )
        )
    return result


def run_freshness(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    """Top-1 accuracy, macro/micro F1 and the confusion-matrix figure.

    Live mode runs the real classifier over `data/test_freshness/` (labels taken
    from the folder names); replay mode scores a predictions JSON.
    """
    images_dir = Path(args.freshness_dir or DEFAULT_FRESHNESS_IMAGES)
    source: Dict[str, Any] = {}

    if not args.labels and images_dir.exists() and list_images(images_dir):
        logger.info("freshness: live inference over %s", images_dir)
        run = run_classifier_over_directory(
            images_dir,
            class_map_file=Path(args.class_map) if args.class_map else None,
            limit=args.limit,
        )
        samples = run["samples"]
        source = {
            "mode": "live",
            "images_dir": str(images_dir),
            "frames": run["images"],
            "skipped": run["skipped"],
            "model_version": run.get("model_version"),
            "folder_mapping": run.get("folder_mapping"),
        }
        if not samples:
            return {"skipped": run.get("note", "no crops classified"), "source": source}
        write_report(run_dir, "freshness_predictions", {"source": source, "samples": samples})
    else:
        labels_path = Path(args.labels or DEFAULT_FRESHNESS_LABELS)
        if not labels_path.exists():
            logger.warning("freshness: %s missing, skipping", labels_path)
            return {"skipped": "missing labels"}
        if not args.labels:
            logger.warning("freshness: using example fixtures — not publication numbers")

        payload = load_json(labels_path)
        samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
        source = {"mode": "replay", "labels": str(labels_path)}

    y_true = [s["truth_label"] for s in samples]
    y_pred = [s["predicted_label"] for s in samples]

    result = classification.evaluate(
        y_true,
        y_pred,
        labels=list(settings.FRESHNESS_CLASSES),
        report_dir=run_dir,
        prefix="freshness",
    )
    result["source"] = source
    latencies = [s.get("latency_ms") for s in samples if s.get("latency_ms") is not None]
    if latencies:
        result["latency"] = detection.latency_stats(latencies)
    result["figure_metrics"] = str(
        plot_metric_bars(
            {
                "Top-1 accuracy": float(result.get("top1_accuracy", 0.0)),
                "F1 (macro)": float(result.get("f1_macro", 0.0)),
                "F1 (micro)": float(result.get("f1_micro", 0.0)),
                "F1 (weighted)": float(result.get("f1_weighted", 0.0)),
            },
            run_dir / "freshness_metrics.png",
            "Freshness classification",
        )
    )

    # PR curves need per-class scores, which only the live runner records.
    probabilities = [s.get("probabilities") for s in samples if s.get("probabilities")]
    if len(probabilities) == len(samples) and samples:
        curves, average_precision = classification.pr_curve_from_probabilities(
            y_true, probabilities, list(settings.FRESHNESS_CLASSES)
        )
        if curves:
            result["average_precision"] = average_precision
            result["pr_curve_figure"] = str(
                plot_pr_curve(
                    curves,
                    run_dir / "freshness_pr_curve.png",
                    title="Freshness precision-recall (one-vs-rest)",
                    average_precision=average_precision,
                )
            )
    return result


def run_ocr(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    """CER, WER and date-parsing precision against the labelled expiry set.

    Live mode OCRs real packaging crops from `data/test_expiry/`; replay mode
    scores pre-transcribed text from a ground-truth JSON.
    """
    images_dir = Path(args.ocr_dir or DEFAULT_EXPIRY_IMAGES)
    source: Dict[str, Any] = {}

    if not args.ground_truth and images_dir.exists() and list_images(images_dir):
        logger.info("ocr: live OCR over %s", images_dir)
        run = run_ocr_over_directory(images_dir, limit=args.limit)
        samples = run["samples"]
        source = {
            "mode": "live",
            "images_dir": str(images_dir),
            "frames": run["images"],
            "labelled_images": run["labelled_images"],
            "ground_truth_file": run["ground_truth_file"],
            "variant_usage": run["variant_usage"],
            "skipped": run["skipped"],
            "model_version": run.get("model_version"),
        }
        if not samples:
            return {"skipped": run.get("note", "no crops read"), "source": source}
        write_report(run_dir, "ocr_reads", {"source": source, "samples": samples})

        if run["labelled_images"] == 0:
            # Latency and read-rate are real; accuracy without labels is not.
            logger.warning("ocr: no ground truth — reporting latency and read rate only")
            read = sum(1 for s in samples if s.get("predicted_date"))
            return {
                "source": source,
                "support": len(samples),
                "date_read_rate": round(read / len(samples), 4),
                "latency": detection.latency_stats(run["latencies_ms"]),
                "note": (
                    "No ground-truth CSV/JSON found; CER/WER and date precision omitted. "
                    f"Add ground_truth.csv beside {images_dir}."
                ),
            }
    else:
        path = Path(args.ground_truth or DEFAULT_EXPIRY_GT)
        if not path.exists():
            logger.warning("ocr: ground truth %s missing, skipping", path)
            return {"skipped": f"missing {path}"}
        payload = load_json(path)
        samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
        source = {"mode": "replay", "ground_truth": str(path)}

    result = ocr.evaluate(samples, dayfirst=settings.EXPIRY_DAYFIRST)
    result["source"] = source
    latencies = [s.get("latency_ms") for s in samples if s.get("latency_ms") is not None]
    if latencies:
        result["latency"] = detection.latency_stats(latencies)
    result["figure"] = str(
        plot_metric_bars(
            {
                "Date precision": result["date_parsing_precision"],
                "Date recall": result["date_parsing_recall"],
                "1 - CER": round(1 - result["cer"], 4),
                "1 - WER": round(1 - result["wer"], 4),
            },
            run_dir / "ocr_metrics.png",
            "Expiry OCR engine (higher is better)",
        )
    )
    return result


def run_compliance(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    """Spatial alignment accuracy and the discrepancy false-positive rate."""
    planogram_path = Path(args.planogram or DEFAULT_PLANOGRAM)
    gt_path = Path(args.ground_truth or DEFAULT_COMPLIANCE_GT)
    if not planogram_path.exists() or not gt_path.exists():
        logger.warning("compliance: %s or %s missing, skipping", planogram_path, gt_path)
        return {"skipped": "missing planogram or ground truth"}

    document = PlanogramDocument.model_validate(load_json(planogram_path))
    payload = load_json(gt_path)
    frames = payload.get("frames", payload) if isinstance(payload, dict) else payload

    result = compliance.evaluate_frames(document, frames)
    result["planogram_id"] = document.planogram_id
    result["figure"] = str(
        plot_metric_bars(
            {
                "Spatial alignment acc.": result["spatial_alignment_accuracy"],
                "Discrepancy recall": result["discrepancy_recall"],
                "Mean IoU": result["mean_iou"],
                "False-positive rate": result["discrepancy_false_positive_rate"],
            },
            run_dir / "compliance_metrics.png",
            "Planogram compliance engine",
        )
    )
    return result


SUITES = {
    "detection": run_detection,
    "freshness": run_freshness,
    "ocr": run_ocr,
    "compliance": run_compliance,
}


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="ShelfSight AI evaluation harness (IEEE submission metrics)",
    )
    parser.add_argument(
        "suite", choices=[*SUITES.keys(), "all"], help="which evaluation suite to run"
    )
    parser.add_argument("--predictions", help="detection predictions JSON")
    parser.add_argument("--targets", help="detection ground-truth JSON")
    parser.add_argument("--labels", help="freshness labels JSON (truth_label/predicted_label)")
    parser.add_argument("--ground-truth", help="OCR or compliance ground-truth JSON")
    parser.add_argument("--planogram", help="planogram JSON for the compliance suite")
    parser.add_argument("--iou", type=float, default=0.5, help="detection IoU threshold")
    parser.add_argument(
        "--images-dir",
        help=f"frames for live detection (default: {DEFAULT_TEST_IMAGES})",
    )
    parser.add_argument("--labels-dir", help="YOLO-format .txt labels (default: <images>/labels)")
    parser.add_argument("--conf", type=float, help="override the detector confidence threshold")
    parser.add_argument("--limit", type=int, help="cap the number of frames inferred")
    parser.add_argument(
        "--freshness-dir",
        help=f"labelled produce crops for live classification (default: {DEFAULT_FRESHNESS_IMAGES})",
    )
    parser.add_argument(
        "--class-map", help="JSON overriding the freshness folder→label mapping"
    )
    parser.add_argument(
        "--ocr-dir",
        help=f"packaging crops for live OCR (default: {DEFAULT_EXPIRY_IMAGES})",
    )
    parser.add_argument("--run-id", help="override the generated run directory name")
    args = parser.parse_args(argv)

    settings.ensure_directories()
    run_dir = new_run_dir(args.run_id)
    logger.info("Report directory: %s", run_dir)

    selected = list(SUITES) if args.suite == "all" else [args.suite]
    results: Dict[str, Any] = {}
    for name in selected:
        logger.info("Running suite: %s", name)
        try:
            results[name] = SUITES[name](args, run_dir)
        except Exception as exc:  # noqa: BLE001 - one bad suite must not lose the rest
            logger.exception("Suite %s failed", name)
            results[name] = {"error": str(exc)}

    summary = {
        "run_id": run_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suites": selected,
        "environment": environment_block(),
        "results": results,
    }
    write_report(run_dir, "benchmark_report", summary)

    print(json.dumps({k: _headline(v) for k, v in results.items()}, indent=2, default=str))
    return 0


def _headline(result: Dict[str, Any]) -> Dict[str, Any]:
    """Terminal-friendly digest; the full numbers live in the JSON report."""
    keys = (
        "map_50",
        "map_50_95",
        "precision",
        "recall",
        "top1_accuracy",
        "f1_macro",
        "cer",
        "wer",
        "date_parsing_precision",
        "spatial_alignment_accuracy",
        "discrepancy_false_positive_rate",
        "skipped",
        "error",
    )
    return {k: result[k] for k in keys if isinstance(result, dict) and k in result}


if __name__ == "__main__":
    raise SystemExit(main())
