"""Fine-tune YOLOv8 on the shelf-product dataset.

    python models/train_detector.py --data models/datasets/shelf.yaml --epochs 100

Ultralytics writes its own results (weights, curves, confusion matrix) into
`runs/detect/<name>/`; this script copies the best checkpoint to
`models/weights/` so the API picks it up without extra configuration, and mirrors
the validation metrics into an `evaluation/reports/` JSON for the paper.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ShelfSight YOLOv8 detector")
    parser.add_argument("--data", required=True, help="dataset YAML (Ultralytics format)")
    parser.add_argument("--weights", default="yolov8n.pt", help="starting checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=settings.DETECTION_IMG_SIZE)
    parser.add_argument("--device", default=settings.DETECTION_DEVICE)
    parser.add_argument("--name", default="shelfsight-detector")
    parser.add_argument("--seed", type=int, default=42, help="fixed for reproducibility")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics missing — pip install -r requirements-ml.txt")
        return 1

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        name=args.name,
        plots=True,
    )

    metrics = model.val()
    best = Path(getattr(results, "save_dir", f"runs/detect/{args.name}")) / "weights" / "best.pt"
    if best.exists():
        settings.DETECTION_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, settings.DETECTION_WEIGHTS)
        logger.info("Deployed %s -> %s", best, settings.DETECTION_WEIGHTS)
    else:
        logger.warning("best.pt not found at %s", best)

    box = getattr(metrics, "box", None)
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "map_50": float(getattr(box, "map50", 0.0)) if box else None,
        "map_50_95": float(getattr(box, "map", 0.0)) if box else None,
        "precision": float(getattr(box, "mp", 0.0)) if box else None,
        "recall": float(getattr(box, "mr", 0.0)) if box else None,
        "weights": str(settings.DETECTION_WEIGHTS),
    }
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = settings.REPORTS_DIR / f"detector_training_{args.name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Training report: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
