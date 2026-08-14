"""Run the real freshness classifier over a labelled image directory.

Labels come from the folder structure — the same layouts `models/dataset.py`
resolves — so the evaluation set is the dataset's own validation split, with no
separate annotation format to maintain.

    data/test_freshness/
    ├─ fresh/*.jpg
    ├─ ripening/*.jpg
    └─ spoiled/*.jpg

Output feeds `evaluation.metrics.classification` unchanged, so the live path and
the JSON-replay path share one metric implementation and one confusion matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger
from app.services.freshness import (
    FreshnessError,
    FreshnessService,
    get_freshness_service,
)
from app.utils.vision import ImageDecodeError, read_image_file
from models.dataset import CANONICAL_CLASSES, FreshnessDataset

logger = get_logger(__name__)


def run_classifier_over_directory(
    images_dir: Path,
    service: Optional[FreshnessService] = None,
    class_map_file: Optional[Path] = None,
    batch_size: int = 16,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Classify every labelled crop and return truth/prediction pairs.

    Unreadable files are skipped and reported rather than aborting the run — a
    500-image benchmark must not die on one truncated JPEG.
    """
    service = service or get_freshness_service()

    dataset = FreshnessDataset.discover(
        Path(images_dir),
        class_map_file=class_map_file,
        val_split=0.0,  # score everything; this directory *is* the eval set
        classes=CANONICAL_CLASSES,
    )
    items = dataset.train + dataset.val
    if limit is not None:
        items = items[:limit]

    if not items:
        return {
            "images": 0,
            "samples": [],
            "skipped": [],
            "note": f"No labelled images found in {images_dir}",
        }

    if not service.load():
        raise FreshnessError(service.load_failure or "Freshness classifier unavailable")

    samples: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    latencies: List[float] = []

    for start in range(0, len(items), max(1, batch_size)):
        chunk = items[start : start + max(1, batch_size)]
        frames: List[Any] = []
        truths: List[str] = []
        paths: List[Path] = []

        for path, label_index in chunk:
            try:
                frames.append(read_image_file(path))
            except ImageDecodeError as exc:
                logger.warning("Skipping %s: %s", path.name, exc)
                skipped.append({"image": path.name, "reason": str(exc)})
                continue
            truths.append(dataset.classes[label_index])
            paths.append(path)

        if not frames:
            continue

        try:
            results = service.predict_batch(frames)
        except FreshnessError as exc:
            logger.warning("Classification failed for a batch of %d: %s", len(frames), exc)
            skipped.extend({"image": p.name, "reason": str(exc)} for p in paths)
            continue

        for path, truth, result in zip(paths, truths, results):
            latencies.append(result.latency_ms)
            samples.append(
                {
                    "id": path.name,
                    "truth_label": truth,
                    "predicted_label": result.label.value,
                    "confidence": round(result.confidence, 4),
                    "probabilities": result.probabilities,
                    "latency_ms": round(result.latency_ms, 3),
                }
            )

    return {
        "images": len(samples),
        "samples": samples,
        "skipped": skipped,
        "latencies_ms": latencies,
        "model_version": service.version,
        "classes": list(dataset.classes),
        "distribution": dataset.distribution("train"),
        "folder_mapping": [
            {"folder": f.folder_name, "label": f.label, "count": f.count}
            for f in dataset.folders
        ],
    }
