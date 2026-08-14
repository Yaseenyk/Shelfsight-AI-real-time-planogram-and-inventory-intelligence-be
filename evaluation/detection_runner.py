"""Run the real `YoloDetector` over a directory of frames for benchmarking.

Labels use the standard **YOLO txt format** — one `.txt` per image, one line per
object, `class_id cx cy w h` with all four geometry values normalised to [0, 1].
That is what Roboflow/CVAT/labelImg export, so an annotated dataset drops in
without a conversion step.

Images with no matching label file are still inferred (they contribute latency
and, if `--strict-labels` is off, an empty ground truth) — that keeps a quick
smoke run possible before any annotation exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logging import get_logger
from app.services.detection import DetectionError, YoloDetector, get_detector
from app.utils.vision import ImageDecodeError, list_images, read_image_file

logger = get_logger(__name__)

LABEL_SUFFIX = ".txt"


def yolo_line_to_xyxy(parts: Sequence[str]) -> Optional[Tuple[int, List[float]]]:
    """`class cx cy w h` (normalised) → `(class_id, [x1, y1, x2, y2])`."""
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        cx, cy, width, height = (float(v) for v in parts[1:5])
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None

    x1, y1 = cx - width / 2.0, cy - height / 2.0
    x2, y2 = cx + width / 2.0, cy + height / 2.0
    clipped = [min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2)]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return class_id, clipped


def load_labels(path: Path) -> Dict[str, List[Any]]:
    """Parse one YOLO label file into `{"boxes": [...], "labels": [...]}`."""
    boxes: List[List[float]] = []
    labels: List[int] = []
    if not path.exists():
        return {"boxes": boxes, "labels": labels}

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = yolo_line_to_xyxy(line.split())
        if parsed is None:
            logger.warning("Malformed label at %s:%d — skipped", path.name, line_no)
            continue
        class_id, box = parsed
        labels.append(class_id)
        boxes.append(box)
    return {"boxes": boxes, "labels": labels}


def resolve_label_path(image: Path, labels_dir: Optional[Path]) -> Path:
    """Find the label file for an image: explicit dir, sibling `labels/`, or beside it."""
    if labels_dir is not None:
        return labels_dir / f"{image.stem}{LABEL_SUFFIX}"

    # Ultralytics convention: .../images/frame.jpg -> .../labels/frame.txt
    if image.parent.name == "images":
        sibling = image.parent.parent / "labels" / f"{image.stem}{LABEL_SUFFIX}"
        if sibling.exists():
            return sibling
    return image.with_suffix(LABEL_SUFFIX)


def run_detector_over_directory(
    images_dir: Path,
    labels_dir: Optional[Path] = None,
    detector: Optional[YoloDetector] = None,
    conf: Optional[float] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Infer over every image and return prediction/target lists plus diagnostics.

    The returned `predictions`/`targets` feed `evaluation.metrics.detection`
    unchanged, so the live path and the fixture path share one metric
    implementation.
    """
    detector = detector or get_detector()
    images = list_images(images_dir)
    if limit is not None:
        images = images[:limit]

    if not images:
        return {
            "images": 0,
            "predictions": [],
            "targets": [],
            "skipped": [],
            "labelled_frames": 0,
            "note": f"No images found in {images_dir}",
        }

    if not detector.load():
        raise DetectionError(detector.load_failure or "Detector unavailable")

    predictions: List[Dict[str, Any]] = []
    targets: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    labelled_frames = 0

    for image_path in images:
        try:
            frame = read_image_file(image_path)
        except ImageDecodeError as exc:
            # One unreadable file must not abort a 500-frame benchmark run.
            logger.warning("Skipping %s: %s", image_path.name, exc)
            skipped.append({"image": image_path.name, "reason": str(exc)})
            continue

        try:
            result = detector.predict_with_metrics(frame, conf=conf)
        except DetectionError as exc:
            logger.warning("Detection failed on %s: %s", image_path.name, exc)
            skipped.append({"image": image_path.name, "reason": str(exc)})
            continue

        predictions.append(
            {
                "frame_id": image_path.name,
                "boxes": [list(d.bbox.as_tuple()) for d in result.detections],
                "labels": [d.class_id for d in result.detections],
                "scores": [d.confidence for d in result.detections],
                "class_names": [d.class_name for d in result.detections],
                "latency_ms": round(result.latency_ms, 3),
                "inference_ms": round(result.inference_ms, 3),
                "postprocess_ms": round(result.postprocess_ms, 3),
                "image_width": result.image_width,
                "image_height": result.image_height,
            }
        )

        label_path = resolve_label_path(image_path, labels_dir)
        target = load_labels(label_path)
        if target["boxes"]:
            labelled_frames += 1
        targets.append({"frame_id": image_path.name, **target})

    return {
        "images": len(predictions),
        "predictions": predictions,
        "targets": targets,
        "skipped": skipped,
        "labelled_frames": labelled_frames,
        "model_version": detector.version,
        "class_names": detector.class_names,
    }
