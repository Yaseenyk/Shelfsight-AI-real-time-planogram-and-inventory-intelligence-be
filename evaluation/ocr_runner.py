"""Run the real OCR expiry engine over labelled packaging crops.

Ground truth is a **CSV or JSON** file next to the images:

    image,truth_text,truth_date
    exp-001.jpg,EXP 12/09/2026,2026-09-12
    exp-006.jpg,,                       # human could not read it either

    {"reference_date": "2026-08-14",
     "samples": [{"image": "exp-001.jpg", "truth_text": "...", "truth_date": "..."}]}

An empty `truth_date` means *the stamp is genuinely unreadable*. Those rows are
kept, not dropped: they are the denominator that stops the read-rate from being
flattered by quietly discarding hard samples.

The output rows carry `ocr_text` (what the engine read) alongside the labels, in
exactly the shape `evaluation.metrics.ocr.evaluate` consumes — so live OCR and
replayed fixtures share one CER/WER implementation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.core.logging import get_logger
from app.services.ocr_expiry import ExpiryOCRService, OCRError, get_ocr_service
from app.utils.vision import ImageDecodeError, list_images, read_image_file

logger = get_logger(__name__)

GROUND_TRUTH_NAMES = (
    "ground_truth.csv",
    "ground_truth.json",
    "labels.csv",
    "labels.json",
)


def find_ground_truth(images_dir: Path) -> Optional[Path]:
    """Locate a ground-truth file beside (or one level above) the images."""
    directory = Path(images_dir)
    for candidate_dir in (directory, directory.parent):
        for name in GROUND_TRUTH_NAMES:
            candidate = candidate_dir / name
            if candidate.exists():
                return candidate
    return None


def load_ground_truth(path: Path) -> Dict[str, Any]:
    """Read a CSV or JSON label file into `{reference_date, by_image}`."""
    file_path = Path(path)
    if not file_path.exists():
        return {"reference_date": None, "by_image": {}}

    if file_path.suffix.lower() == ".csv":
        return {"reference_date": None, "by_image": _load_csv(file_path)}
    return _load_json(file_path)


def _load_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            logger.error("Ground-truth CSV %s has no header row", path.name)
            return rows
        # Accept a few common column spellings rather than demanding ours.
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        image_col = _first(fields, ("image", "filename", "file", "id"))
        text_col = _first(fields, ("truth_text", "text", "label", "transcript"))
        date_col = _first(fields, ("truth_date", "date", "expiry", "expiry_date"))
        if image_col is None:
            logger.error("Ground-truth CSV %s needs an 'image' column", path.name)
            return rows

        for line in reader:
            key = str(line.get(image_col, "")).strip()
            if not key:
                continue
            rows[key] = {
                "truth_text": (line.get(text_col) or "").strip() if text_col else "",
                "truth_date": (line.get(date_col) or "").strip() or None if date_col else None,
            }
    return rows


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Invalid ground-truth JSON %s: %s", path.name, exc)
        return {"reference_date": None, "by_image": {}}

    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    by_image: Dict[str, Dict[str, Any]] = {}
    for sample in samples or []:
        if not isinstance(sample, dict):
            continue
        key = str(sample.get("image") or sample.get("id") or "").strip()
        if not key:
            continue
        by_image[Path(key).name] = {
            "truth_text": sample.get("truth_text", "") or "",
            "truth_date": sample.get("truth_date"),
        }
    reference = payload.get("reference_date") if isinstance(payload, dict) else None
    return {"reference_date": reference, "by_image": by_image}


def _first(fields: Dict[str, str], names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in fields:
            return fields[name]
    return None


def run_ocr_over_directory(
    images_dir: Path,
    ground_truth: Optional[Path] = None,
    service: Optional[ExpiryOCRService] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """OCR every image and pair the reads with their labels."""
    service = service or get_ocr_service()
    directory = Path(images_dir)

    images = list_images(directory)
    if limit is not None:
        images = images[:limit]
    if not images:
        return {
            "images": 0,
            "samples": [],
            "skipped": [],
            "note": f"No images found in {directory}",
        }

    gt_path = Path(ground_truth) if ground_truth else find_ground_truth(directory)
    truth = load_ground_truth(gt_path) if gt_path else {"reference_date": None, "by_image": {}}
    by_image: Dict[str, Dict[str, Any]] = truth.get("by_image", {})
    reference_date = _as_date(truth.get("reference_date"))

    if not by_image:
        logger.warning(
            "No ground truth found for %s — CER/WER and date precision will be meaningless",
            directory,
        )

    if not service.load():
        raise OCRError(service.load_failure or "EasyOCR unavailable")

    samples: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    latencies: List[float] = []
    variant_usage: Dict[str, int] = {}

    for image_path in images:
        try:
            frame = read_image_file(image_path)
        except ImageDecodeError as exc:
            logger.warning("Skipping %s: %s", image_path.name, exc)
            skipped.append({"image": image_path.name, "reason": str(exc)})
            continue

        try:
            result = service.extract_expiry(frame, reference_date=reference_date)
        except OCRError as exc:
            logger.warning("OCR failed on %s: %s", image_path.name, exc)
            skipped.append({"image": image_path.name, "reason": str(exc)})
            continue

        latencies.append(result.latency_ms)
        if result.variant_used:
            variant_usage[result.variant_used] = variant_usage.get(result.variant_used, 0) + 1

        labels = by_image.get(image_path.name, {})
        best = result.best
        samples.append(
            {
                "id": image_path.name,
                # `ocr_text` is what metrics.ocr scores CER/WER against; use the
                # winning line when one parsed, else everything the engine read.
                "ocr_text": (best.raw_text if best and best.raw_text else result.raw_text),
                "truth_text": labels.get("truth_text", ""),
                "truth_date": labels.get("truth_date"),
                "predicted_date": best.parsed_date.isoformat()
                if best and best.parsed_date
                else None,
                "status": best.status.value if best else "unreadable",
                "matched_pattern": best.matched_pattern if best else None,
                "ocr_confidence": round(best.ocr_confidence, 4)
                if best and best.ocr_confidence
                else None,
                "variant_used": result.variant_used,
                "latency_ms": round(result.latency_ms, 3),
                "labelled": image_path.name in by_image,
            }
        )

    return {
        "images": len(samples),
        "samples": samples,
        "skipped": skipped,
        "latencies_ms": latencies,
        "labelled_images": sum(1 for s in samples if s["labelled"]),
        "ground_truth_file": str(gt_path) if gt_path else None,
        "reference_date": truth.get("reference_date"),
        "variant_usage": variant_usage,
        "model_version": service.version,
    }


def _as_date(value: Any):  # noqa: ANN202
    from datetime import date  # noqa: PLC0415

    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        logger.warning("Ignoring invalid reference_date %r", value)
        return None
