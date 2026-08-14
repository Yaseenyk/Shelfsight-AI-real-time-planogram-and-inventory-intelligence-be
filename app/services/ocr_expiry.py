"""EasyOCR expiry-date extraction.

Two entry points:
- `read_image()` — full OCR pass over packaging crops (needs easyocr).
- `parse_texts()` — pure regex/normalisation path, dependency-free, used by the
  API's text endpoint and by the CER/WER benchmark.
"""

from __future__ import annotations

import time
from datetime import date
from threading import Lock
from typing import Any, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import ExpiryStatus
from app.schemas.common import BoundingBox
from app.schemas.expiry import ExpiryExtraction
from app.utils.dates import classify_status, parse_expiry_date

logger = get_logger(__name__)


class ExpiryOCRService:
    def __init__(self, languages: Optional[Sequence[str]] = None, gpu: Optional[bool] = None) -> None:
        self.languages = list(languages or settings.OCR_LANGUAGES)
        self.gpu = settings.OCR_GPU if gpu is None else gpu
        self._reader: Any = None
        self._lock = Lock()

    @property
    def is_ready(self) -> bool:
        return self._reader is not None

    def load(self) -> bool:
        if self._reader is not None:
            return True
        with self._lock:
            if self._reader is not None:
                return True
            try:
                import easyocr  # noqa: PLC0415
            except ImportError:
                logger.warning("easyocr not installed — image OCR disabled")
                return False
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR reader ready (langs=%s, gpu=%s)", self.languages, self.gpu)
            return True

    def read_image(
        self, image: Any, reference_date: Optional[date] = None
    ) -> Tuple[List[ExpiryExtraction], float]:
        """OCR an image and parse every text line into an expiry candidate."""
        if not self.load():
            return [], 0.0

        started = time.perf_counter()
        raw_results = self._reader.readtext(image)
        latency_ms = (time.perf_counter() - started) * 1000.0

        extractions: List[ExpiryExtraction] = []
        for polygon, text, confidence in raw_results:
            if confidence < settings.OCR_MIN_CONFIDENCE:
                continue
            extraction = parse_single(text, reference_date)
            extraction.ocr_confidence = float(confidence)
            extraction.bbox = _polygon_to_bbox(polygon)
            extraction.latency_ms = latency_ms / max(1, len(raw_results))
            extractions.append(extraction)

        # Surface the most decisive read first (expired > near > valid).
        priority = {
            ExpiryStatus.EXPIRED: 0,
            ExpiryStatus.NEAR_EXPIRY: 1,
            ExpiryStatus.VALID: 2,
            ExpiryStatus.UNREADABLE: 3,
        }
        extractions.sort(key=lambda e: priority[e.status])
        return extractions, latency_ms


def parse_single(text: str, reference_date: Optional[date] = None) -> ExpiryExtraction:
    """Normalise → regex-match → classify a single OCR string."""
    started = time.perf_counter()
    parsed, pattern, normalized = parse_expiry_date(text, dayfirst=settings.EXPIRY_DAYFIRST)
    status, remaining = classify_status(
        parsed, reference_date, settings.EXPIRY_NEAR_THRESHOLD_DAYS
    )
    return ExpiryExtraction(
        raw_text=text,
        normalized_text=normalized,
        matched_pattern=pattern,
        parsed_date=parsed,
        days_remaining=remaining,
        status=ExpiryStatus(status),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def parse_texts(
    texts: Sequence[str], reference_date: Optional[date] = None
) -> List[ExpiryExtraction]:
    return [parse_single(text, reference_date) for text in texts]


def _polygon_to_bbox(polygon: Sequence[Sequence[float]]) -> Optional[BoundingBox]:
    """EasyOCR returns a 4-point polygon in pixels; clamp to a normalised box.

    Coordinates are already normalised when the caller passes a pre-scaled crop;
    values outside [0, 1] are clipped so the schema validator accepts them.
    """
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, IndexError, ValueError):
        return None

    scale_x = max(1.0, max(xs))
    scale_y = max(1.0, max(ys))
    x1, x2 = min(xs) / scale_x, max(xs) / scale_x
    y1, y2 = min(ys) / scale_y, max(ys) / scale_y
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


_service: Optional[ExpiryOCRService] = None


def get_ocr_service() -> ExpiryOCRService:
    global _service
    if _service is None:
        _service = ExpiryOCRService()
    return _service
