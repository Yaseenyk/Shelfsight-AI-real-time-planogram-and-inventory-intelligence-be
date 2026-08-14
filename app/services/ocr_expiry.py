"""Packaging OCR → expiry date extraction.

Phase 2 contract:

    service = get_ocr_service()
    result: ExpiryReadResult = service.extract_expiry(bgr_ndarray)
    result.best          # ExpiryExtraction | None
    result.extractions   # every candidate, most decisive first
    result.raw_text      # everything OCR read, for audit

Why this is not "just call EasyOCR"
-----------------------------------
Expiry codes are the worst text on a package: printed by dot-matrix/inkjet heads
as clouds of disconnected dots, onto curved foil, in low contrast, often
light-on-dark. A single preprocessing choice that fixes one of those cases
breaks another. So the service runs an ordered list of **variants** — raw,
contrast-enhanced, Otsu, adaptive+morphological-close, inverted — and stops at
the first that yields a *parseable date* above the early-stop confidence.

The parse itself is delegated to `app/utils/dates.py` (Phase 0), which owns the
glyph repairs (`O→0`, `I→1`, `S→5`) and the seven date-format patterns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import ExpiryStatus
from app.schemas.common import BoundingBox
from app.schemas.expiry import ExpiryExtraction
from app.utils.dates import classify_status, parse_expiry_date
from app.utils.vision import (
    adaptive_threshold,
    describe,
    enhance_contrast,
    load_image,
    morphology,
    otsu_threshold,
    sharpen,
    to_grayscale,
    upscale,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

    Image = NDArray[np.uint8]
else:
    Image = Any

logger = get_logger(__name__)

#: Ordering used to surface the most decisive read first.
STATUS_PRIORITY: Dict[ExpiryStatus, int] = {
    ExpiryStatus.EXPIRED: 0,
    ExpiryStatus.NEAR_EXPIRY: 1,
    ExpiryStatus.VALID: 2,
    ExpiryStatus.UNREADABLE: 3,
}


class OCRError(RuntimeError):
    """OCR failed for a reason the caller can act on."""


class OCRUnavailableError(OCRError):
    """EasyOCR is not installed or its models could not be fetched — API 503."""


@dataclass(frozen=True)
class OCRLine:
    """One text line as EasyOCR reported it, plus which variant produced it."""

    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    variant: str = "raw"


@dataclass(frozen=True)
class ExpiryReadResult:
    """Everything one packaging crop produced."""

    extractions: List[ExpiryExtraction] = field(default_factory=list)
    lines: List[OCRLine] = field(default_factory=list)
    latency_ms: float = 0.0
    ocr_ms: float = 0.0
    variants_tried: List[str] = field(default_factory=list)
    variant_used: Optional[str] = None
    image_width: int = 0
    image_height: int = 0

    @property
    def best(self) -> Optional[ExpiryExtraction]:
        """The most decisive dated read, or None when nothing parsed."""
        dated = [e for e in self.extractions if e.parsed_date is not None]
        if not dated:
            return self.extractions[0] if self.extractions else None
        return sorted(
            dated,
            key=lambda e: (STATUS_PRIORITY[e.status], -(e.ocr_confidence or 0.0)),
        )[0]

    @property
    def raw_text(self) -> str:
        """Everything OCR read, newline-joined — persisted for audit."""
        return "\n".join(line.text for line in self.lines)

    @property
    def found_date(self) -> bool:
        return any(e.parsed_date is not None for e in self.extractions)


# ------------------------------------------------------- preprocessing ----
def _variant_raw(image: "Image") -> "Image":
    """Upscaled grayscale. Best on clean laser/thermal print."""
    return upscale(
        to_grayscale(image),
        factor=settings.OCR_UPSCALE_FACTOR,
        min_height=settings.OCR_MIN_HEIGHT,
    )


def _variant_clahe_sharpen(image: "Image") -> "Image":
    """Contrast-limited equalisation + unsharp — faint ink on light packaging."""
    return sharpen(
        upscale(
            enhance_contrast(image),
            factor=settings.OCR_UPSCALE_FACTOR,
            min_height=settings.OCR_MIN_HEIGHT,
        )
    )


def _variant_otsu(image: "Image") -> "Image":
    """Global binarisation — flat, evenly-lit labels."""
    return otsu_threshold(_variant_raw(image))


def _variant_otsu_invert(image: "Image") -> "Image":
    """Inverted binarisation — light-on-dark stamps, which OCR otherwise misses."""
    return otsu_threshold(_variant_raw(image), invert=True)


def _variant_adaptive_close(image: "Image") -> "Image":
    """Local threshold + morphological close.

    The dot-matrix case: `close` bridges the dot cloud of each glyph into a
    continuous stroke. Kernel stays at 2px — larger merges adjacent digits.
    """
    binary = adaptive_threshold(_variant_raw(image), block_size=31, offset=10)
    return morphology(binary, operation="close", kernel_size=2, iterations=1)


VARIANTS: Dict[str, Callable[["Image"], "Image"]] = {
    "raw": _variant_raw,
    "clahe_sharpen": _variant_clahe_sharpen,
    "otsu": _variant_otsu,
    "otsu_invert": _variant_otsu_invert,
    "adaptive_close": _variant_adaptive_close,
}


class ExpiryOCRService:
    """Singleton wrapper around an EasyOCR reader."""

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        gpu: Optional[bool] = None,
        variants: Optional[Sequence[str]] = None,
    ) -> None:
        self.languages: List[str] = list(languages or settings.OCR_LANGUAGES)
        self.gpu: bool = settings.OCR_GPU if gpu is None else gpu
        self.variants: List[str] = [
            name for name in (variants or settings.OCR_VARIANTS) if name in VARIANTS
        ] or ["raw"]

        self._reader: Any = None
        self._lock = Lock()
        self._load_failure: Optional[str] = None

    # ------------------------------------------------------------ lifecycle --
    @property
    def is_ready(self) -> bool:
        return self._reader is not None

    @property
    def version(self) -> str:
        return f"easyocr:{'+'.join(self.languages)}@gpu={self.gpu}"

    @property
    def load_failure(self) -> Optional[str]:
        return self._load_failure

    def load(self) -> bool:
        """Construct the reader once (first call downloads ~100 MB of models)."""
        if self._reader is not None:
            return True

        with self._lock:
            if self._reader is not None:
                return True
            try:
                import easyocr  # noqa: PLC0415 - deferred heavy import
            except ImportError:
                self._load_failure = (
                    "easyocr is not installed — pip install -r requirements-ml.txt"
                )
                logger.warning(self._load_failure)
                return False

            try:
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    download_enabled=settings.OCR_ALLOW_DOWNLOAD,
                    verbose=False,
                )
            except Exception as exc:  # noqa: BLE001 - offline machines land here
                self._load_failure = f"Could not initialise EasyOCR: {exc}"
                logger.error(self._load_failure)
                return False

            self._load_failure = None
            logger.info("EasyOCR ready (langs=%s, gpu=%s)", self.languages, self.gpu)
            return True

    def unload(self) -> None:
        with self._lock:
            self._reader = None

    # ------------------------------------------------------------ extraction --
    def extract_expiry(
        self,
        image: "Image",
        reference_date: Optional[date] = None,
        variants: Optional[Sequence[str]] = None,
    ) -> ExpiryReadResult:
        """Read a packaging crop and parse every candidate date out of it.

        Never raises for "no date found" — that is `ExpiryStatus.UNREADABLE`,
        a legitimate audit outcome. It raises only when OCR itself is broken.
        """
        if not self.load():
            raise OCRUnavailableError(self._load_failure or "EasyOCR unavailable")

        meta = describe(image)
        started = time.perf_counter()
        ocr_ms = 0.0

        chosen = [name for name in (variants or self.variants) if name in VARIANTS] or ["raw"]
        lines: List[OCRLine] = []
        extractions: List[ExpiryExtraction] = []
        tried: List[str] = []
        variant_used: Optional[str] = None
        seen_text: set[str] = set()

        for name in chosen:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if tried and elapsed_ms > settings.OCR_TIME_BUDGET_MS:
                logger.warning(
                    "OCR time budget (%.0f ms) exhausted after %s — stopping with %d candidate(s)",
                    settings.OCR_TIME_BUDGET_MS,
                    tried,
                    len(extractions),
                )
                break

            tried.append(name)
            try:
                prepared = VARIANTS[name](image)
            except Exception as exc:  # noqa: BLE001 - a bad variant must not end the read
                logger.warning("Preprocessing variant '%s' failed: %s", name, exc)
                continue

            variant_started = time.perf_counter()
            try:
                raw_results = self._reader.readtext(prepared)
            except Exception as exc:  # noqa: BLE001
                logger.warning("EasyOCR failed on variant '%s': %s", name, exc)
                continue
            ocr_ms += (time.perf_counter() - variant_started) * 1000.0

            variant_lines, variant_extractions = self._parse_results(
                raw_results, name, reference_date, meta.width, meta.height
            )
            lines.extend(variant_lines)

            for extraction in variant_extractions:
                # De-duplicate across variants: the same stamp read five ways is
                # one finding, not five.
                key = (extraction.normalized_text or "").strip().lower()
                if key and key in seen_text:
                    continue
                if key:
                    seen_text.add(key)
                extractions.append(extraction)

            best_here = _best_dated(variant_extractions)
            if best_here is not None:
                variant_used = variant_used or name
                if (best_here.ocr_confidence or 0.0) >= settings.OCR_EARLY_STOP_CONFIDENCE:
                    logger.debug("Variant '%s' produced a confident date; stopping early", name)
                    break

        extractions.sort(
            key=lambda e: (STATUS_PRIORITY[e.status], -(e.ocr_confidence or 0.0))
        )
        return ExpiryReadResult(
            extractions=extractions,
            lines=lines,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            ocr_ms=ocr_ms,
            variants_tried=tried,
            variant_used=variant_used,
            image_width=meta.width,
            image_height=meta.height,
        )

    def extract_from_source(
        self, source: Any, reference_date: Optional[date] = None
    ) -> ExpiryReadResult:
        """Ingest bytes or a path, then extract."""
        return self.extract_expiry(load_image(source), reference_date)

    def _parse_results(
        self,
        raw_results: Sequence[Any],
        variant: str,
        reference_date: Optional[date],
        width: int,
        height: int,
    ) -> Tuple[List[OCRLine], List[ExpiryExtraction]]:
        """Turn EasyOCR's `(polygon, text, confidence)` triples into candidates."""
        lines: List[OCRLine] = []
        extractions: List[ExpiryExtraction] = []

        for entry in raw_results or []:
            try:
                polygon, text, confidence = entry[0], entry[1], float(entry[2])
            except (TypeError, IndexError, ValueError):
                logger.debug("Skipping malformed OCR row from variant '%s'", variant)
                continue

            if confidence < settings.OCR_MIN_CONFIDENCE:
                continue
            text = str(text).strip()
            if not text:
                continue

            bbox = polygon_to_bbox(polygon)
            lines.append(OCRLine(text=text, confidence=confidence, bbox=bbox, variant=variant))

            extraction = parse_single(text, reference_date)
            extractions.append(
                extraction.model_copy(
                    update={
                        "ocr_confidence": confidence,
                        "bbox": bbox,
                        "matched_pattern": extraction.matched_pattern,
                    }
                )
            )

        # A date split across two OCR lines ("BEST BEFORE" / "12 09 2026") only
        # parses when they are joined, so try the concatenation too.
        if len(lines) > 1:
            joined = " ".join(line.text for line in lines)
            merged = parse_single(joined, reference_date)
            if merged.parsed_date is not None:
                confidence = min(line.confidence for line in lines)
                extractions.append(
                    merged.model_copy(update={"ocr_confidence": confidence, "bbox": None})
                )

        return lines, extractions


# --------------------------------------------------------- text-only path --
def parse_single(text: str, reference_date: Optional[date] = None) -> ExpiryExtraction:
    """Normalise → regex-match → classify a single string. No OCR involved."""
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


def polygon_to_bbox(polygon: Sequence[Sequence[float]]) -> Optional[BoundingBox]:
    """EasyOCR's 4-point pixel polygon → a normalised, clipped `BoundingBox`.

    Coordinates are normalised against the polygon's own extent rather than the
    frame: variants upscale the crop, so the OCR-space size is not the frame
    size. This keeps the box a faithful *relative* region of the crop.
    """
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, IndexError, ValueError):
        return None
    if not xs or not ys:
        return None

    scale_x = max(1.0, max(xs))
    scale_y = max(1.0, max(ys))
    x1, x2 = min(xs) / scale_x, max(xs) / scale_x
    y1, y2 = min(ys) / scale_y, max(ys) / scale_y

    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    try:
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    except ValueError:
        return None


def _best_dated(extractions: Sequence[ExpiryExtraction]) -> Optional[ExpiryExtraction]:
    dated = [e for e in extractions if e.parsed_date is not None]
    if not dated:
        return None
    return max(dated, key=lambda e: e.ocr_confidence or 0.0)


_service: Optional[ExpiryOCRService] = None
_service_lock = Lock()


def get_ocr_service() -> ExpiryOCRService:
    """Process-wide EasyOCR singleton."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ExpiryOCRService()
    return _service
