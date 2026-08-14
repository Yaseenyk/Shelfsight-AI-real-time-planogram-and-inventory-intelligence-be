"""YOLOv8 inference wrapper.

Phase 1 contract:

    detector = get_detector()
    detections: List[Detection] = detector.predict(bgr_ndarray)
    result:     DetectionResult = detector.predict_with_metrics(bgr_ndarray)

`predict()` returns `Detection` objects rather than bare `BoundingBox` values:
the compliance engine matches a slot only against a detection of the *same SKU*,
so class identity and confidence have to travel with the geometry. Each
`Detection.bbox` is a validated Phase 0 `BoundingBox` in normalised xyxy.

Design notes
------------
- **Singleton with a load lock.** Weights are hundreds of MB and Uvicorn serves
  requests from a threadpool; two concurrent first-requests must not both load.
- **Lazy heavy imports.** `ultralytics`/`torch` are imported inside `load()`, so
  the API, the schema layer and the test suite all import without them.
- **Never raise on a missing model at import time.** `is_ready` reports the
  truth and the API turns that into a 503 with an actionable message.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.common import BoundingBox, Detection
from app.utils.geometry import iou_xyxy
from app.utils.vision import ImageSource, describe, load_image

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

    Image = NDArray[np.uint8]
else:
    Image = Any

logger = get_logger(__name__)


class DetectionError(RuntimeError):
    """Inference failed for a reason the caller can act on."""


class DetectorUnavailableError(DetectionError):
    """Weights or the ultralytics package are missing — the API maps this to 503."""


@dataclass(frozen=True)
class DetectionResult:
    """Detections plus the instrumentation the paper reports."""

    detections: List[Detection]
    latency_ms: float
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    model_version: str = ""
    suppressed: int = 0  # boxes removed by our own NMS/area filters
    class_counts: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):  # noqa: ANN204 - iterating the result yields detections
        return iter(self.detections)

    @property
    def fps(self) -> float:
        return 1000.0 / self.latency_ms if self.latency_ms > 0 else 0.0


def non_max_suppression(
    detections: Sequence[Detection],
    iou_threshold: float = 0.45,
    class_agnostic: bool = True,
) -> Tuple[List[Detection], int]:
    """Greedy NMS over already-decoded detections.

    Ultralytics runs NMS inside `predict()`, but only ever *per class* unless
    `agnostic_nms` is set. This pass exists for the cross-class case that matters
    on a shelf: one physical bottle detected as both `bottle` and `cup` is one
    object, and leaving both in inflates the facing count and fabricates an
    `EXTRA` compliance violation.

    Returns `(kept, suppressed_count)`.
    """
    if not detections:
        return [], 0

    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    for candidate in ordered:
        overlaps = False
        for keeper in kept:
            if not class_agnostic and candidate.class_id != keeper.class_id:
                continue
            if iou_xyxy(candidate.bbox.as_tuple(), keeper.bbox.as_tuple()) >= iou_threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(candidate)
    return kept, len(detections) - len(kept)


class YoloDetector:
    """Singleton wrapper around an Ultralytics YOLOv8 model."""

    def __init__(
        self,
        weights: Optional[Path] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
    ) -> None:
        self.weights: Path = Path(weights or settings.DETECTION_WEIGHTS)
        self.conf: float = conf if conf is not None else settings.DETECTION_CONF_THRESHOLD
        self.iou: float = iou if iou is not None else settings.DETECTION_IOU_NMS
        self.device: str = device or settings.DETECTION_DEVICE
        self.imgsz: int = imgsz or settings.DETECTION_IMG_SIZE

        self._model: Any = None
        self._names: Dict[int, str] = {}
        self._lock = Lock()
        self._load_failure: Optional[str] = None

    # ------------------------------------------------------------ lifecycle --
    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> Dict[int, str]:
        """Class id → name, available once the model is loaded."""
        return dict(self._names)

    @property
    def version(self) -> str:
        return f"yolov8:{self.weights.name}@conf{self.conf}:iou{self.iou}:imgsz{self.imgsz}"

    @property
    def load_failure(self) -> Optional[str]:
        """Why the last load attempt failed, for /health and error responses."""
        return self._load_failure

    def load(self) -> bool:
        """Load weights once. Returns False and records the reason on failure."""
        if self._model is not None:
            return True

        with self._lock:
            if self._model is not None:  # another thread won the race
                return True
            try:
                from ultralytics import YOLO  # noqa: PLC0415 - deferred heavy import
            except ImportError:
                self._load_failure = (
                    "ultralytics is not installed — pip install -r requirements-ml.txt"
                )
                logger.warning(self._load_failure)
                return False

            source = self._resolve_weights()
            if source is None:
                return False

            try:
                model = YOLO(str(source))
                model.to(self.device)
            except Exception as exc:  # noqa: BLE001 - surface any loader failure
                self._load_failure = f"Could not load {source}: {exc}"
                logger.error(self._load_failure)
                return False

            self._model = model
            self._names = {int(k): str(v) for k, v in (getattr(model, "names", {}) or {}).items()}
            self._load_failure = None
            logger.info(
                "Detector ready: %s on %s (%d classes)",
                self.weights.name,
                self.device,
                len(self._names),
            )
            self._warn_if_generic()
            if settings.DETECTION_WARMUP:
                self._warmup()
            return True

    #: Classes present in the COCO baseline and absent from any shelf detector.
    _COCO_TELLS = frozenset({"person", "bicycle", "car", "traffic light", "fire hydrant"})

    @property
    def is_generic_baseline(self) -> bool:
        """True when the loaded model is the stock COCO detector.

        Worth detecting because every path to it is silent. The weights file is
        gitignored, so a fresh clone has none; Ultralytics then downloads
        yolov8n.pt automatically; the service loads it without error and the
        health endpoint reports `detector_loaded: true`. The system looks
        entirely well and reports people and cars on a shelf photograph.

        Identified by the class vocabulary rather than the filename, so a
        renamed or relocated copy is caught just the same.
        """
        if not self._names:
            return False
        lowered = {name.lower() for name in self._names.values()}
        return len(self._COCO_TELLS & lowered) >= 3

    def _warn_if_generic(self) -> None:
        if not self.is_generic_baseline:
            return
        logger.warning(
            "=" * 72 + "\n  The loaded detector is the generic COCO model (%d classes), not a"
            "\n  shelf detector. It recognises people, cars and animals -- shelf"
            "\n  results will be meaningless."
            "\n"
            "\n  Loaded from : %s"
            "\n  Fix         : point DETECTION_WEIGHTS at your trained checkpoint,"
            "\n                or copy the supplied models/weights folder into place."
            "\n" + "=" * 72,
            len(self._names),
            self.weights,
        )

    def _resolve_weights(self) -> Optional[Path]:
        """Return a usable checkpoint path, downloading the base model if allowed."""
        if self.weights.exists():
            return self.weights

        if not settings.DETECTION_ALLOW_DOWNLOAD:
            self._load_failure = (
                f"Detector weights not found at {self.weights} and downloads are disabled "
                "(DETECTION_ALLOW_DOWNLOAD=false)"
            )
            logger.warning(self._load_failure)
            return None

        # Ultralytics fetches a bare model name into the current working dir;
        # move it under models/weights/ so the next start is offline-clean.
        logger.info(
            "Weights missing at %s — fetching pretrained %s",
            self.weights,
            settings.DETECTION_BASE_MODEL,
        )
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            YOLO(settings.DETECTION_BASE_MODEL)  # triggers the download
        except Exception as exc:  # noqa: BLE001 - offline machines land here
            self._load_failure = (
                f"Could not download {settings.DETECTION_BASE_MODEL}: {exc}. "
                f"Place a checkpoint at {self.weights} manually."
            )
            logger.error(self._load_failure)
            return None

        downloaded = Path(settings.DETECTION_BASE_MODEL)
        if downloaded.exists():
            self.weights.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(self.weights))
            logger.info("Cached pretrained weights at %s", self.weights)
            return self.weights
        return Path(settings.DETECTION_BASE_MODEL)

    def _warmup(self) -> None:
        """First inference pays lazy CUDA/graph init; spend it at startup."""
        try:
            import numpy as np  # noqa: PLC0415

            blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._model.predict(source=blank, imgsz=self.imgsz, device=self.device, verbose=False)
            logger.debug("Detector warmup complete")
        except Exception as exc:  # noqa: BLE001 - warmup must never block startup
            logger.warning("Detector warmup skipped: %s", exc)

    def unload(self) -> None:
        """Release the model (used by tests and by weight hot-swaps)."""
        with self._lock:
            self._model = None
            self._names = {}

    # ------------------------------------------------------------ inference --
    def predict(
        self,
        image: "Image",
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Detection]:
        """Detect products in a BGR `ndarray`. The primary Phase 1 entry point."""
        return self.predict_with_metrics(image, conf=conf, iou=iou).detections

    def predict_with_metrics(
        self,
        image: "Image",
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> DetectionResult:
        """Detect and return timings alongside the boxes.

        Raises `DetectorUnavailableError` when no model could be loaded and
        `DetectionError` when inference itself fails — a caller must never
        mistake a broken pipeline for an empty shelf.
        """
        if not self.load():
            raise DetectorUnavailableError(self._load_failure or "Detector is unavailable")

        meta = describe(image)
        started = time.perf_counter()
        try:
            # Ultralytics accepts BGR ndarrays directly and handles the RGB swap
            # and letterboxing internally — feeding it a pre-normalised CHW
            # tensor here would double-apply both.
            results = self._model.predict(
                source=image,
                conf=self.conf if conf is None else conf,
                iou=self.iou if iou is None else iou,
                imgsz=self.imgsz,
                device=self.device,
                max_det=settings.DETECTION_MAX_DET,
                agnostic_nms=settings.DETECTION_AGNOSTIC_NMS,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - wrap any backend failure
            raise DetectionError(f"YOLOv8 inference failed: {exc}") from exc

        inference_ms = (time.perf_counter() - started) * 1000.0

        post_started = time.perf_counter()
        detections, dropped = self._parse(results)

        if settings.DETECTION_EXTRA_NMS:
            detections, suppressed = non_max_suppression(
                detections,
                iou_threshold=self.iou if iou is None else iou,
                class_agnostic=True,
            )
            dropped += suppressed
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0

        speed = getattr(results[0], "speed", {}) if results else {}
        counts: Dict[str, int] = {}
        for detection in detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1

        return DetectionResult(
            detections=detections,
            latency_ms=inference_ms + postprocess_ms,
            preprocess_ms=float(speed.get("preprocess", 0.0) or 0.0),
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            image_width=meta.width,
            image_height=meta.height,
            model_version=self.version,
            suppressed=dropped,
            class_counts=counts,
        )

    def predict_source(self, source: ImageSource, conf: Optional[float] = None) -> DetectionResult:
        """Ingest bytes or a path, then detect. Convenience for API/benchmark."""
        return self.predict_with_metrics(load_image(source), conf=conf)

    # -------------------------------------------------------------- parsing --
    def _parse(self, results: Sequence[Any]) -> Tuple[List[Detection], int]:
        """Convert raw Ultralytics output into validated `Detection` objects.

        Every box is individually guarded: one malformed row (a degenerate box,
        a NaN score, an unknown class id) is dropped and counted, never allowed
        to abort a frame that produced 40 good detections.
        """
        detections: List[Detection] = []
        dropped = 0

        for result in results:
            names: Dict[int, str] = {
                int(k): str(v) for k, v in (getattr(result, "names", {}) or {}).items()
            }
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            # `xyxyn` is already normalised to [0, 1] against the ORIGINAL frame
            # size — Ultralytics undoes its own letterbox before exposing it.
            for xyxyn, cls_value, conf_value in zip(boxes.xyxyn, boxes.cls, boxes.conf):
                try:
                    coords = [float(v) for v in _to_list(xyxyn)]
                    class_id = int(_to_scalar(cls_value))
                    confidence = float(_to_scalar(conf_value))
                except (TypeError, ValueError) as exc:
                    logger.debug("Skipping malformed detection row: %s", exc)
                    dropped += 1
                    continue

                detection = self._build_detection(coords, class_id, confidence, names)
                if detection is None:
                    dropped += 1
                    continue
                detections.append(detection)

        return detections, dropped

    def _build_detection(
        self,
        coords: List[float],
        class_id: int,
        confidence: float,
        names: Dict[int, str],
    ) -> Optional[Detection]:
        if len(coords) != 4 or any(v != v for v in coords):  # NaN check
            return None
        if not 0.0 <= confidence <= 1.0:
            logger.debug("Dropping detection with out-of-range confidence %s", confidence)
            return None

        x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in coords)
        if (x2 - x1) * (y2 - y1) < settings.DETECTION_MIN_BOX_AREA:
            return None

        try:
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        except ValueError:
            # Degenerate after clipping (zero width/height at the frame edge).
            return None

        return Detection(
            class_id=class_id,
            class_name=names.get(class_id) or self._names.get(class_id) or str(class_id),
            confidence=confidence,
            bbox=bbox,
        )


def _to_list(value: Any) -> Sequence[float]:
    """Torch tensor, numpy array or plain sequence → list of floats."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _to_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return value.item()
    return float(value)


_detector: Optional[YoloDetector] = None
_detector_lock = Lock()


def get_detector() -> YoloDetector:
    """Process-wide detector singleton."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = YoloDetector()
    return _detector


#: Phase 0 alias — existing imports keep working.
get_detection_service = get_detector
DetectionService = YoloDetector
