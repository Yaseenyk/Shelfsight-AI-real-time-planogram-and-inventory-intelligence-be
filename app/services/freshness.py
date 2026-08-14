"""Perishable freshness classification (MobileNetV2 / ResNet50).

Phase 2 contract:

    service = get_freshness_service()
    result: FreshnessResult = service.predict_freshness(bgr_ndarray)
    result.label            # FreshnessLabel.FRESH | RIPENING | SPOILED
    result.probabilities    # {"fresh": 0.02, "ripening": 0.11, "spoiled": 0.87}
    result.latency_ms

Mirrors `services/detection.py` deliberately — same singleton-with-load-lock,
same lazy heavy imports, same "never silently degrade" error contract — so the
two vision services behave identically under failure.

Checkpoint format (written by `models/train_freshness.py`):

    {"state_dict": ..., "backbone": "mobilenet_v2", "classes": ["fresh", ...]}

The class list travels *inside* the checkpoint: label order is a property of the
trained head, and reading it from config instead is how a model silently starts
reporting "spoiled" for fresh produce after someone reorders a settings list.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import FreshnessLabel
from app.schemas.common import BoundingBox
from app.schemas.freshness import FreshnessPrediction
from app.utils.vision import ImageDecodeError, describe, load_image, to_rgb

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

    Image = NDArray[np.uint8]
else:
    Image = Any

logger = get_logger(__name__)

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Canonical label order used when a checkpoint does not carry its own.
DEFAULT_CLASSES: Tuple[str, ...] = ("fresh", "ripening", "spoiled")


class FreshnessError(RuntimeError):
    """Classification failed for a reason the caller can act on."""


class FreshnessUnavailableError(FreshnessError):
    """Weights or torch are missing — the API maps this to 503."""


@dataclass(frozen=True)
class FreshnessResult:
    """One classified crop, plus the instrumentation the paper reports."""

    label: FreshnessLabel
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    backbone: str = ""
    model_version: str = ""
    image_width: int = 0
    image_height: int = 0
    bbox: Optional[BoundingBox] = None

    @property
    def is_actionable(self) -> bool:
        """Spoiled or ripening stock needs a human — fresh stock does not."""
        return self.label is not FreshnessLabel.FRESH

    def to_prediction(self) -> FreshnessPrediction:
        """Adapt to the API response schema."""
        return FreshnessPrediction(
            label=self.label,
            confidence=self.confidence,
            class_probabilities=self.probabilities,
            bbox=self.bbox,
            backbone=self.backbone or None,
            latency_ms=self.latency_ms,
        )


class FreshnessService:
    """Singleton wrapper around the fine-tuned freshness CNN."""

    def __init__(
        self,
        weights: Optional[Path] = None,
        backbone: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
        input_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.weights: Path = Path(weights or settings.FRESHNESS_WEIGHTS)
        self.backbone: str = backbone or settings.FRESHNESS_BACKBONE
        self.classes: List[str] = list(classes or settings.FRESHNESS_CLASSES or DEFAULT_CLASSES)
        self.input_size: int = input_size or settings.FRESHNESS_INPUT_SIZE
        self.device: str = device or settings.DETECTION_DEVICE

        self._model: Any = None
        self._transform: Any = None
        self._lock = Lock()
        self._load_failure: Optional[str] = None
        self._trained_at: Optional[str] = None

    # ------------------------------------------------------------ lifecycle --
    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return f"{self.backbone}:{self.weights.name}@{self.input_size}px"

    @property
    def load_failure(self) -> Optional[str]:
        return self._load_failure

    @property
    def trained_at(self) -> Optional[str]:
        return self._trained_at

    def load(self) -> bool:
        """Load the checkpoint once. Returns False and records why on failure."""
        if self._model is not None:
            return True

        with self._lock:
            if self._model is not None:  # another thread won the race
                return True

            if not self.weights.exists():
                self._load_failure = (
                    f"Freshness weights not found at {self.weights} — train one with "
                    "python models/train_freshness.py --data-dir <dataset>"
                )
                logger.warning(self._load_failure)
                return False

            try:
                import torch  # noqa: PLC0415 - deferred heavy import
                from torchvision import transforms  # noqa: PLC0415
            except ImportError:
                self._load_failure = (
                    "torch/torchvision are not installed — pip install -r requirements-ml.txt"
                )
                logger.warning(self._load_failure)
                return False

            try:
                checkpoint = self._read_checkpoint(torch)
                model = self._build_model(checkpoint)
            except Exception as exc:  # noqa: BLE001 - surface any loader failure
                self._load_failure = f"Could not load {self.weights}: {exc}"
                logger.error(self._load_failure)
                return False

            model.eval()
            model.to(self.device)
            self._model = model
            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.input_size, self.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
            self._load_failure = None
            logger.info(
                "Freshness classifier ready: %s (%s) classes=%s",
                self.weights.name,
                self.backbone,
                self.classes,
            )
            return True

    def _read_checkpoint(self, torch: Any) -> Any:
        """Load the checkpoint file, preferring the safe (weights-only) path.

        torch >= 2.6 defaults `weights_only=True`, which refuses pickled module
        objects. Our own checkpoints are plain dicts and load fine that way; a
        fully-serialised module needs the unsafe path, so it is attempted second
        and logged — the operator should know when arbitrary pickle is executed.
        """
        try:
            return torch.load(self.weights, map_location=self.device, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - torch raises several types here
            logger.warning(
                "Safe load failed for %s (%s); retrying with weights_only=False. "
                "Only load checkpoints you produced.",
                self.weights.name,
                exc,
            )
            return torch.load(self.weights, map_location=self.device, weights_only=False)

    def _build_model(self, checkpoint: Any) -> Any:
        """Reconstruct the network from a state-dict checkpoint or a pickled module."""
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.backbone = str(checkpoint.get("backbone", self.backbone))
            classes = checkpoint.get("classes")
            if classes:
                self.classes = [str(c) for c in classes]
            self._trained_at = checkpoint.get("trained_at")
            self.input_size = int(checkpoint.get("input_size", self.input_size))

            model = build_backbone(self.backbone, len(self.classes), pretrained=False)
            model.load_state_dict(checkpoint["state_dict"])
            return model

        # A fully-serialised nn.Module: trust its own class list if it carries one.
        classes = getattr(checkpoint, "classes", None)
        if classes:
            self.classes = [str(c) for c in classes]
        return checkpoint

    def unload(self) -> None:
        """Release the model (used by tests and weight hot-swaps)."""
        with self._lock:
            self._model = None
            self._transform = None

    # ------------------------------------------------------------ inference --
    def predict_freshness(
        self, image: "Image", bbox: Optional[BoundingBox] = None
    ) -> FreshnessResult:
        """Classify a single BGR crop. The primary Phase 2 entry point."""
        results = self.predict_batch([image])
        if not results:
            raise FreshnessError("Classifier returned no prediction for the frame")
        result = results[0]
        return result if bbox is None else _with_bbox(result, bbox)

    def predict_batch(self, images: Sequence["Image"]) -> List[FreshnessResult]:
        """Classify several crops in one forward pass.

        Batching matters here: a shelf frame yields one crop per detected
        perishable, and paying the per-call overhead N times is the difference
        between a responsive scan and a stalled one.
        """
        if not images:
            return []
        if not self.load():
            raise FreshnessUnavailableError(self._load_failure or "Classifier unavailable")

        import torch  # noqa: PLC0415

        preprocess_started = time.perf_counter()
        try:
            tensors = [self._transform(self._as_rgb(image)) for image in images]
            batch = torch.stack(tensors).to(self.device)
        except ImageDecodeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FreshnessError(f"Could not preprocess crop for classification: {exc}") from exc
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        inference_started = time.perf_counter()
        try:
            with torch.no_grad():
                logits = self._model(batch)
                probabilities = torch.softmax(logits, dim=1)
        except Exception as exc:  # noqa: BLE001 - wrap any backend failure
            raise FreshnessError(f"Freshness inference failed: {exc}") from exc
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        per_image = inference_ms / len(images)
        results: List[FreshnessResult] = []
        for row, image in zip(probabilities, images):
            scores = self._label_scores(row)
            best_name = max(scores, key=lambda name: scores[name])
            meta = describe(image)
            results.append(
                FreshnessResult(
                    label=_coerce_label(best_name),
                    confidence=scores[best_name],
                    probabilities=scores,
                    latency_ms=preprocess_ms / len(images) + per_image,
                    preprocess_ms=preprocess_ms / len(images),
                    inference_ms=per_image,
                    backbone=self.backbone,
                    model_version=self.version,
                    image_width=meta.width,
                    image_height=meta.height,
                )
            )
        return results

    def predict_source(self, source: Any) -> FreshnessResult:
        """Ingest bytes or a path, then classify. Convenience for API/benchmark."""
        return self.predict_freshness(load_image(source))

    def _label_scores(self, row: Any) -> Dict[str, float]:
        """Softmax row → `{class_name: probability}`, defensively sized.

        A checkpoint whose head is wider or narrower than `self.classes` would
        otherwise mislabel silently; zip truncation plus this guard makes the
        mismatch visible.
        """
        values = [float(v) for v in row.tolist()]
        if len(values) != len(self.classes):
            logger.warning(
                "Checkpoint head has %d outputs but %d class names are configured",
                len(values),
                len(self.classes),
            )
        names = self.classes[: len(values)]
        return {name: round(value, 6) for name, value in zip(names, values)}

    def _as_rgb(self, image: "Image") -> "Image":
        """Torchvision transforms expect RGB; OpenCV hands us BGR."""
        meta = describe(image)
        if meta.channels == 3:
            return to_rgb(image)
        return to_rgb(image)  # grayscale/BGRA are widened by to_rgb too


def _coerce_label(name: str) -> FreshnessLabel:
    """Map a checkpoint's class name onto the canonical enum.

    Real datasets use `rotten`, `stale`, `overripe`… — accepting only the exact
    three enum spellings would make half of Kaggle unusable.
    """
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return FreshnessLabel(key)
    except ValueError:
        pass

    # Order matters: "overripe" and "notfresh" contain "ripe"/"fresh", so the
    # compound and negated forms must be tested before the bare keywords.
    if any(
        token in key
        for token in ("overripe", "veryripe", "notfresh", "rotten", "spoil", "decay",
                      "mould", "mold", "stale", "bad")
    ):
        return FreshnessLabel.SPOILED
    if any(token in key for token in ("unripe",)):
        return FreshnessLabel.FRESH
    if any(token in key for token in ("ripen", "ripe", "semi", "turning", "aging")):
        return FreshnessLabel.RIPENING
    if any(token in key for token in ("fresh", "good", "healthy")):
        return FreshnessLabel.FRESH
    raise FreshnessError(f"Cannot map class '{name}' onto a FreshnessLabel")


def _with_bbox(result: FreshnessResult, bbox: BoundingBox) -> FreshnessResult:
    return FreshnessResult(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        latency_ms=result.latency_ms,
        preprocess_ms=result.preprocess_ms,
        inference_ms=result.inference_ms,
        backbone=result.backbone,
        model_version=result.model_version,
        image_width=result.image_width,
        image_height=result.image_height,
        bbox=bbox,
    )


def build_backbone(name: str, num_classes: int, pretrained: bool = True) -> Any:
    """ImageNet-pretrained backbone with a fresh classifier head.

    `pretrained=False` at inference time: the fine-tuned weights are about to
    overwrite everything anyway, and downloading ImageNet weights to throw them
    away would make the API's first request depend on the network.
    """
    import torch.nn as nn  # noqa: PLC0415
    from torchvision import models  # noqa: PLC0415

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {name!r} (expected mobilenet_v2 or resnet50)")


_service: Optional[FreshnessService] = None
_service_lock = Lock()


def get_freshness_service() -> FreshnessService:
    """Process-wide freshness classifier singleton."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = FreshnessService()
    return _service
