"""Perishable freshness classifier (MobileNetV2 / ResNet50 transfer learning).

The checkpoint is expected to be a dict with `state_dict`, `backbone` and
`classes` keys (see `models/train_freshness.py`), which keeps the label order
pinned to the checkpoint rather than to config drift.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import FreshnessLabel
from app.schemas.freshness import FreshnessPrediction

logger = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FreshnessService:
    def __init__(
        self,
        weights: Optional[Path] = None,
        backbone: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
    ) -> None:
        self.weights = Path(weights or settings.FRESHNESS_WEIGHTS)
        self.backbone = backbone or settings.FRESHNESS_BACKBONE
        self.classes: List[str] = list(classes or settings.FRESHNESS_CLASSES)
        self._model: Any = None
        self._transform: Any = None
        self._lock = Lock()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return f"{self.backbone}:{self.weights.name}"

    def load(self) -> bool:
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            if not self.weights.exists():
                logger.warning("Freshness weights missing at %s — classifier disabled", self.weights)
                return False
            try:
                import torch  # noqa: PLC0415
                from torchvision import transforms  # noqa: PLC0415
            except ImportError:
                logger.warning("torch/torchvision not installed — install requirements-ml.txt")
                return False

            checkpoint = torch.load(self.weights, map_location=settings.DETECTION_DEVICE)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.backbone = checkpoint.get("backbone", self.backbone)
                self.classes = list(checkpoint.get("classes", self.classes))
                model = build_backbone(self.backbone, len(self.classes))
                model.load_state_dict(checkpoint["state_dict"])
            else:  # a fully-serialised module
                model = checkpoint

            model.eval()
            self._model = model
            self._transform = transforms.Compose(
                [
                    transforms.Resize(
                        (settings.FRESHNESS_INPUT_SIZE, settings.FRESHNESS_INPUT_SIZE)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
            logger.info("Loaded freshness classifier %s (%s)", self.weights.name, self.backbone)
            return True

    def predict(self, images: Sequence[Any]) -> Tuple[List[FreshnessPrediction], float]:
        """Classify PIL images / file paths. Returns `(predictions, latency_ms)`."""
        if not images or not self.load():
            return [], 0.0

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        tensors = []
        for item in images:
            image = Image.open(item).convert("RGB") if isinstance(item, (str, Path)) else item
            tensors.append(self._transform(image))

        started = time.perf_counter()
        with torch.no_grad():
            logits = self._model(torch.stack(tensors))
            probabilities = torch.softmax(logits, dim=1)
        latency_ms = (time.perf_counter() - started) * 1000.0

        predictions: List[FreshnessPrediction] = []
        for row in probabilities:
            scores = {name: float(row[i]) for i, name in enumerate(self.classes)}
            best = max(scores, key=scores.get)
            predictions.append(
                FreshnessPrediction(
                    label=FreshnessLabel(best),
                    confidence=scores[best],
                    class_probabilities=scores,
                    backbone=self.backbone,
                    latency_ms=latency_ms / len(images),
                )
            )
        return predictions, latency_ms


def build_backbone(name: str, num_classes: int) -> Any:
    """Construct an ImageNet-pretrained backbone with a fresh classifier head."""
    import torch.nn as nn  # noqa: PLC0415
    from torchvision import models  # noqa: PLC0415

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported backbone: {name}")


_service: Optional[FreshnessService] = None


def get_freshness_service() -> FreshnessService:
    global _service
    if _service is None:
        _service = FreshnessService()
    return _service
