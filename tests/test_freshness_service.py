"""Freshness classifier service.

The end-to-end tests build a *real* checkpoint from a tiny CNN, so the whole
load → preprocess → softmax → label path runs without downloading ImageNet
weights or needing a GPU.
"""

from __future__ import annotations

import pytest

from app.models.enums import FreshnessLabel
from app.schemas.common import BoundingBox
from app.services.freshness import (
    DEFAULT_CLASSES,
    FreshnessError,
    FreshnessResult,
    FreshnessService,
    FreshnessUnavailableError,
    _coerce_label,
    build_backbone,
)

torch = pytest.importorskip("torch", reason="requirements-ml.txt not installed")
np = pytest.importorskip("numpy")


@pytest.fixture()
def crop() -> "np.ndarray":
    """A 64×64 BGR crop with a distinctive colour cast."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 180  # green-ish produce
    return image


@pytest.fixture()
def checkpoint(tmp_path):  # noqa: ANN001
    """A 3-class checkpoint in the format train_freshness.py writes."""
    from torch import nn

    model = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 3),
    )
    path = tmp_path / "freshness_test.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": "tiny_test_cnn",
            "classes": ["fresh", "ripening", "spoiled"],
            "input_size": 64,
            "trained_at": "2026-08-14T00:00:00+00:00",
        },
        path,
    )
    return path, model


def _service_with(model, classes=("fresh", "ripening", "spoiled")):  # noqa: ANN001
    """A service with the model injected, bypassing checkpoint loading."""
    from torchvision import transforms

    service = FreshnessService(weights="unused.pt", classes=list(classes), input_size=64)
    model.eval()
    service._model = model
    service._transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    return service


class _FixedLogits(torch.nn.Module):
    """Returns a preset logit vector — makes softmax assertions deterministic."""

    def __init__(self, logits) -> None:  # noqa: ANN001
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, batch):  # noqa: ANN001, ANN201
        return self.logits.unsqueeze(0).repeat(batch.shape[0], 1)


# ---------------------------------------------------------------- inference --
def test_predict_freshness_returns_all_three_probabilities(crop):  # noqa: ANN001
    service = _service_with(_FixedLogits([0.1, 0.3, 5.0]))
    result = service.predict_freshness(crop)

    assert isinstance(result, FreshnessResult)
    assert result.label is FreshnessLabel.SPOILED
    assert set(result.probabilities) == {"fresh", "ripening", "spoiled"}
    assert result.probabilities["spoiled"] == pytest.approx(result.confidence)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert result.latency_ms > 0
    assert (result.image_width, result.image_height) == (64, 64)


def test_predict_freshness_flags_actionable_stock(crop):  # noqa: ANN001
    fresh = _service_with(_FixedLogits([9.0, 0.1, 0.1])).predict_freshness(crop)
    spoiled = _service_with(_FixedLogits([0.1, 0.1, 9.0])).predict_freshness(crop)
    assert fresh.is_actionable is False
    assert spoiled.is_actionable is True


def test_predict_batch_classifies_every_crop(crop):  # noqa: ANN001
    results = _service_with(_FixedLogits([0.1, 5.0, 0.1])).predict_batch([crop, crop, crop])
    assert len(results) == 3
    assert all(r.label is FreshnessLabel.RIPENING for r in results)


def test_predict_batch_on_empty_input_returns_empty():
    assert _service_with(_FixedLogits([1.0, 0.0, 0.0])).predict_batch([]) == []


def test_grayscale_crop_is_widened_to_three_channels():
    gray = np.full((32, 32), 120, dtype=np.uint8)
    result = _service_with(_FixedLogits([5.0, 0.1, 0.1])).predict_freshness(gray)
    assert result.label is FreshnessLabel.FRESH


def test_bbox_is_attached_when_supplied(crop):  # noqa: ANN001
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.4, y2=0.6)
    result = _service_with(_FixedLogits([5.0, 0.1, 0.1])).predict_freshness(crop, bbox=bbox)
    assert result.bbox == bbox
    assert result.to_prediction().bbox == bbox


def test_head_wider_than_class_names_does_not_mislabel(crop):  # noqa: ANN001
    # A 4-output head with 3 configured names: extra outputs are ignored, and
    # the label still comes from a name that actually exists.
    service = _service_with(_FixedLogits([0.1, 0.2, 9.0, 8.0]))
    result = service.predict_freshness(crop)
    assert result.label is FreshnessLabel.SPOILED
    assert len(result.probabilities) == 3


def test_inference_failure_is_wrapped(crop):  # noqa: ANN001
    class _Exploding(torch.nn.Module):
        def forward(self, batch):  # noqa: ANN001, ANN201
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(FreshnessError, match="inference failed"):
        _service_with(_Exploding()).predict_freshness(crop)


# ------------------------------------------------------------------ loading --
def test_missing_weights_raise_unavailable(tmp_path, crop):  # noqa: ANN001
    service = FreshnessService(weights=tmp_path / "absent.pt")
    with pytest.raises(FreshnessUnavailableError, match="not found"):
        service.predict_freshness(crop)
    assert service.is_ready is False
    assert "absent.pt" in (service.load_failure or "")


def test_corrupt_checkpoint_reports_instead_of_crashing(tmp_path, crop):  # noqa: ANN001
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a torch checkpoint")
    service = FreshnessService(weights=broken)
    with pytest.raises(FreshnessUnavailableError):
        service.predict_freshness(crop)
    assert "Could not load" in (service.load_failure or "")


def test_checkpoint_class_list_overrides_config(checkpoint, monkeypatch):  # noqa: ANN001
    path, model = checkpoint
    service = FreshnessService(weights=path, classes=["wrong", "names", "here"])

    # Bypass build_backbone: the checkpoint holds a toy net, not a torchvision one.
    monkeypatch.setattr(
        "app.services.freshness.build_backbone", lambda *a, **k: model  # noqa: ARG005
    )
    assert service.load() is True
    assert service.classes == ["fresh", "ripening", "spoiled"]
    assert service.backbone == "tiny_test_cnn"
    assert service.trained_at is not None


def test_unload_allows_reloading(crop):  # noqa: ANN001
    service = _service_with(_FixedLogits([5.0, 0.1, 0.1]))
    assert service.is_ready is True
    service.unload()
    assert service.is_ready is False


# ------------------------------------------------------------ label mapping --
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fresh", FreshnessLabel.FRESH),
        ("Fresh", FreshnessLabel.FRESH),
        ("good_quality", FreshnessLabel.FRESH),
        ("ripening", FreshnessLabel.RIPENING),
        ("semi-ripe", FreshnessLabel.RIPENING),
        ("overripe", FreshnessLabel.SPOILED),
        ("rotten", FreshnessLabel.SPOILED),
        ("mouldy", FreshnessLabel.SPOILED),
        ("decayed", FreshnessLabel.SPOILED),
    ],
)
def test_dataset_class_names_map_onto_the_enum(name, expected):  # noqa: ANN001
    assert _coerce_label(name) is expected


def test_unmappable_class_name_raises():
    with pytest.raises(FreshnessError, match="Cannot map class"):
        _coerce_label("schrodinger")


def test_default_classes_match_the_enum():
    assert tuple(label.value for label in FreshnessLabel) == DEFAULT_CLASSES


def test_build_backbone_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_backbone("alexnet", 3, pretrained=False)


def test_build_backbone_head_matches_class_count():
    model = build_backbone("mobilenet_v2", 3, pretrained=False)
    assert model.classifier[1].out_features == 3
