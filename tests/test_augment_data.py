"""Dataset augmentation: colour transforms, degradation, and the audit trail."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from models.augment_data import (
    DATE_TEMPLATES,
    SEVERITY_PROFILES,
    degrade_stamp,
    generate_ocr_set,
    generate_ripening_set,
    render_date_stamp,
    synthesize_ripening,
)

cv2 = pytest.importorskip("cv2", reason="requirements-ml.txt not installed")
np = pytest.importorskip("numpy")


def _green_produce(size: int = 96) -> "np.ndarray":
    """A saturated green blob on a neutral tray — a stand-in for fresh produce."""
    image = np.full((size, size, 3), 200, dtype=np.uint8)  # unsaturated background
    cv2.circle(image, (size // 2, size // 2), size // 3, (40, 180, 60), -1)
    return image


def _mean_hue(image: "np.ndarray") -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 55) & (hsv[..., 2] > 45)
    return float(hsv[..., 0][mask].mean()) if mask.any() else 0.0


# ------------------------------------------------------------ ripening HSV --
def test_ripening_shifts_hue_from_green_toward_yellow():
    rng = np.random.default_rng(0)
    source = _green_produce()
    ripened, params = synthesize_ripening(source, progress=0.8, rng=rng)

    before, after = _mean_hue(source), _mean_hue(ripened)
    assert before > 40  # green
    assert after < before  # moved toward yellow/orange
    assert params.progress == 0.8


def test_progress_controls_how_far_the_shift_goes():
    source = _green_produce()
    mild = _mean_hue(synthesize_ripening(source, 0.2, np.random.default_rng(0))[0])
    strong = _mean_hue(synthesize_ripening(source, 0.9, np.random.default_rng(0))[0])
    assert strong < mild


def test_zero_progress_leaves_the_image_essentially_unchanged():
    source = _green_produce()
    output, _ = synthesize_ripening(source, 0.0, np.random.default_rng(0))
    assert abs(_mean_hue(output) - _mean_hue(source)) < 1.0


def test_unsaturated_background_is_not_recoloured():
    """Shifting the whole frame would teach the classifier the background."""
    source = _green_produce()
    output, _ = synthesize_ripening(source, 0.9, np.random.default_rng(0))
    assert np.allclose(output[2, 2], source[2, 2], atol=6)


def test_spoiled_mode_darkens_and_desaturates():
    source = _green_produce()
    ripe, _ = synthesize_ripening(source, 0.8, np.random.default_rng(1), spoiled=False)
    rotten, params = synthesize_ripening(source, 0.8, np.random.default_rng(1), spoiled=True)

    assert params.saturation_gain < 1.0
    assert float(rotten.mean()) < float(ripe.mean())


def test_ripening_is_deterministic_for_a_seed():
    source = _green_produce()
    first, _ = synthesize_ripening(source, 0.6, np.random.default_rng(7))
    second, _ = synthesize_ripening(source, 0.6, np.random.default_rng(7))
    assert np.array_equal(first, second)


# --------------------------------------------------------- ripening set I/O --
def test_generate_ripening_set_writes_images_and_manifest(tmp_path: Path):
    source_dir = tmp_path / "fresh"
    source_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(source_dir / f"fresh_{i}.jpg"), _green_produce())

    out_dir = tmp_path / "ripening"
    manifest = generate_ripening_set(source_dir, out_dir, count=6, seed=42)

    images = sorted(out_dir.glob("*.jpg"))
    assert len(images) == 6
    assert manifest["count"] == 6
    assert manifest["synthetic"] is True
    assert "SYNTHETIC" in manifest["note"]

    written = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert written["samples"][0]["source"].startswith("fresh_")
    assert written["samples"][0]["params"]["hue_target"] > 0


def test_generate_ripening_set_requires_source_images(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No source images"):
        generate_ripening_set(empty, tmp_path / "out", count=2)


def test_generated_set_is_reproducible(tmp_path: Path):
    source_dir = tmp_path / "fresh"
    source_dir.mkdir()
    cv2.imwrite(str(source_dir / "a.jpg"), _green_produce())

    first = generate_ripening_set(source_dir, tmp_path / "out1", count=4, seed=11)
    second = generate_ripening_set(source_dir, tmp_path / "out2", count=4, seed=11)
    assert [s["params"] for s in first["samples"]] == [s["params"] for s in second["samples"]]


# ------------------------------------------------------- OCR degradation --
def test_render_produces_a_readable_canvas():
    image = render_date_stamp("EXP 12/09/2026", np.random.default_rng(0))
    assert image.shape[2] == 3
    assert image.std() > 10  # there is actually ink on it


@pytest.mark.parametrize("severity", sorted(SEVERITY_PROFILES))
def test_degradation_runs_at_every_severity(severity):  # noqa: ANN001
    rng = np.random.default_rng(3)
    clean = render_date_stamp("EXP 12/09/2026", rng)
    degraded, params = degrade_stamp(clean, severity, rng)

    assert degraded.shape == clean.shape
    assert params.severity == severity


def test_harsher_settings_reduce_contrast_more():
    def contrast(severity: str) -> float:
        rng = np.random.default_rng(5)
        clean = render_date_stamp("EXP 12/09/2026", rng)
        return float(degrade_stamp(clean, severity, rng)[0].std())

    assert contrast("harsh") < contrast("mild")


def test_degradation_records_every_applied_transform():
    rng = np.random.default_rng(9)
    clean = render_date_stamp("BB 20260818", rng)
    _, params = degrade_stamp(clean, "harsh", rng)

    assert params.blur_sigma > 0
    assert params.noise_sigma > 0
    assert params.contrast < 1.0
    assert params.jpeg_quality <= 95


def test_generate_ocr_set_emits_scoreable_ground_truth(tmp_path: Path):
    manifest = generate_ocr_set(tmp_path, count=9, seed=42)

    images = sorted(tmp_path.glob("*.jpg"))
    assert len(images) == 9
    assert manifest["count"] == 9

    with (tmp_path / "ground_truth.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert set(rows[0]) == {"image", "truth_text", "truth_date"}
    # Every CSV row must point at a file that exists, or the benchmark scores air.
    assert all((tmp_path / row["image"]).exists() for row in rows)


def test_ocr_set_spans_severities_and_date_formats(tmp_path: Path):
    manifest = generate_ocr_set(tmp_path, count=18, seed=1)
    severities = {s["params"]["severity"] for s in manifest["samples"]}
    patterns = {s["date_pattern"] for s in manifest["samples"]}

    assert severities == {"mild", "moderate", "harsh"}
    assert len(patterns) > 1
    assert patterns <= {pattern for _template, pattern in DATE_TEMPLATES}


def test_ocr_set_covers_expired_and_valid_dates(tmp_path: Path):
    manifest = generate_ocr_set(tmp_path, count=30, seed=2)
    offsets = [s["days_from_reference"] for s in manifest["samples"]]
    assert min(offsets) < 0 < max(offsets)


def test_ocr_ground_truth_dates_match_the_rendered_text(tmp_path: Path):
    """The label must come from what was rendered, not from re-parsing the image."""
    from app.utils.dates import parse_expiry_date

    manifest = generate_ocr_set(tmp_path, count=12, seed=4)
    for sample in manifest["samples"]:
        parsed, _pattern, _ = parse_expiry_date(sample["truth_text"], dayfirst=True)
        assert parsed is not None, sample["truth_text"]
        assert parsed.isoformat() == sample["truth_date"]
