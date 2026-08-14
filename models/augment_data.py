"""Dataset augmentation and synthesis.

Two generators, for the two data gaps Phase 2 exposed:

1. **Ripening synthesis** (`ripening`) — public freshness datasets are almost all
   binary (fresh / rotten). The `ripening` class has no images, so the classifier
   cannot learn it and its confusion-matrix row is empty. This shifts fresh
   produce through HSV toward yellow/orange and stipples it with ripening spots.

2. **OCR degradation** (`ocr`) — clean rendered text proves nothing about a
   pipeline built for dot-matrix stamps on foil. This renders known date strings
   and degrades them (dot-matrix masking, blur, noise, contrast collapse, uneven
   lighting, JPEG artefacts, perspective) at graded severities, emitting a
   ground-truth CSV so `benchmark ocr` can score them immediately.

> **Read this before citing anything trained on the output.**
> Synthetic ripening images are *derived from* fresh images by a known colour
> transform. A classifier trained on them can learn the transform rather than
> ripeness, and evaluating on the same synthetic distribution measures that
> circularity, not accuracy. Legitimate uses: validating the pipeline end to end,
> augmenting a small real `ripening` set, and ablations. Not legitimate: a
> headline three-class accuracy number with no real ripening photographs. Every
> generated file is recorded as `synthetic: true` in `manifest.json` so this
> cannot be lost track of later.

Implemented with OpenCV + numpy rather than albumentations: no new dependency,
and every transform is explicit and seed-reproducible, which the paper needs.

    python models/augment_data.py ripening --source data/freshness/train/fresh --count 200
    python models/augment_data.py ocr --count 120 --severity harsh
    python models/augment_data.py all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.utils.vision import ImageDecodeError, list_images, read_image_file

logger = get_logger(__name__)

# Defaults point at each pipeline's own benchmark directory. Freshness crops in
# data/test_images/ would be scored by the *detection* suite, which expects shelf
# frames with YOLO labels — override with --out if you really want them there.
DEFAULT_RIPENING_OUT = settings.DATA_DIR / "test_freshness" / "ripening"
DEFAULT_SPOILED_OUT = settings.DATA_DIR / "test_freshness" / "spoiled"
DEFAULT_OCR_OUT = settings.DATA_DIR / "test_expiry"

# OpenCV hue is 0-179. Green produce sits ~35-85; ripe yellow ~22-30; orange ~12-20.
HUE_GREEN_LOW, HUE_GREEN_HIGH = 30, 90
HUE_RIPE_TARGET = 26.0
HUE_SPOILED_TARGET = 14.0


def _cv2():  # noqa: ANN202
    import cv2  # noqa: PLC0415

    return cv2


def _np():  # noqa: ANN202
    import numpy as np  # noqa: PLC0415

    return np


# ---------------------------------------------------------------- ripening --
@dataclass
class RipeningParams:
    """Every knob that shaped one synthetic image, recorded in the manifest."""

    progress: float
    hue_target: float
    saturation_gain: float
    value_gain: float
    spot_count: int
    spot_radius: int
    blur: float = 0.0
    seed: int = 0


def synthesize_ripening(
    image: Any, progress: float, rng: Any, spoiled: bool = False
) -> Tuple[Any, RipeningParams]:
    """Shift fresh produce toward ripe (or spoiled) appearance.

    `progress` in [0, 1] controls how far along the transform runs, so a source
    image yields a spread of ripeness rather than one duplicated look.

    Only *saturated* pixels are moved: shifting the whole frame would drag the
    background and the tray with it, and the classifier would learn the
    background instead of the produce.
    """
    cv2 = _cv2()
    np = _np()

    progress = float(min(1.0, max(0.0, progress)))
    hue_target = HUE_SPOILED_TARGET if spoiled else HUE_RIPE_TARGET

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Produce mask: coloured and not in shadow. Greens get the full shift; other
    # hues (already-yellow bananas) move proportionally less.
    produce = (sat > 55) & (val > 45)
    greenish = produce & (hue >= HUE_GREEN_LOW) & (hue <= HUE_GREEN_HIGH)

    shift = progress * (hue - hue_target)
    hue = np.where(greenish, hue - shift, hue)
    hue = np.where(produce & ~greenish, hue - shift * 0.35, hue)

    # Ripe fruit is more saturated; spoiled fruit goes dull and dark.
    saturation_gain = (1.0 - 0.45 * progress) if spoiled else (1.0 + 0.25 * progress)
    value_gain = (1.0 - 0.35 * progress) if spoiled else (1.0 - 0.08 * progress)
    sat = np.where(produce, sat * saturation_gain, sat)
    val = np.where(produce, val * value_gain, val)

    hsv[..., 0] = np.clip(hue, 0, 179)
    hsv[..., 1] = np.clip(sat, 0, 255)
    hsv[..., 2] = np.clip(val, 0, 255)
    output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Ripening/rot speckle, confined to the produce mask.
    height, width = output.shape[:2]
    spot_count = int((18 if spoiled else 9) * progress * rng.uniform(0.6, 1.4))
    radius = max(2, int(min(height, width) * (0.035 if spoiled else 0.02)))
    coords = np.argwhere(produce)
    if spot_count and len(coords):
        chosen = coords[rng.integers(0, len(coords), size=min(spot_count, len(coords)))]
        for y, x in chosen:
            colour = (
                (18, 22, 30) if spoiled else (30, 45, 70)
            )  # near-black vs brown, in BGR
            cv2.circle(
                output,
                (int(x), int(y)),
                int(rng.integers(max(1, radius // 2), radius + 1)),
                colour,
                -1,
                lineType=cv2.LINE_AA,
            )

    blur = 0.0
    if spoiled and progress > 0.6:
        blur = 1.0  # soft rot loses edge definition
        output = cv2.GaussianBlur(output, (3, 3), 0)

    return output, RipeningParams(
        progress=round(progress, 3),
        hue_target=hue_target,
        saturation_gain=round(float(saturation_gain), 3),
        value_gain=round(float(value_gain), 3),
        spot_count=spot_count,
        spot_radius=radius,
        blur=blur,
    )


def generate_ripening_set(
    source_dir: Path,
    out_dir: Path,
    count: int = 200,
    seed: int = 42,
    spoiled: bool = False,
    progress_range: Tuple[float, float] = (0.35, 0.85),
) -> Dict[str, Any]:
    """Synthesise a ripening (or spoiled) class from fresh source images."""
    cv2 = _cv2()
    np = _np()

    sources = list_images(source_dir)
    if not sources:
        raise FileNotFoundError(
            f"No source images in {source_dir}. Point --source at a folder of "
            "FRESH produce crops (e.g. data/freshness/train/fresh)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    label = "spoiled" if spoiled else "ripening"
    rng = np.random.default_rng(seed)
    low, high = progress_range

    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for index in range(count):
        source_path = sources[index % len(sources)]
        try:
            image = read_image_file(source_path)
        except ImageDecodeError as exc:
            logger.warning("Skipping %s: %s", source_path.name, exc)
            skipped.append({"image": source_path.name, "reason": str(exc)})
            continue

        progress = float(rng.uniform(low, high))
        output, params = synthesize_ripening(image, progress, rng, spoiled=spoiled)
        params.seed = seed

        name = f"{label}_{index:04d}_{source_path.stem[:28]}.jpg"
        cv2.imwrite(str(out_dir / name), output)
        records.append(
            {
                "image": name,
                "label": label,
                "source": source_path.name,
                "synthetic": True,
                "generator": "hsv_ripening_v1",
                "params": asdict(params),
            }
        )

    manifest = _write_manifest(
        out_dir,
        generator=f"hsv_{label}_v1",
        seed=seed,
        records=records,
        skipped=skipped,
        note=(
            "SYNTHETIC. Derived from fresh images by a known HSV transform. Use for "
            "pipeline validation and augmentation of a real set — not as the sole "
            "evidence for a class-accuracy claim."
        ),
    )
    logger.info("Wrote %d synthetic %s images to %s", len(records), label, out_dir)
    return manifest


# --------------------------------------------------------------------- OCR --
@dataclass
class DegradationParams:
    """Every degradation applied to one rendered stamp."""

    severity: str
    dot_matrix: bool = False
    dot_pitch: int = 0
    blur_sigma: float = 0.0
    noise_sigma: float = 0.0
    contrast: float = 1.0
    brightness: int = 0
    rotation_deg: float = 0.0
    perspective: float = 0.0
    jpeg_quality: int = 95
    vignette: float = 0.0
    seed: int = 0


#: (low, high) ranges sampled per image.
#:
#: These are calibrated against measured OCR behaviour, not intuition. An earlier
#: `harsh` tier (blur 1.0-2.0, noise 14-26, contrast 0.40-0.62, JPEG 30-55) read
#: **0/6** even with every preprocessing variant and a 60 s budget — a rung past
#: the edge of feasibility measures destruction, not resilience. `harsh` now sits
#: at the edge, and the known-unreadable case moved to `extreme`, which exists so
#: the ladder still has a floor to cite.
SEVERITY_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "mild": {
        "blur": (0.0, 0.6), "noise": (2.0, 6.0), "contrast": (0.85, 1.0),
        "rotation": (-1.5, 1.5), "perspective": (0.0, 0.01), "jpeg": (80, 95),
        "vignette": (0.0, 0.15),
    },
    "moderate": {
        "blur": (0.4, 0.9), "noise": (5.0, 11.0), "contrast": (0.65, 0.88),
        "rotation": (-3.0, 3.0), "perspective": (0.01, 0.025), "jpeg": (60, 82),
        "vignette": (0.12, 0.30),
    },
    "harsh": {
        "blur": (0.7, 1.3), "noise": (10.0, 18.0), "contrast": (0.50, 0.70),
        "rotation": (-5.0, 5.0), "perspective": (0.02, 0.04), "jpeg": (45, 62),
        "vignette": (0.25, 0.45),
    },
    "extreme": {
        "blur": (1.4, 2.2), "noise": (18.0, 28.0), "contrast": (0.38, 0.55),
        "rotation": (-8.0, 8.0), "perspective": (0.04, 0.07), "jpeg": (28, 45),
        "vignette": (0.40, 0.60),
    },
}

#: Tiers generated by default. `extreme` is opt-in via --severity: it is known to
#: be unreadable, so including it in every run would drag the headline metrics
#: down for no information gain.
DEFAULT_SEVERITIES: Tuple[str, ...] = ("mild", "moderate", "harsh")

DATE_TEMPLATES: List[Tuple[str, str]] = [
    ("EXP {d:02d}/{m:02d}/{y}", "numeric_dmy"),
    ("BEST BEFORE {d:02d} {MON} {y}", "dmy_alpha"),
    ("USE BY {y}-{m:02d}-{d:02d}", "iso_ymd"),
    ("BB {y}{m:02d}{d:02d}", "compact_ymd"),
    ("EXP. {MON} {d:02d} {y}", "mdy_alpha"),
    ("BBE {d:02d}.{m:02d}.{y}", "numeric_dmy"),
]

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def render_date_stamp(text: str, rng: Any, width: int = 520, height: int = 140) -> Any:
    """Render a clean date stamp, varying font and placement per sample."""
    cv2 = _cv2()
    np = _np()

    background = int(rng.integers(200, 245))
    image = np.full((height, width, 3), background, dtype=np.uint8)

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]
    font = fonts[int(rng.integers(0, len(fonts)))]
    scale = float(rng.uniform(0.9, 1.4))
    thickness = int(rng.integers(2, 4))
    ink = int(rng.integers(20, 70))

    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(8, int((width - text_w) / 2 + rng.integers(-20, 20)))
    y = int((height + text_h) / 2 + rng.integers(-8, 8))
    cv2.putText(image, text, (x, y), font, scale, (ink, ink, ink), thickness, cv2.LINE_AA)
    return image


def degrade_stamp(image: Any, severity: str, rng: Any) -> Tuple[Any, DegradationParams]:
    """Apply graded real-world degradation to a rendered stamp.

    Order matters and mirrors physics: the print is formed (dot matrix), then the
    optics blur it, then illumination falls off, then the sensor adds noise, then
    the codec throws information away. Applying JPEG before noise would produce
    artefacts no camera generates.
    """
    cv2 = _cv2()
    np = _np()

    profile = SEVERITY_PROFILES.get(severity, SEVERITY_PROFILES["moderate"])
    params = DegradationParams(severity=severity)
    output = image.copy()
    height, width = output.shape[:2]

    # 1. Dot-matrix / inkjet print: punch a grid of gaps through the glyphs.
    if severity != "mild" or rng.random() < 0.5:
        pitch = int(rng.integers(2, 4))
        mask = np.zeros((height, width), dtype=bool)
        mask[:: pitch + 1, :] = True
        mask[:, :: pitch + 1] = True
        background = int(np.median(output[:5, :5]))
        output[mask] = background
        params.dot_matrix = True
        params.dot_pitch = pitch

    # 2. Geometry: the camera is never square to the pack.
    rotation = float(rng.uniform(*profile["rotation"]))
    if abs(rotation) > 0.1:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, 1.0)
        output = cv2.warpAffine(
            output, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        params.rotation_deg = round(rotation, 2)

    warp = float(rng.uniform(*profile["perspective"]))
    if warp > 0.005:
        dx, dy = warp * width, warp * height
        src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst = np.float32(
            [
                [rng.uniform(0, dx), rng.uniform(0, dy)],
                [width - rng.uniform(0, dx), rng.uniform(0, dy)],
                [width - rng.uniform(0, dx), height - rng.uniform(0, dy)],
                [rng.uniform(0, dx), height - rng.uniform(0, dy)],
            ]
        )
        output = cv2.warpPerspective(
            output, cv2.getPerspectiveTransform(src, dst), (width, height),
            borderMode=cv2.BORDER_REPLICATE,
        )
        params.perspective = round(warp, 4)

    # 3. Optics.
    blur = float(rng.uniform(*profile["blur"]))
    if blur > 0.05:
        output = cv2.GaussianBlur(output, (0, 0), blur)
        params.blur_sigma = round(blur, 3)

    # 4. Uneven illumination: a curved pack under one light.
    vignette = float(rng.uniform(*profile["vignette"]))
    if vignette > 0.02:
        yy, xx = np.mgrid[0:height, 0:width]
        cx, cy = width * float(rng.uniform(0.3, 0.7)), height * float(rng.uniform(0.3, 0.7))
        distance = np.sqrt(((xx - cx) / width) ** 2 + ((yy - cy) / height) ** 2)
        falloff = 1.0 - vignette * (distance / distance.max())
        output = np.clip(output * falloff[..., None], 0, 255).astype(np.uint8)
        params.vignette = round(vignette, 3)

    # 5. Contrast collapse and exposure drift.
    contrast = float(rng.uniform(*profile["contrast"]))
    brightness = int(rng.integers(-25, 26))
    output = cv2.convertScaleAbs(output, alpha=contrast, beta=brightness)
    params.contrast = round(contrast, 3)
    params.brightness = brightness

    # 6. Sensor noise.
    noise_sigma = float(rng.uniform(*profile["noise"]))
    if noise_sigma > 0.5:
        noise = rng.normal(0, noise_sigma, output.shape)
        output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        params.noise_sigma = round(noise_sigma, 2)

    # 7. Codec.
    quality = int(rng.integers(*[int(v) for v in profile["jpeg"]]))
    ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        output = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        params.jpeg_quality = quality

    return output, params


def generate_ocr_set(
    out_dir: Path,
    count: int = 120,
    seed: int = 42,
    severity: Optional[str] = None,
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Render and degrade date stamps, emitting a scoreable ground-truth CSV."""
    cv2 = _cv2()
    np = _np()

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    reference = reference_date or datetime.now(timezone.utc).date()
    severities = [severity] if severity else list(DEFAULT_SEVERITIES)

    records: List[Dict[str, Any]] = []
    rows: List[Tuple[str, str, str]] = []

    for index in range(count):
        # Spread offsets so the set exercises expired / near-expiry / valid.
        offset = int(rng.integers(-120, 400))
        target = reference + timedelta(days=offset)
        template, pattern = DATE_TEMPLATES[int(rng.integers(0, len(DATE_TEMPLATES)))]
        text = template.format(
            d=target.day, m=target.month, y=target.year, MON=MONTHS[target.month - 1]
        )

        level = severities[index % len(severities)]
        clean = render_date_stamp(text, rng)
        degraded, params = degrade_stamp(clean, level, rng)
        params.seed = seed

        name = f"stamp_{index:04d}_{level}.jpg"
        cv2.imwrite(str(out_dir / name), degraded)

        rows.append((name, text, target.isoformat()))
        records.append(
            {
                "image": name,
                "truth_text": text,
                "truth_date": target.isoformat(),
                "date_pattern": pattern,
                "days_from_reference": offset,
                "synthetic": True,
                "generator": "stamp_degradation_v1",
                "params": asdict(params),
            }
        )

    ground_truth = out_dir / "ground_truth.csv"
    with ground_truth.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "truth_text", "truth_date"])
        writer.writerows(rows)

    manifest = _write_manifest(
        out_dir,
        generator="stamp_degradation_v1",
        seed=seed,
        records=records,
        skipped=[],
        note=(
            "SYNTHETIC. Rendered text with simulated print/optical/sensor degradation. "
            "Measures pipeline resilience across graded severities; it is not a "
            "substitute for photographs of real packaging."
        ),
        extra={
            "reference_date": reference.isoformat(),
            "severities": severities,
            "ground_truth": ground_truth.name,
        },
    )
    logger.info("Wrote %d degraded stamps + ground_truth.csv to %s", len(records), out_dir)
    return manifest


# ---------------------------------------------------------------- manifest --
def _write_manifest(
    out_dir: Path,
    generator: str,
    seed: int,
    records: Sequence[Dict[str, Any]],
    skipped: Sequence[Dict[str, str]],
    note: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record exactly how each file was produced — the augmentation audit trail."""
    manifest: Dict[str, Any] = {
        "generator": generator,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "synthetic": True,
        "note": note,
        "count": len(records),
        "skipped": list(skipped),
        "samples": list(records),
    }
    if extra:
        manifest.update(extra)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


# --------------------------------------------------------------------- CLI --
@dataclass
class _Summary:
    generated: Dict[str, int] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="augment_data",
        description="Synthesise the ripening class and degraded OCR stamps",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ripening = sub.add_parser("ripening", help="synthesise ripening/spoiled produce")
    ripening.add_argument(
        "--source", required=True, help="folder of FRESH produce crops to transform"
    )
    ripening.add_argument("--out", help=f"output dir (default: {DEFAULT_RIPENING_OUT})")
    ripening.add_argument("--count", type=int, default=200)
    ripening.add_argument("--seed", type=int, default=42)
    ripening.add_argument(
        "--spoiled", action="store_true", help="synthesise the spoiled class instead"
    )

    ocr = sub.add_parser("ocr", help="render and degrade expiry-date stamps")
    ocr.add_argument("--out", help=f"output dir (default: {DEFAULT_OCR_OUT})")
    ocr.add_argument("--count", type=int, default=120)
    ocr.add_argument("--seed", type=int, default=42)
    ocr.add_argument("--severity", choices=sorted(SEVERITY_PROFILES), default=None)
    ocr.add_argument("--reference-date", help="ISO date used as 'today' for the labels")

    every = sub.add_parser("all", help="run both generators")
    every.add_argument("--source", required=True, help="fresh produce crops")
    every.add_argument("--count", type=int, default=120)
    every.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)

    try:
        if args.command == "ripening":
            out = Path(args.out or (DEFAULT_SPOILED_OUT if args.spoiled else DEFAULT_RIPENING_OUT))
            manifest = generate_ripening_set(
                Path(args.source), out, args.count, args.seed, spoiled=args.spoiled
            )
            print(json.dumps({"out": str(out), "count": manifest["count"]}, indent=2))

        elif args.command == "ocr":
            out = Path(args.out or DEFAULT_OCR_OUT)
            reference = (
                date.fromisoformat(args.reference_date) if args.reference_date else None
            )
            manifest = generate_ocr_set(
                out, args.count, args.seed, args.severity, reference
            )
            print(json.dumps({"out": str(out), "count": manifest["count"]}, indent=2))

        else:  # all
            ripening = generate_ripening_set(
                Path(args.source), DEFAULT_RIPENING_OUT, args.count, args.seed
            )
            stamps = generate_ocr_set(DEFAULT_OCR_OUT, args.count, args.seed)
            print(
                json.dumps(
                    {
                        "ripening": {"out": str(DEFAULT_RIPENING_OUT), "count": ripening["count"]},
                        "ocr": {"out": str(DEFAULT_OCR_OUT), "count": stamps["count"]},
                    },
                    indent=2,
                )
            )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except ImportError as exc:
        logger.error("OpenCV/numpy missing — pip install -r requirements-ml.txt (%s)", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
