"""OpenCV image ingestion and preprocessing.

Everything that turns *bytes on the wire* into *an array a model can consume*
lives here, so the detector service never touches I/O and stays unit-testable.

Conventions used throughout:
- OpenCV decodes to **BGR**; Ultralytics/torchvision expect **RGB**. The
  conversion point is explicit (`to_rgb`) rather than implicit anywhere.
- Arrays are `uint8 HWC` until `preprocess()` is called; only that function
  produces float/CHW tensors.
- Failures raise `ImageDecodeError` (a domain error the API maps to 4xx) instead
  of returning `None`, so an unreadable upload can never silently become
  "zero detections".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, List, Sequence, Tuple, Union

from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

    Image = NDArray[np.uint8]
else:  # numpy is a runtime dependency, but keep import cost off module load
    Image = "np.ndarray"  # type: ignore[assignment]

logger = get_logger(__name__)

ImageSource = Union[bytes, bytearray, memoryview, str, Path]

#: Guard against decompression-bomb style uploads (≈ 8K × 8K RGB).
MAX_IMAGE_PIXELS: Final[int] = 64_000_000
#: Anything smaller than this in either dimension cannot hold a readable shelf.
MIN_IMAGE_SIDE: Final[int] = 32

IMAGENET_MEAN: Final[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[Tuple[float, float, float]] = (0.229, 0.224, 0.225)


class ImageDecodeError(ValueError):
    """Raised when bytes/paths cannot be turned into a usable image."""


@dataclass(frozen=True)
class ImageMeta:
    """Dimensions of an ingested frame, kept for de-normalising boxes later."""

    width: int
    height: int
    channels: int

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass(frozen=True)
class LetterboxResult:
    """Output of `letterbox()`, carrying the inverse-transform parameters.

    `ratio` and `padding` are what you need to map model-space coordinates back
    onto the original frame — without them, boxes drift by the pad offset.
    """

    image: "Image"
    ratio: float
    padding: Tuple[float, float]  # (dw, dh) applied to each side
    original: ImageMeta


def _cv2():  # noqa: ANN202 - thin lazy-import shim
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImageDecodeError(
            "opencv-python-headless is not installed — pip install -r requirements-ml.txt"
        ) from exc
    return cv2


def _np():  # noqa: ANN202
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImageDecodeError("numpy is not installed") from exc
    return np


# --------------------------------------------------------------- ingestion --
def decode_image_bytes(data: Union[bytes, bytearray, memoryview]) -> "Image":
    """Decode raw image bytes (an upload or a camera frame) into a BGR array.

    Raises `ImageDecodeError` for empty payloads, corrupt/unsupported encodings,
    absurdly large frames, or images too small to contain a shelf.
    """
    if not data:
        raise ImageDecodeError("Empty image payload")

    cv2 = _cv2()
    np = _np()

    buffer = np.frombuffer(bytes(data), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError(
            "Could not decode image — the file is corrupt or not a supported format "
            "(JPEG, PNG, WebP, BMP)"
        )
    return _validate(image)


def read_image_file(path: Union[str, Path]) -> "Image":
    """Read an image from disk into a BGR array.

    Uses `fromfile` + `imdecode` rather than `cv2.imread`: on Windows, `imread`
    fails silently on non-ASCII paths, which is exactly the kind of bug that
    shows up only on someone else's machine mid-experiment.
    """
    np = _np()
    file_path = Path(path)
    if not file_path.exists():
        raise ImageDecodeError(f"Image not found: {file_path}")
    if not file_path.is_file():
        raise ImageDecodeError(f"Not a file: {file_path}")

    try:
        raw = np.fromfile(str(file_path), dtype=np.uint8)
    except OSError as exc:
        raise ImageDecodeError(f"Could not read {file_path}: {exc}") from exc

    if raw.size == 0:
        raise ImageDecodeError(f"Empty file: {file_path}")

    cv2 = _cv2()
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError(f"Unsupported or corrupt image: {file_path}")
    return _validate(image)


def load_image(source: ImageSource) -> "Image":
    """Ingest from bytes or a path — the single entry point for callers."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return decode_image_bytes(source)
    return read_image_file(source)


def _validate(image: "Image") -> "Image":
    meta = describe(image)
    if meta.pixels > MAX_IMAGE_PIXELS:
        raise ImageDecodeError(
            f"Image is too large ({meta.width}×{meta.height}); "
            f"the limit is {MAX_IMAGE_PIXELS:,} pixels"
        )
    if meta.width < MIN_IMAGE_SIDE or meta.height < MIN_IMAGE_SIDE:
        raise ImageDecodeError(
            f"Image is too small ({meta.width}×{meta.height}); "
            f"each side must be at least {MIN_IMAGE_SIDE}px"
        )
    return image


def describe(image: "Image") -> ImageMeta:
    """Shape metadata for an ingested frame."""
    if image is None or getattr(image, "size", 0) == 0:
        raise ImageDecodeError("Image array is empty")
    if image.ndim == 2:
        height, width = image.shape
        return ImageMeta(width=int(width), height=int(height), channels=1)
    if image.ndim == 3:
        height, width, channels = image.shape
        return ImageMeta(width=int(width), height=int(height), channels=int(channels))
    raise ImageDecodeError(f"Unexpected image shape: {image.shape}")


# ------------------------------------------------------------ colour space --
def to_rgb(image: "Image") -> "Image":
    """BGR (OpenCV) → RGB (torch/Ultralytics). Grayscale and BGRA are handled."""
    cv2 = _cv2()
    meta = describe(image)
    if meta.channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if meta.channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_bgr(image: "Image") -> "Image":
    """RGB → BGR, for handing an array back to OpenCV drawing/encoding calls."""
    cv2 = _cv2()
    meta = describe(image)
    if meta.channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if meta.channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------- preprocessing --
def resize(image: "Image", width: int, height: int) -> "Image":
    """Plain resize, ignoring aspect ratio. Prefer `letterbox()` for detection."""
    cv2 = _cv2()
    meta = describe(image)
    # INTER_AREA is the correct kernel when shrinking; it avoids the aliasing
    # that makes small facings disappear before the detector ever sees them.
    interpolation = (
        cv2.INTER_AREA if (width < meta.width or height < meta.height) else cv2.INTER_LINEAR
    )
    return cv2.resize(image, (int(width), int(height)), interpolation=interpolation)


def letterbox(
    image: "Image",
    size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> LetterboxResult:
    """Resize to a square canvas preserving aspect ratio, padding the remainder.

    This is the transform YOLOv8 applies internally; replicating it here lets the
    benchmark feed the model pre-sized frames and still map boxes back exactly.
    """
    cv2 = _cv2()
    meta = describe(image)

    ratio = min(size / meta.width, size / meta.height)
    new_w, new_h = max(1, round(meta.width * ratio)), max(1, round(meta.height * ratio))
    resized = resize(image, new_w, new_h)

    dw, dh = (size - new_w) / 2.0, (size - new_h) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return LetterboxResult(image=padded, ratio=ratio, padding=(dw, dh), original=meta)


def preprocess(
    image: "Image",
    size: int = 640,
    *,
    to_rgb_order: bool = True,
    normalize: bool = True,
    imagenet_stats: bool = False,
    chw: bool = True,
) -> "Image":
    """Full detector-ready transform: letterbox → RGB → float → CHW.

    `imagenet_stats` applies mean/std normalisation (what the freshness CNN
    wants); YOLOv8 only needs the 0-1 scaling, which is the default.
    """
    np = _np()

    canvas = letterbox(image, size).image
    if to_rgb_order:
        canvas = to_rgb(canvas)

    array = canvas.astype(np.float32)
    if normalize:
        array /= 255.0
        if imagenet_stats:
            array = (array - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
                IMAGENET_STD, dtype=np.float32
            )
    if chw:
        array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return array


# -------------------------------------------------------- coordinate utils --
def normalize_box(
    box: Sequence[float], width: int, height: int, clip: bool = True
) -> Tuple[float, float, float, float]:
    """Pixel xyxy → normalised xyxy, clipped into [0, 1].

    Clipping matters: YOLO regularly predicts boxes a few pixels outside the
    frame, and `BoundingBox` rejects out-of-range values.
    """
    if width <= 0 or height <= 0:
        raise ImageDecodeError(f"Invalid frame size for normalisation: {width}×{height}")

    x1, y1, x2, y2 = (float(v) for v in box[:4])
    values = (x1 / width, y1 / height, x2 / width, y2 / height)
    if not clip:
        return values
    return tuple(min(1.0, max(0.0, v)) for v in values)  # type: ignore[return-value]


def denormalize_box(
    box: Sequence[float], width: int, height: int
) -> Tuple[int, int, int, int]:
    """Normalised xyxy → integer pixel xyxy, for cropping or drawing."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def crop(image: "Image", box: Sequence[float], padding: float = 0.0) -> "Image":
    """Crop a normalised box out of a frame (used for freshness/OCR sub-models).

    `padding` expands the box by a fraction of its size — packaging dates often
    sit just outside the detected product edge.
    """
    meta = describe(image)
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if padding:
        pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
        x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    px1, py1, px2, py2 = denormalize_box((x1, y1, x2, y2), meta.width, meta.height)
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(meta.width, px2), min(meta.height, py2)
    if px2 <= px1 or py2 <= py1:
        raise ImageDecodeError(f"Degenerate crop region: {(px1, py1, px2, py2)}")
    return image[py1:py2, px1:px2]


def encode_jpeg(image: "Image", quality: int = 90) -> bytes:
    """Encode a BGR array back to JPEG bytes (annotated-frame responses)."""
    cv2 = _cv2()
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ImageDecodeError("JPEG encoding failed")
    return buffer.tobytes()


def list_images(directory: Union[str, Path]) -> List[Path]:
    """Sorted image files in a directory — the benchmark's frame source."""
    root = Path(directory)
    if not root.exists():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes and p.is_file())


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ImageDecodeError",
    "ImageMeta",
    "ImageSource",
    "LetterboxResult",
    "MAX_IMAGE_PIXELS",
    "MIN_IMAGE_SIDE",
    "crop",
    "decode_image_bytes",
    "denormalize_box",
    "describe",
    "encode_jpeg",
    "letterbox",
    "list_images",
    "load_image",
    "normalize_box",
    "preprocess",
    "read_image_file",
    "resize",
    "to_bgr",
    "to_rgb",
]
