"""Ingestion/preprocessing tests. Skipped wholesale when OpenCV is absent."""

from __future__ import annotations

import pytest

from app.utils.vision import (
    ImageDecodeError,
    crop,
    decode_image_bytes,
    denormalize_box,
    describe,
    encode_jpeg,
    letterbox,
    normalize_box,
    preprocess,
    read_image_file,
    to_rgb,
)

cv2 = pytest.importorskip("cv2", reason="requirements-ml.txt not installed")
np = pytest.importorskip("numpy")


@pytest.fixture()
def frame() -> "np.ndarray":
    """A 200×100 BGR frame with a distinct blue channel for order assertions."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # blue in BGR
    return image


@pytest.fixture()
def png_bytes(frame) -> bytes:  # noqa: ANN001
    ok, buffer = cv2.imencode(".png", frame)
    assert ok
    return buffer.tobytes()


def test_decode_roundtrip(png_bytes, frame):  # noqa: ANN001
    decoded = decode_image_bytes(png_bytes)
    meta = describe(decoded)
    assert (meta.width, meta.height, meta.channels) == (200, 100, 3)
    assert np.array_equal(decoded, frame)


def test_decode_rejects_empty_payload():
    with pytest.raises(ImageDecodeError, match="Empty"):
        decode_image_bytes(b"")


def test_decode_rejects_garbage():
    with pytest.raises(ImageDecodeError, match="decode"):
        decode_image_bytes(b"this is definitely not an image" * 10)


def test_decode_rejects_tiny_frame():
    tiny = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", tiny)
    assert ok
    with pytest.raises(ImageDecodeError, match="too small"):
        decode_image_bytes(buffer.tobytes())


def test_read_missing_file(tmp_path):  # noqa: ANN001
    with pytest.raises(ImageDecodeError, match="not found"):
        read_image_file(tmp_path / "nope.jpg")


def test_read_empty_file(tmp_path):  # noqa: ANN001
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    with pytest.raises(ImageDecodeError, match="Empty file"):
        read_image_file(empty)


def test_read_unicode_path(tmp_path, png_bytes):  # noqa: ANN001
    # cv2.imread returns None here on Windows; read_image_file must not.
    target = tmp_path / "shelf-café-№1.png"
    target.write_bytes(png_bytes)
    assert describe(read_image_file(target)).width == 200


def test_to_rgb_swaps_channel_order(frame):  # noqa: ANN001
    rgb = to_rgb(frame)
    assert rgb[0, 0, 0] == 0 and rgb[0, 0, 2] == 255


def test_letterbox_preserves_aspect_ratio(frame):  # noqa: ANN001
    result = letterbox(frame, size=640)
    assert result.image.shape[:2] == (640, 640)
    assert result.ratio == pytest.approx(640 / 200)
    # 200×100 scaled by 3.2 -> 640×320, so 160px of padding top and bottom.
    assert result.padding[1] == pytest.approx(160.0)
    assert result.padding[0] == pytest.approx(0.0)


def test_preprocess_shape_and_range(frame):  # noqa: ANN001
    tensor = preprocess(frame, size=320)
    assert tensor.shape == (3, 320, 320)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0


def test_preprocess_hwc_without_normalisation(frame):  # noqa: ANN001
    array = preprocess(frame, size=64, normalize=False, chw=False)
    assert array.shape == (64, 64, 3)
    assert float(array.max()) > 1.0


def test_normalize_box_clips_out_of_frame_predictions():
    # YOLO routinely predicts a few pixels outside the frame.
    assert normalize_box((-5, -5, 210, 110), 200, 100) == (0.0, 0.0, 1.0, 1.0)


def test_normalize_denormalize_roundtrip():
    normalised = normalize_box((20, 10, 120, 60), 200, 100)
    assert denormalize_box(normalised, 200, 100) == (20, 10, 120, 60)


def test_normalize_box_rejects_zero_size_frame():
    with pytest.raises(ImageDecodeError):
        normalize_box((0, 0, 1, 1), 0, 100)


def test_crop_respects_frame_bounds(frame):  # noqa: ANN001
    patch = crop(frame, (0.25, 0.2, 0.75, 0.8))
    assert patch.shape[:2] == (60, 100)


def test_crop_padding_clamps_at_edges(frame):  # noqa: ANN001
    patch = crop(frame, (0.0, 0.0, 0.2, 0.2), padding=0.5)
    assert patch.shape[0] > 0 and patch.shape[1] > 0


def test_crop_rejects_degenerate_region(frame):  # noqa: ANN001
    with pytest.raises(ImageDecodeError, match="Degenerate"):
        crop(frame, (0.5, 0.5, 0.5001, 0.5001))


def test_encode_jpeg_is_decodable(frame):  # noqa: ANN001
    assert describe(decode_image_bytes(encode_jpeg(frame))).width == 200
