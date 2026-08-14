"""OpenCV preprocessing ops used by the OCR pipeline."""

from __future__ import annotations

import pytest

from app.utils.vision import (
    adaptive_threshold,
    denoise,
    describe,
    enhance_contrast,
    morphology,
    otsu_threshold,
    sharpen,
    to_grayscale,
    upscale,
)

cv2 = pytest.importorskip("cv2", reason="requirements-ml.txt not installed")
np = pytest.importorskip("numpy")


@pytest.fixture()
def stamp() -> "np.ndarray":
    """A 160×60 light panel with dark 'ink' — stands in for a date stamp."""
    image = np.full((60, 160, 3), 220, dtype=np.uint8)
    image[20:40, 20:140] = 40
    return image


@pytest.fixture()
def dotted() -> "np.ndarray":
    """Dot-matrix print: disconnected 2px dots, as an inkjet head produces."""
    image = np.full((60, 160), 230, dtype=np.uint8)
    for x in range(20, 140, 4):
        for y in range(24, 40, 4):
            image[y : y + 2, x : x + 2] = 30
    return image


def test_grayscale_is_single_channel_and_idempotent(stamp):  # noqa: ANN001
    gray = to_grayscale(stamp)
    assert gray.ndim == 2
    assert np.array_equal(to_grayscale(gray), gray)


def test_otsu_produces_a_binary_image(stamp):  # noqa: ANN001
    binary = otsu_threshold(stamp)
    assert set(np.unique(binary).tolist()) <= {0, 255}
    # Ink is dark, so the default (non-inverted) pass leaves it black.
    assert binary[30, 80] == 0


def test_otsu_invert_flips_polarity(stamp):  # noqa: ANN001
    assert otsu_threshold(stamp, invert=True)[30, 80] == 255


def test_adaptive_threshold_snaps_even_block_size_instead_of_crashing(stamp):  # noqa: ANN001
    # cv2 raises on an even blockSize; a config typo must not 500 the request.
    result = adaptive_threshold(stamp, block_size=30)
    assert result.shape == (60, 160)


def test_adaptive_threshold_handles_uneven_lighting():
    gradient = np.tile(np.linspace(40, 240, 160, dtype=np.uint8), (60, 1))
    gradient[25:35, 30:130] = 10
    binary = adaptive_threshold(gradient, block_size=31, offset=10)
    # A global threshold would lose the dark bar in the bright half; local does not.
    assert binary[30, 120] == 0


def test_morphology_close_bridges_dot_matrix_gaps(dotted):  # noqa: ANN001
    binary = otsu_threshold(dotted, invert=True)  # ink -> white
    before = int((binary > 0).sum())
    closed = morphology(binary, operation="close", kernel_size=3)
    # Closing fills the gaps between dots, so ink coverage grows.
    assert int((closed > 0).sum()) > before


def test_morphology_dilate_and_erode_are_inverse_in_direction(dotted):  # noqa: ANN001
    binary = otsu_threshold(dotted, invert=True)
    dilated = int((morphology(binary, "dilate", 2) > 0).sum())
    eroded = int((morphology(binary, "erode", 2) > 0).sum())
    assert eroded < int((binary > 0).sum()) <= dilated


def test_morphology_rejects_unknown_operation(dotted):  # noqa: ANN001
    with pytest.raises(ValueError, match="Unknown morphology"):
        morphology(dotted, operation="sparkle")


def test_enhance_contrast_widens_the_histogram():
    flat = np.full((60, 160), 128, dtype=np.uint8)
    flat[20:40, 20:140] = 150  # low-contrast ink
    assert enhance_contrast(flat).std() > flat.std()


def test_denoise_removes_salt_and_pepper(stamp):  # noqa: ANN001
    noisy = to_grayscale(stamp).copy()
    noisy[5::7, 5::7] = 0
    cleaned = denoise(noisy, strength=3)
    assert int((cleaned == 0).sum()) < int((noisy == 0).sum())


def test_sharpen_preserves_shape(stamp):  # noqa: ANN001
    assert sharpen(to_grayscale(stamp)).shape == (60, 160)


def test_upscale_respects_min_height(stamp):  # noqa: ANN001
    result = upscale(to_grayscale(stamp), factor=1.0, min_height=240)
    assert describe(result).height >= 240


def test_upscale_is_a_noop_when_already_large_enough(stamp):  # noqa: ANN001
    gray = to_grayscale(stamp)
    assert upscale(gray, factor=1.0, min_height=10) is gray


def test_upscale_refuses_to_exceed_the_pixel_budget():
    big = np.zeros((4000, 4000), dtype=np.uint8)
    # 4000×4000×16 would blow past MAX_IMAGE_PIXELS; the op must decline.
    assert upscale(big, factor=4.0).shape == (4000, 4000)
