"""Expiry OCR service: variant strategy, parsing, and graceful degradation.

EasyOCR itself is stubbed — these tests cover the logic *around* the recogniser
(variant selection, de-duplication, line joining, status verdicts, failure
handling), which is where the behaviour that matters lives.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.models.enums import ExpiryStatus
from app.services.ocr_expiry import (
    VARIANTS,
    ExpiryOCRService,
    OCRUnavailableError,
    parse_single,
    parse_texts,
    polygon_to_bbox,
)

np = pytest.importorskip("numpy", reason="requirements-ml.txt not installed")
pytest.importorskip("cv2")

REFERENCE = date(2026, 8, 14)


@pytest.fixture()
def panel() -> "np.ndarray":
    image = np.full((80, 240, 3), 210, dtype=np.uint8)
    image[30:50, 20:220] = 35
    return image


class _StubReader:
    """Stands in for `easyocr.Reader`, returning per-call scripted results."""

    def __init__(self, script) -> None:  # noqa: ANN001
        self.script = list(script)
        self.calls = 0

    def readtext(self, _image):  # noqa: ANN001, ANN201
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        result = self.script[index]
        if isinstance(result, Exception):
            raise result
        return result


def _service(script, variants=None) -> ExpiryOCRService:  # noqa: ANN001
    service = ExpiryOCRService(variants=variants or ["raw", "otsu", "adaptive_close"])
    service._reader = _StubReader(script)
    return service


def _line(text: str, confidence: float = 0.9):  # noqa: ANN202
    return ([[0, 0], [100, 0], [100, 30], [0, 30]], text, confidence)


# --------------------------------------------------------- text-only parse --
@pytest.mark.parametrize(
    ("text", "expected_date", "expected_status"),
    [
        ("EXP 12/09/2026", date(2026, 9, 12), ExpiryStatus.VALID),
        ("USE BY 2026-08-10", date(2026, 8, 10), ExpiryStatus.EXPIRED),
        ("BEST BEFORE 18/08/2026", date(2026, 8, 18), ExpiryStatus.NEAR_EXPIRY),
        ("BB 20260821", date(2026, 8, 21), ExpiryStatus.NEAR_EXPIRY),
        ("BEST BEFORE: 3O NOV 2O26", date(2026, 11, 30), ExpiryStatus.VALID),
    ],
)
def test_status_verdicts(text, expected_date, expected_status):  # noqa: ANN001
    result = parse_single(text, REFERENCE)
    assert result.parsed_date == expected_date
    assert result.status is expected_status


def test_boundary_exactly_seven_days_is_near_expiry():
    # The spec draws the line at "<= 7 days"; day 7 must not read as VALID.
    assert parse_single("EXP 21/08/2026", REFERENCE).status is ExpiryStatus.NEAR_EXPIRY


def test_boundary_eight_days_is_valid():
    assert parse_single("EXP 22/08/2026", REFERENCE).status is ExpiryStatus.VALID


def test_today_counts_as_near_expiry_not_expired():
    result = parse_single("EXP 14/08/2026", REFERENCE)
    assert result.days_remaining == 0
    assert result.status is ExpiryStatus.NEAR_EXPIRY


def test_unparseable_text_is_unreadable_not_an_error():
    result = parse_single("~~ smudged ~~", REFERENCE)
    assert result.parsed_date is None
    assert result.status is ExpiryStatus.UNREADABLE
    assert result.matched_pattern is None


def test_parse_texts_handles_empty_and_blank_input():
    assert parse_texts([], REFERENCE) == []
    assert parse_texts([""], REFERENCE)[0].status is ExpiryStatus.UNREADABLE


# ------------------------------------------------------------ image extract --
def test_extract_returns_best_dated_candidate(panel):  # noqa: ANN001
    service = _service([[_line("EXP 12/09/2026", 0.95)]])
    result = service.extract_expiry(panel, reference_date=REFERENCE)

    assert result.found_date is True
    assert result.best is not None
    assert result.best.parsed_date == date(2026, 9, 12)
    assert result.best.status is ExpiryStatus.VALID
    assert result.variant_used == "raw"
    assert result.latency_ms > 0
    assert "EXP 12/09/2026" in result.raw_text


def test_expired_read_outranks_a_valid_one(panel):  # noqa: ANN001
    # A pack showing both a print date and an expiry: the decisive read wins,
    # even though the other line is more confident.
    service = _service([[_line("EXP 01/01/2020", 0.7), _line("EXP 01/01/2030", 0.99)]])
    best = service.extract_expiry(panel, reference_date=REFERENCE).best
    assert best is not None and best.status is ExpiryStatus.EXPIRED


def test_falls_through_variants_until_a_date_appears(panel):  # noqa: ANN001
    service = _service(
        [
            [],                                   # raw: nothing
            [_line("|||", 0.9)],                  # otsu: garbage
            [_line("BB 20260818", 0.88)],         # adaptive_close: the dot-matrix win
        ]
    )
    result = service.extract_expiry(panel, reference_date=REFERENCE)
    assert result.variant_used == "adaptive_close"
    assert result.variants_tried == ["raw", "otsu", "adaptive_close"]
    assert result.best.parsed_date == date(2026, 8, 18)


def test_stops_early_on_a_confident_date(panel):  # noqa: ANN001
    service = _service([[_line("EXP 12/09/2026", 0.95)], [_line("EXP 01/01/2020", 0.99)]])
    result = service.extract_expiry(panel, reference_date=REFERENCE)
    # The second variant never ran, so the stale date never entered the result.
    assert result.variants_tried == ["raw"]
    assert service._reader.calls == 1


def test_low_confidence_lines_are_discarded(panel, monkeypatch):  # noqa: ANN001
    from app.core.config import settings

    monkeypatch.setattr(settings, "OCR_MIN_CONFIDENCE", 0.5)
    service = _service([[_line("EXP 12/09/2026", 0.10)]], variants=["raw"])
    result = service.extract_expiry(panel, reference_date=REFERENCE)
    assert result.lines == []
    assert result.found_date is False


def test_date_split_across_two_lines_is_joined(panel):  # noqa: ANN001
    service = _service([[_line("BEST BEFORE", 0.9), _line("12 09 2026", 0.85)]], variants=["raw"])
    result = service.extract_expiry(panel, reference_date=REFERENCE)
    assert result.found_date is True
    assert result.best.parsed_date == date(2026, 9, 12)


def test_duplicate_reads_across_variants_are_deduplicated(panel):  # noqa: ANN001
    service = _service(
        [[_line("EXP 12/09/2026", 0.4)], [_line("EXP 12/09/2026", 0.45)]],
        variants=["raw", "otsu"],
    )
    result = service.extract_expiry(panel, reference_date=REFERENCE)
    dated = [e for e in result.extractions if e.parsed_date]
    assert len(dated) == 1


def test_unreadable_image_yields_a_result_not_an_exception(panel):  # noqa: ANN001
    service = _service([[], [], []])
    result = service.extract_expiry(panel, reference_date=REFERENCE)
    assert result.extractions == []
    assert result.best is None
    assert result.found_date is False
    assert result.raw_text == ""


def test_one_failing_variant_does_not_abort_the_read(panel):  # noqa: ANN001
    service = _service(
        [RuntimeError("recogniser exploded"), [_line("EXP 12/09/2026", 0.9)]],
        variants=["raw", "otsu"],
    )
    result = service.extract_expiry(panel, reference_date=REFERENCE)
    assert result.found_date is True


def test_malformed_ocr_rows_are_skipped(panel):  # noqa: ANN001
    service = _service([[("bad", "row"), _line("EXP 12/09/2026", 0.9)]], variants=["raw"])
    result = service.extract_expiry(panel, reference_date=REFERENCE)
    assert result.found_date is True
    assert len(result.lines) == 1


def test_missing_reader_raises_unavailable(panel):  # noqa: ANN001
    service = ExpiryOCRService()
    service._reader = None
    service.load = lambda: False  # type: ignore[method-assign]
    service._load_failure = "easyocr is not installed"
    with pytest.raises(OCRUnavailableError, match="easyocr"):
        service.extract_expiry(panel)


def test_time_budget_stops_the_variant_sweep(panel, monkeypatch):  # noqa: ANN001
    """An unreadable crop must not cost the full sweep.

    Regression guard: a blank panel triggered every variant (~11 s on CPU)
    because nothing ever satisfied the early-stop rule — the worst case landed
    on exactly the frames that produce nothing.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "OCR_TIME_BUDGET_MS", 0.0)
    service = _service([[], [], [], [], []], variants=["raw", "otsu", "adaptive_close"])
    result = service.extract_expiry(panel, reference_date=REFERENCE)

    assert result.variants_tried == ["raw"]  # budget checked before variant 2
    assert service._reader.calls == 1


def test_generous_budget_allows_the_full_sweep(panel, monkeypatch):  # noqa: ANN001
    from app.core.config import settings

    monkeypatch.setattr(settings, "OCR_TIME_BUDGET_MS", 60_000.0)
    service = _service([[], [], []], variants=["raw", "otsu", "adaptive_close"])
    result = service.extract_expiry(panel, reference_date=REFERENCE)
    assert result.variants_tried == ["raw", "otsu", "adaptive_close"]


def test_unknown_variant_names_fall_back_to_raw():
    service = ExpiryOCRService(variants=["does_not_exist"])
    assert service.variants == ["raw"]


def test_every_configured_variant_runs_on_a_real_image(panel):  # noqa: ANN001
    # Guards the preprocessing chain itself: each variant must return a usable
    # array for a real frame, not raise on dtype/channel assumptions.
    for name, fn in VARIANTS.items():
        output = fn(panel)
        assert output is not None and output.size > 0, name


# -------------------------------------------------------------- bbox helper --
def test_polygon_to_bbox_normalises_and_clips():
    bbox = polygon_to_bbox([[10, 5], [110, 5], [110, 45], [10, 45]])
    assert bbox is not None
    assert 0.0 <= bbox.x1 < bbox.x2 <= 1.0
    assert 0.0 <= bbox.y1 < bbox.y2 <= 1.0


def test_polygon_to_bbox_rejects_degenerate_and_malformed():
    assert polygon_to_bbox([[5, 5], [5, 5], [5, 5], [5, 5]]) is None
    assert polygon_to_bbox([]) is None
    assert polygon_to_bbox([["x", "y"]]) is None
