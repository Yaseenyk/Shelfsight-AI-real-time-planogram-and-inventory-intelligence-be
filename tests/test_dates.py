from datetime import date

import pytest

from app.utils.dates import (
    MAX_PLAUSIBLE_YEAR,
    MIN_PLAUSIBLE_YEAR,
    classify_status,
    normalize_text,
    parse_expiry_date,
)

REFERENCE = date(2026, 8, 14)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("EXP 12/09/2026", date(2026, 9, 12)),          # dayfirst
        ("USE BY 2026-08-10", date(2026, 8, 10)),        # ISO
        ("BB 20260818", date(2026, 8, 18)),              # compact
        ("BEST BEFORE: 3O NOV 2O26", date(2026, 11, 30)),  # O/0 glyph repair
        ("EXP. Sep 05 2026", date(2026, 9, 5)),          # alpha month first
        ("Best before 11/26", date(2026, 11, 1)),        # month/year only
    ],
)
def test_parses_common_retail_formats(raw, expected):
    parsed, pattern, _ = parse_expiry_date(raw, dayfirst=True)
    assert parsed == expected
    assert pattern is not None


def test_month_token_survives_glyph_repair():
    assert "NOV" in normalize_text("BEST BEFORE: 3O NOV 2O26")


def test_unreadable_text_returns_none():
    parsed, pattern, _ = parse_expiry_date("smudged")
    assert parsed is None and pattern is None


@pytest.mark.parametrize(
    "raw",
    [
        "BEST BEFORE 19 Aug 202",   # year truncated by a crop or a blur
        "EXP 12/09/202",
        "USE BY 0202-08-19",
        "EXP 19 Aug 12026",         # duplicated digit
    ],
)
def test_implausible_years_are_rejected(raw):
    """A truncated year must read as unreadable, never as a confident date.

    Regression: OCR losing the last digit of "2027" produced `date(202, 8, 19)`,
    which is a valid Python date and was reported to the operator as fact.
    """
    parsed, _pattern, _ = parse_expiry_date(raw)
    assert parsed is None or MIN_PLAUSIBLE_YEAR <= parsed.year <= MAX_PLAUSIBLE_YEAR


def test_plausible_years_still_parse():
    assert parse_expiry_date("EXP 12/09/2026")[0] == date(2026, 9, 12)
    assert parse_expiry_date("EXP 12/09/26")[0] == date(2026, 9, 12)


def test_impossible_date_falls_through_to_alternate_order():
    # 25 cannot be a month, so the parser must reinterpret as DD/MM.
    parsed, _, _ = parse_expiry_date("13/25/2026", dayfirst=False)
    assert parsed == date(2026, 12, 25) or parsed is None


@pytest.mark.parametrize(
    ("target", "expected_status"),
    [
        (date(2026, 8, 10), "expired"),
        (date(2026, 8, 14), "near_expiry"),
        (date(2026, 8, 18), "near_expiry"),
        (date(2026, 12, 1), "valid"),
        (None, "unreadable"),
    ],
)
def test_status_classification(target, expected_status):
    status, _ = classify_status(target, REFERENCE, near_threshold_days=7)
    assert status == expected_status
