"""Indian packaging date conventions.

FSSAI-labelled packs differ from the Western samples the parser was first built
against: `MFG`/`PKD` prefixes, DD/MM/YYYY ordering as the norm rather than an
ambiguity, and month-year-only stamps on FMCG items.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.utils.dates import classify_status, parse_expiry_date

REFERENCE = date(2026, 8, 14)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The common FSSAI layouts.
        ("USE BY 12/09/2026", date(2026, 9, 12)),
        ("BEST BEFORE 30 NOV 2026", date(2026, 11, 30)),
        ("EXP. DT. 05/12/2026", date(2026, 12, 5)),
        ("MFG 12/09/2026", date(2026, 9, 12)),
        ("MFD: 01.03.2026", date(2026, 3, 1)),
        ("PKD 15/07/2026", date(2026, 7, 15)),
        ("PACKED ON 15 JUL 2026", date(2026, 7, 15)),
        ("BEST BEFORE DATE 18/08/2026", date(2026, 8, 18)),
    ],
)
def test_indian_prefixes_are_stripped(raw, expected):  # noqa: ANN001
    parsed, pattern, _ = parse_expiry_date(raw, dayfirst=True)
    assert parsed == expected, f"{raw!r} -> {parsed}"
    assert pattern is not None


def test_day_first_is_the_indian_default():
    """05/12/2026 is 5 December in India, not 12 May.

    The reverse reading would move an expiry seven months and could pass expired
    stock as valid, so this is a food-safety-relevant default, not a formatting
    preference.
    """
    assert parse_expiry_date("EXP 05/12/2026", dayfirst=True)[0] == date(2026, 12, 5)


def test_month_year_only_stamp():
    """Many Indian FMCG packs print only MM/YYYY."""
    parsed, _pattern, _ = parse_expiry_date("BEST BEFORE 11/2026", dayfirst=True)
    assert parsed == date(2026, 11, 1)


def test_dotted_separator_common_on_indian_packs():
    assert parse_expiry_date("USE BY 18.08.2026", dayfirst=True)[0] == date(2026, 8, 18)


def test_status_thresholds_hold_for_indian_dairy():
    """Amul toned milk has a ~3 day shelf life; the near-expiry window must catch it."""
    status, remaining = classify_status(date(2026, 8, 16), REFERENCE, near_threshold_days=7)
    assert status == "near_expiry" and remaining == 2


def test_manufacture_date_in_the_past_reads_as_expired():
    """A stripped MFG date is a real date and will read as expired.

    Recorded deliberately: the parser cannot tell MFG from EXP once the prefix is
    gone, so an operator scanning a manufacture stamp sees `expired`. The pattern
    name and raw text are persisted, which is what makes that distinguishable
    downstream.
    """
    parsed, _pattern, _ = parse_expiry_date("MFG 01/01/2026", dayfirst=True)
    status, _ = classify_status(parsed, REFERENCE)
    assert parsed == date(2026, 1, 1)
    assert status == "expired"
