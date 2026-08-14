"""Expiry-date normalisation.

OCR on packaging is noisy: glyph confusions (O/0, I/1, S/5), missing separators,
`EXP`/`BB`/`USE BY` prefixes and at least six regional date orders. This module
turns a raw OCR string into a `date` plus the pattern that matched — the pattern
name is persisted so the paper can report per-format parsing precision.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import List, Optional, Tuple

MONTH_TOKENS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_PREFIX_RE = re.compile(
    r"\b(exp(?:iry|ires|\.)?|e\.?x\.?p|best\s*before|bb[eé]?|use\s*by|ubd|bbd|"
    r"consume\s*by|valid\s*(?:un)?til)\b[:\s.-]*",
    re.IGNORECASE,
)
_GLYPH_FIXES = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"})
_SEPARATOR_RE = re.compile(r"[.\-\s/]+")


def normalize_text(raw: str) -> str:
    """Strip prefixes, unify separators and repair common glyph confusions.

    Alphabetic month tokens are protected from the glyph pass so `NOV` does not
    become `N0V`.
    """
    text = _PREFIX_RE.sub("", raw or "").strip()
    text = text.replace(",", " ")

    parts: List[str] = []
    for token in re.split(r"(\b[A-Za-z]{3,4}\b)", text):
        if token.lower().strip() in MONTH_TOKENS:
            parts.append(token.upper())
        else:
            parts.append(token.translate(_GLYPH_FIXES))
    text = "".join(parts)
    return re.sub(r"\s{2,}", " ", text).strip(" :.-")


def _y4(year: int) -> int:
    """Expand a 2-digit year using a retail-friendly pivot (00-79 -> 2000s)."""
    if year >= 100:
        return year
    return 2000 + year if year < 80 else 1900 + year


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(_y4(year), month, day)
    except ValueError:
        return None


# (pattern_name, compiled regex, builder) — evaluated in order, first hit wins.
_PATTERNS: List[Tuple[str, re.Pattern, object]] = [
    (
        "iso_ymd",
        re.compile(r"\b(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})\b"),
        lambda m: _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3))),
    ),
    (
        "compact_ymd",
        re.compile(r"\b(\d{4})(\d{2})(\d{2})\b"),
        lambda m: _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3))),
    ),
    (
        "dmy_alpha",
        re.compile(r"\b(\d{1,2})[\s./\-]*([A-Za-z]{3,4})[\s./\-]*(\d{2,4})\b"),
        lambda m: _safe_date(
            int(m.group(3)), MONTH_TOKENS.get(m.group(2).lower().rstrip("."), 0), int(m.group(1))
        ),
    ),
    (
        "mdy_alpha",
        re.compile(r"\b([A-Za-z]{3,4})[\s./\-]*(\d{1,2})[\s./\-]*(\d{2,4})\b"),
        lambda m: _safe_date(
            int(m.group(3)), MONTH_TOKENS.get(m.group(1).lower().rstrip("."), 0), int(m.group(2))
        ),
    ),
    (
        "my_alpha",  # "NOV 2026" -> last day of month is the conservative read
        re.compile(r"\b([A-Za-z]{3,4})[\s./\-]*(\d{4})\b"),
        lambda m: _safe_date(int(m.group(2)), MONTH_TOKENS.get(m.group(1).lower(), 0), 1),
    ),
    (
        "numeric_dmy",  # ambiguity resolved by `dayfirst`
        # Whitespace counts as a separator: OCR routinely loses faint slashes and
        # dots, so "12 09 2026" is the same stamp as "12/09/2026". Requiring a
        # punctuation separator here silently drops a large share of real reads.
        re.compile(r"\b(\d{1,2})[./\-\s](\d{1,2})[./\-\s](\d{2,4})\b"),
        None,  # handled specially below
    ),
    (
        "numeric_my",  # "11/26"
        re.compile(r"\b(\d{1,2})[./\-](\d{2,4})\b"),
        None,
    ),
]


def parse_expiry_date(raw: str, dayfirst: bool = True) -> Tuple[Optional[date], Optional[str], str]:
    """Return `(parsed_date, pattern_name, normalized_text)`."""
    normalized = normalize_text(raw)
    if not normalized:
        return None, None, normalized

    for name, regex, builder in _PATTERNS:
        match = regex.search(normalized)
        if not match:
            continue

        if name == "numeric_dmy":
            a, b, c = (int(match.group(i)) for i in (1, 2, 3))
            first, second = (a, b) if dayfirst else (b, a)
            parsed = _safe_date(c, second, first) or _safe_date(c, first, second)
        elif name == "numeric_my":
            month, year = int(match.group(1)), int(match.group(2))
            parsed = _safe_date(year, month, 1)
        else:
            parsed = builder(match)  # type: ignore[misc]

        if parsed is not None:
            return parsed, name, normalized

    return None, None, normalized


def days_until(target: date, reference: Optional[date] = None) -> int:
    return (target - (reference or datetime.utcnow().date())).days


def classify_status(
    parsed: Optional[date],
    reference: Optional[date] = None,
    near_threshold_days: int = 7,
) -> Tuple[str, Optional[int]]:
    """Map a parsed date to `valid | near_expiry | expired | unreadable`."""
    if parsed is None:
        return "unreadable", None
    remaining = days_until(parsed, reference)
    if remaining < 0:
        return "expired", remaining
    if remaining <= near_threshold_days:
        return "near_expiry", remaining
    return "valid", remaining
