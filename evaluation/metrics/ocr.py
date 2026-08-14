"""OCR expiry-engine metrics: CER, WER and date-parsing precision.

Definitions used in the paper:
- **CER** = Levenshtein(chars) / len(reference chars), averaged over samples.
- **WER** = Levenshtein(words) / len(reference words), averaged over samples.
- **Date parsing precision** = correct dates / dates the engine *claimed* to read.
- **Date parsing recall** = correct dates / dates a human could read.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from app.utils.dates import parse_expiry_date


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Classic DP edit distance over any sequence (chars or word lists)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (item_a != item_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = reference or ""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis or "") / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = (reference or "").split()
    if not ref_words:
        return 0.0 if not (hypothesis or "").split() else 1.0
    return levenshtein(ref_words, (hypothesis or "").split()) / len(ref_words)


def evaluate(samples: Sequence[Dict[str, Any]], dayfirst: bool = True) -> Dict[str, Any]:
    """Score a ground-truth file (see `data/ground_truth/expiry_ground_truth.json`).

    Each sample needs `ocr_text` (engine output), `truth_text` (human transcript)
    and `truth_date` (ISO string or null when unreadable by a human).
    """
    cer_values: List[float] = []
    wer_values: List[float] = []
    per_pattern: Dict[str, Dict[str, int]] = {}
    rows: List[Dict[str, Any]] = []

    claimed = correct = readable = 0

    for sample in samples:
        hypothesis = sample.get("ocr_text") or ""
        reference = sample.get("truth_text") or ""
        truth_date = _as_date(sample.get("truth_date"))

        cer = character_error_rate(reference, hypothesis)
        wer = word_error_rate(reference, hypothesis)
        cer_values.append(cer)
        wer_values.append(wer)

        parsed, pattern, normalized = parse_expiry_date(hypothesis, dayfirst=dayfirst)
        if truth_date is not None:
            readable += 1
        if parsed is not None:
            claimed += 1
            bucket = per_pattern.setdefault(pattern or "unknown", {"claimed": 0, "correct": 0})
            bucket["claimed"] += 1
            if parsed == truth_date:
                correct += 1
                bucket["correct"] += 1

        rows.append(
            {
                "id": sample.get("id"),
                "ocr_text": hypothesis,
                "normalized": normalized,
                "pattern": pattern,
                "parsed_date": parsed.isoformat() if parsed else None,
                "truth_date": truth_date.isoformat() if truth_date else None,
                "match": bool(parsed and truth_date and parsed == truth_date),
                "cer": round(cer, 4),
                "wer": round(wer, 4),
            }
        )

    precision = correct / claimed if claimed else 0.0
    recall = correct / readable if readable else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "support": len(samples),
        "cer": round(sum(cer_values) / len(cer_values), 4) if cer_values else 0.0,
        "wer": round(sum(wer_values) / len(wer_values), 4) if wer_values else 0.0,
        "date_parsing_precision": round(precision, 4),
        "date_parsing_recall": round(recall, 4),
        "date_parsing_f1": round(f1, 4),
        "dates_claimed": claimed,
        "dates_correct": correct,
        "human_readable_dates": readable,
        "per_pattern": {
            name: {
                **counts,
                "precision": round(counts["correct"] / counts["claimed"], 4)
                if counts["claimed"]
                else 0.0,
            }
            for name, counts in sorted(per_pattern.items())
        },
        "samples": rows,
    }


def _as_date(value: Any) -> Optional[date]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None
