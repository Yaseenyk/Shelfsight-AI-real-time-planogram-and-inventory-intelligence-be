# `data/test_expiry/` — OCR benchmark crops

Drop packaging photos showing a date panel here and
`python -m evaluation.benchmark ocr` runs the **real** EasyOCR pipeline over
them, reporting CER, WER and date-parsing precision.

## Layout

```
data/test_expiry/
├─ ground_truth.csv        (or ground_truth.json / labels.csv)
├─ exp-001.jpg
└─ exp-002.jpg
```

## Ground truth — CSV

```csv
image,truth_text,truth_date
exp-001.jpg,EXP 12/09/2026,2026-09-12
exp-002.jpg,BEST BEFORE: 30 NOV 2026,2026-11-30
exp-006.jpg,,
```

Column names are matched loosely (`image`/`filename`/`file`/`id`,
`truth_text`/`text`/`label`, `truth_date`/`date`/`expiry`).

## Ground truth — JSON

```json
{
  "reference_date": "2026-08-14",
  "samples": [
    { "image": "exp-001.jpg", "truth_text": "EXP 12/09/2026", "truth_date": "2026-09-12" }
  ]
}
```

`reference_date` pins "today" so `valid` / `near_expiry` / `expired` verdicts stay
reproducible as the calendar moves — without it, a benchmark re-run next month
silently produces different statuses.

**Leave `truth_date` empty for genuinely unreadable stamps.** Those rows stay in
the denominator; dropping them would flatter the read rate by discarding exactly
the hard samples the paper should report on.

## Running

```bash
python -m evaluation.benchmark ocr
python -m evaluation.benchmark ocr --ocr-dir path/to/crops --limit 100
```

Notes:

- **No ground-truth file?** The run still reports the date read-rate and real
  latency, and says accuracy was omitted rather than inventing it.
- The report records which preprocessing variant won each image
  (`variant_usage`) — useful evidence for the paper that the multi-variant
  pipeline earns its cost on dot-matrix stamps.
- Empty directory → the suite replays `data/ground_truth/expiry_ground_truth.json`.

Images are git-ignored.
