# `evaluation/` — publication metric harness

Every claim in the paper should be reproducible by one command in this folder.

```bash
python -m evaluation.benchmark all          # run all four suites
python -m evaluation.benchmark ocr          # single suite
```

Output lands in `evaluation/reports/<UTC timestamp>/`:

```
benchmark_report.json                 # all suites + environment provenance
detection_metrics.png / .pdf
freshness_confusion_matrix.png / .pdf
freshness_confusion_matrix_normalized.png
freshness_metrics.png
ocr_metrics.png
compliance_metrics.png
```

## Suites and the metrics they report

| Suite | Metrics | Input |
| --- | --- | --- |
| `detection` | mAP@0.5, mAP@0.5:0.95, precision, recall, F1, latency (mean/median/p95/FPS) | `--predictions`, `--targets` |
| `freshness` | Top-1 accuracy, macro/micro/weighted F1, per-class report, confusion matrix | `--labels` |
| `ocr` | CER, WER, date-parsing precision/recall/F1, per-pattern precision | `--ground-truth` (defaults to `data/ground_truth/expiry_ground_truth.json`) |
| `compliance` | Spatial alignment accuracy, discrepancy false-positive rate, discrepancy recall, mean IoU | `--planogram`, `--ground-truth` |

## Input formats

**Detection** — two parallel JSON arrays, one entry per frame, normalised xyxy:

```json
[{"boxes": [[0.07,0.09,0.25,0.28]], "labels": [0], "scores": [0.91], "latency_ms": 24.5}]
[{"boxes": [[0.06,0.08,0.26,0.29]], "labels": [0]}]
```

**Freshness** — `{"samples": [{"id": "...", "truth_label": "fresh", "predicted_label": "ripening"}]}`

**OCR** — see `data/ground_truth/expiry_ground_truth.json`.

**Compliance** — see `data/ground_truth/compliance_ground_truth.json`.

## Notes for reviewers

- `map_50_95` is `null` when torchmetrics is absent — the harness falls back to a
  pure-Python AP@0.5 and records `"backend": "fallback"` in the report. Install
  `requirements-ml.txt` before generating final numbers.
- Figures are written as 300 dpi PNG **and** vector PDF for direct LaTeX inclusion.
- Confusion matrices use a single-hue sequential ramp so they survive greyscale
  printing; every cell carries its value as text.
