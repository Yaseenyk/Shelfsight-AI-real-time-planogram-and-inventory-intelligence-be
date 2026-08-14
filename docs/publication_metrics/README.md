# `docs/publication_metrics/`

Figures and the metric report for the paper. **Regenerated, never hand-edited:**

```bash
make evaluate                                   # or:
python models/export_pipeline.py metrics --suites all
```

Each run overwrites this directory from a single benchmark run, so every figure
comes from the same models and the same data. Copying charts in by hand is how a
paper ends up with a confusion matrix from one checkpoint beside a PR curve from
another.

| File | Figure |
| --- | --- |
| `detection_metrics.png/.pdf` | mAP@0.5, mAP@0.5:0.95, precision, recall, F1 |
| `detection_pr_curve.png/.pdf` | Precision-recall per detector class |
| `freshness_metrics.png/.pdf` | Top-1 accuracy, macro/micro/weighted F1 |
| `freshness_confusion_matrix*.png/.pdf` | Raw and row-normalised confusion matrices |
| `freshness_pr_curve.png/.pdf` | One-vs-rest PR curves with per-class AP |
| `ocr_metrics.png/.pdf` | Date precision/recall, 1−CER, 1−WER |
| `compliance_metrics.png/.pdf` | Spatial alignment accuracy, discrepancy FPR, mean IoU |
| `benchmark_report.json` | Every number, plus the environment block |
| `export_manifest.json` | Exported artefacts and the ONNX-vs-PyTorch latency |

Both PNG (300 dpi) and PDF (vector) are written; use the PDF in LaTeX.

## Before citing any of it

Open `benchmark_report.json` and check `results.<suite>.source.mode`:

- `"live"` — real inference over your data.
- `"replay"` — the shipped example fixtures. **Not results.** The run logs
  `using example fixtures — not publication numbers` when this happens.

Also check `results.detection.backend`: a value of `fallback` means torchmetrics
had no COCO backend, so `mAP@0.5:0.95` is absent and `map_50` came from the
pure-Python approximation.

Figures are git-ignored — they are build outputs of a specific run, and
committing them invites citing a stale one.
