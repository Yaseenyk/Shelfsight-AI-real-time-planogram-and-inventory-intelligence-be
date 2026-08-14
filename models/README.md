# `models/` — weights and training scripts

| File | Purpose |
| --- | --- |
| `train_detector.py` | Fine-tunes YOLOv8 on the shelf dataset, deploys `best.pt` to `weights/`. |
| `train_freshness.py` | Transfer-learns MobileNetV2/ResNet50 for Fresh / Ripening / Spoiled. |
| `weights/` | Runtime checkpoints (git-ignored — never commit binaries). |
| `datasets/` | Ultralytics dataset YAMLs (paths only, not the images). |

## Expected checkpoints

| Path | Consumed by |
| --- | --- |
| `weights/yolov8n.pt` | `app/services/detection.py` (override with `DETECTION_WEIGHTS`) |
| `weights/freshness_mobilenetv2.pt` | `app/services/freshness.py` (override with `FRESHNESS_WEIGHTS`) |

The API starts **without** these files: each service logs a warning, reports
`is_ready = False`, and the corresponding endpoints return `503` instead of
crashing. That keeps the dashboard and the non-vision endpoints usable while
models are still training.

## Reproducibility

Both scripts fix `--seed 42` by default and write a JSON report into
`evaluation/reports/`, so training runs cited in the paper can be re-derived.
