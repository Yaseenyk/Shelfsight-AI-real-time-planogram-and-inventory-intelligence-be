# `data/`

| Path | Purpose |
| --- | --- |
| `schemas/planogram.schema.json` | JSON Schema (2020-12) contract for planogram matrices. Cite this in the paper's methodology section. |
| `planograms/*.json` | Planogram instances. Every file here is loaded and upserted by `python -m app.db.init_db --seed`. |
| `ground_truth/` | Human labels used by `evaluation/benchmark.py` (expiry OCR, compliance slot verdicts). |
| `samples/` | Sample shelf frames and packaging crops (git-ignored — keep large media out of the repo). |
| `uploads/` | Runtime destination for frames posted to the API (git-ignored). |

## Coordinate convention

All geometry is **normalised xyxy**: `x1, y1, x2, y2 ∈ [0, 1]`, origin top-left,
`x2 > x1` and `y2 > y1`. This keeps a layout valid across camera resolutions —
the compliance thresholds (`iou_threshold`, `center_distance_threshold`) are
therefore resolution-independent too.

## Validating a planogram

```bash
python -c "from app.schemas.planogram import PlanogramDocument; import pathlib; \
PlanogramDocument.model_validate_json(pathlib.Path('data/planograms/default_planogram.json').read_text()); \
print('valid')"
```
