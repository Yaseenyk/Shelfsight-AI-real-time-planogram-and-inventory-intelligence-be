# `data/test_freshness/` — freshness benchmark crops

Drop labelled produce crops here and `python -m evaluation.benchmark freshness`
runs the **real** classifier over them, exporting Top-1 accuracy, macro/micro F1
and the confusion matrix (PNG + PDF).

## Layout

Labels come from folder names — the same layouts `models/dataset.py` resolves:

```
data/test_freshness/
├─ fresh/*.jpg
├─ ripening/*.jpg
└─ spoiled/*.jpg
```

Public datasets rarely use those exact names, so folders are keyword-mapped:

| Folder name | Mapped to |
| --- | --- |
| `freshapples`, `Fresh`, `good_quality`, `unripe` | `fresh` |
| `semiripe`, `turning`, `ripe`, `aging` | `ripening` |
| `rottenbanana`, `Rotten`, `decayed`, `stale`, `overripe` | `spoiled` |

Anything unmapped is **ignored with a warning** rather than silently folded into
a class. Override with an explicit table:

```bash
python -m evaluation.benchmark freshness --class-map my_classes.json
```

```json
{ "mapping": { "Category_A": "fresh", "Category_B": "spoiled" } }
```

## Running

```bash
python -m evaluation.benchmark freshness
python -m evaluation.benchmark freshness --freshness-dir path/to/val --limit 200
```

Notes:

- **Most public freshness datasets are binary** (fresh/rotten) and contain no
  `ripening` class. The runner reports the class distribution so an empty row in
  the confusion matrix is visibly a dataset property, not a model failure.
- Unreadable files are skipped, listed under `source.skipped`, and the run
  continues.
- Empty directory → the suite replays the JSON fixtures instead, so
  `benchmark all` never hard-fails on a fresh clone.

Images are git-ignored; cite the dataset source in the paper instead.
