# `data/test_images/` — detection benchmark frames

Drop shelf photographs here and `python -m evaluation.benchmark detection` runs
the **real** YOLOv8 detector over them, reporting mAP and per-frame latency.

## Layout

Either flat:

```
data/test_images/
├─ frame-001.jpg
├─ frame-001.txt        # optional label, same stem
└─ frame-002.jpg
```

or the Ultralytics convention (preferred — it is what Roboflow/CVAT export):

```
data/test_images/
├─ images/frame-001.jpg
└─ labels/frame-001.txt
```

## Label format

Standard YOLO txt — one line per object, geometry normalised to `[0, 1]`:

```
<class_id> <center_x> <center_y> <width> <height>
0 0.164 0.185 0.190 0.205
1 0.415 0.185 0.260 0.205
```

`class_id` must match the detector's class indices (`GET /health` reports how
many classes the loaded checkpoint has; `detector.class_names` maps them).

## Running

```bash
python -m evaluation.benchmark detection                      # this directory
python -m evaluation.benchmark detection --limit 50           # quick pass
python -m evaluation.benchmark detection --images-dir path/to/val --conf 0.25
python -m evaluation.benchmark detection --labels-dir path/to/labels
```

Behaviour worth knowing:

- **No labels?** The run still reports real latency (mean/median/p95/FPS) and
  says so in the report; accuracy metrics are omitted rather than faked.
- **Unreadable frame?** It is skipped, listed under `source.skipped` in the
  report, and the run continues.
- **No images here at all?** The suite falls back to replaying the JSON fixtures
  in `data/ground_truth/`, so `benchmark all` never hard-fails on a fresh clone.

Images are git-ignored — keep datasets out of the repo and cite their source in
the paper instead.
