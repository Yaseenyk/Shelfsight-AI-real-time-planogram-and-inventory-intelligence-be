# ShelfSight AI — Phase 0 Architecture

> Real-time planogram & inventory intelligence. This document is the reference for
> the system design, the data model, the algorithms and the evaluation protocol.
> It is written to be quotable in the methodology section of the paper.

## 1. System overview

```
              ┌───────────────────────── Next.js 14 dashboard (fe/) ─────────────────────────┐
              │  Overview tiles · Capture panel · Planogram grid · Freshness · Expiry ·      │
              │  Discrepancy alerts (polled) · AI briefing                                   │
              └───────────────▲──────────────────────────────────────────────────────────────┘
                              │  typed fetch client (lib/api) — REST/JSON, CORS
              ┌───────────────┴──────────────────────────────────────────────────────────────┐
              │                        FastAPI (be/app) — /api/v1                            │
              │  inventory · planogram · freshness · expiry · insights                       │
              ├──────────────────────────────────────────────────────────────────────────────┤
              │ services/                                                                    │
              │  detection.py    YOLOv8  ─┐                                                  │
              │  compliance.py   IoU + Euclidean spatial engine                              │
              │  freshness.py    MobileNetV2 / ResNet50                                      │
              │  ocr_expiry.py   EasyOCR + regex date normaliser                             │
              │  llm_client.py   Ollama /api/generate (rule-based fallback)                  │
              ├──────────────────────────────────────────────────────────────────────────────┤
              │ SQLAlchemy ORM  →  SQLite (WAL)                                              │
              │ Product · ScanSession · InventoryLog · PlanogramLayout ·                     │
              │ ComplianceAudit · FreshnessAudit · ExpiryAudit                               │
              └──────────────────────────────────────────────────────────────────────────────┘
```

**One capture = one `ScanSession`.** Every audit row references it, so any dashboard
number can be traced back to the exact frame, weights and thresholds that produced it.

### Request flow (image path)

1. `POST /api/v1/inventory/scan/image` — frame saved to `data/uploads/`, `ScanSession` opened.
2. `DetectionService.predict()` → normalised `Detection[]` + `detection_latency_ms`.
3. Detections aggregated per SKU → compared against `Product.system_stock` → `InventoryLog` rows.
4. Session closed with `total_latency_ms`; response carries discrepancies ranked by value impact.

Compliance, freshness and expiry follow the same shape on their own routers, so each
capability can be evaluated and published independently.

## 2. Data model

| Table | Role | Key columns |
| --- | --- | --- |
| `products` | Catalogue + system stock (the "system" half of phantom inventory) | `sku`, `system_stock`, `detection_class_name`, `unit_price`, `is_perishable` |
| `scan_sessions` | One capture through the pipeline | `session_uid`, `image_path`, `status`, `detector_version`, `detection_latency_ms`, `detections` (JSON) |
| `inventory_logs` | Detected vs. system per SKU | `detected_count`, `system_count`, `discrepancy`, `discrepancy_type`, `severity` |
| `planogram_layouts` | Versioned layout documents | `planogram_uid`, `version`, `checksum`, `layout_json`, `slot_count` |
| `compliance_audits` | Shelf roll-up + per-slot detail | `compliance_score`, `spatial_alignment_accuracy`, `mean_iou`, `false_positive_rate`, `slot_results` (JSON) |
| `freshness_audits` | Per-crop classification | `label`, `confidence`, `class_probabilities`, `ground_truth_label` |
| `expiry_audits` | OCR read + parsed date | `raw_text`, `matched_pattern`, `parsed_date`, `days_remaining`, `status`, `ground_truth_text/date` |

Design choices worth defending in review:

- **`layout_json` stays denormalised.** A planogram is an *artefact of an experiment*;
  storing the exact document (plus a SHA-256 `checksum`) means a cited result can be
  reproduced even after the layout is edited.
- **`discrepancy` is stored, not computed.** If the classification rule changes, historical
  rows keep the verdict that was actually shown to the operator.
- **`ground_truth_*` columns live beside predictions.** Labelling happens on real captures,
  so the evaluation set is built from production rows rather than a parallel pipeline.

### Discrepancy rule (`InventoryLog.classify`)

| Condition | Type | Severity |
| --- | --- | --- |
| `detected == system` | `match` | info |
| `detected == 0 and system > 0` | **`phantom`** | critical |
| `detected < system` | `undercount` | warning, or critical when the gap ≥ half of system stock |
| `detected > system` | `overcount` | warning |

## 3. Planogram matrix

Contract: [`data/schemas/planogram.schema.json`](../data/schemas/planogram.schema.json)
(JSON Schema 2020-12), enforced at runtime by `app/schemas/planogram.py`.

```
PlanogramDocument
├─ planogram_id, name, version, store_id, aisle, bay
├─ units: "normalized"
├─ tolerances { iou_threshold, center_distance_threshold,
│               row_band_tolerance, min_detection_confidence }
└─ shelves[]            shelf_id, level, y_range [low, high]
   └─ rows[]            row_id, index
      └─ slots[]        slot_id, position, sku, expected_facings,
                        bbox {x1,y1,x2,y2}, orientation, is_mandatory
```

All geometry is **normalised xyxy** (`0..1`, origin top-left). A layout authored on a
1080p rig therefore transfers unchanged to a 4K camera, and the thresholds transfer with it.

## 3a. Vision pipeline (Phase 1)

```
bytes ──► app/utils/vision.py ──► app/services/detection.py ──► app/services/class_map.py ──► compliance
          decode / validate        YOLOv8 + NMS + parsing        detector class → SKU
```

### Ingestion — `app/utils/vision.py`

`decode_image_bytes()` is the single door into the system. It rejects, with a
domain error the API turns into **422**, anything that cannot be scored:

| Guard | Why |
| --- | --- |
| Empty payload | A zero-byte upload must never read as "empty shelf" |
| `imdecode` returns `None` | Corrupt or unsupported encoding |
| `> MAX_IMAGE_PIXELS` (64 MP) | Decompression-bomb guard |
| Either side `< 32 px` | Too small to contain a readable facing |

`read_image_file()` uses `np.fromfile` + `imdecode` rather than `cv2.imread`,
because `imread` returns `None` on non-ASCII paths on Windows — a failure that
appears only on someone else's machine, mid-experiment.

Preprocessing is explicit and inspectable: `letterbox()` reproduces YOLOv8's
aspect-preserving pad and **returns the inverse transform** (`ratio`, `padding`),
`to_rgb()` marks the one place BGR→RGB happens, and `preprocess()` composes
letterbox → RGB → float → CHW with optional ImageNet statistics.

> Note: the API hands Ultralytics the **raw BGR `ndarray`**, not a preprocessed
> tensor. Ultralytics letterboxes and converts internally; feeding it a
> normalised CHW array would apply both transforms twice. `preprocess()` exists
> for the freshness CNN and for offline experiments.

### Detection — `app/services/detection.py`

`YoloDetector` is a process-wide singleton behind a load lock (Uvicorn serves
from a threadpool; two concurrent first-requests must not both load weights).
`ultralytics`/`torch` are imported inside `load()`, so the API, schemas and test
suite import without them.

```python
detections: List[Detection]     = detector.predict(frame)
result:     DetectionResult     = detector.predict_with_metrics(frame)
```

`predict()` returns `Detection`, not bare `BoundingBox`: the compliance engine
only matches a slot against a detection of the *same SKU*, so class identity and
confidence must travel with the geometry. Each `Detection.bbox` is a validated
`BoundingBox` in normalised xyxy.

**Per-box validation.** One malformed row never aborts a frame that produced 40
good detections. Boxes are clipped into `[0, 1]` (YOLO routinely predicts a few
pixels outside the frame); NaN coordinates, out-of-range confidences, and boxes
below `DETECTION_MIN_BOX_AREA` are dropped and *counted* in
`DetectionResult.suppressed`.

**NMS in two layers.** Ultralytics applies NMS during `predict()`
(`iou=`, `agnostic_nms=`, `max_det=`). Our own `non_max_suppression()` adds an
optional class-agnostic pass for the case that matters on a shelf: one physical
bottle detected as both `bottle` and `cup` is one object, and keeping both
inflates the facing count *and* fabricates an `EXTRA` compliance violation. It
deliberately keeps adjacent facings of the same SKU, which barely overlap.

**Failure is explicit.** `DetectorUnavailableError` (missing weights/package) →
**503**; `DetectionError` (inference blew up) → **500**. Neither is silently
converted into "zero detections", and the `ScanSession` is marked `FAILED` with
the reason, so a broken frame is auditable rather than absent.

### Class resolution — `app/services/class_map.py`

A detector emits its training vocabulary (`bottle`, `banana`, … for COCO); the
planogram speaks SKUs. Resolution order:

1. `Product.detection_class_name` — authoritative once the detector is
   fine-tuned on real SKU classes.
2. `data/class_map.json` — an explicit table that makes a **stock COCO
   YOLOv8n usable end-to-end today**, before the fine-tuned model exists.

Unmapped detections keep `sku = None` and are **not discarded**: they still
reach the compliance engine, where they count as `EXTRA`. An unrecognised object
on a shelf is a finding, not a non-event.

### Reconciliation scope

A shelf-scoped scan asks the planogram what *should* be present
(`planogram_store.expected_skus`) and reconciles against that set. Without it, a
scoped scan could only report on SKUs it happened to detect — and an empty shelf
would produce zero rows, making phantom inventory undetectable exactly where the
capability matters. Scope rules:

| Scope | Products considered |
| --- | --- |
| `expected_skus` given (shelf scan) | Those SKUs ∪ detected |
| `shelf_id` only, no planogram | Detected only (nothing else is knowable) |
| No scope (store reconciliation) | Entire catalogue |

## 4. Compliance algorithm

Implemented in `app/services/compliance.py`; geometry primitives in `app/utils/geometry.py`.

1. **Confidence gate** — drop detections below `min_detection_confidence`.
2. **Spatial sorting** — cluster detections into rows by y-centre proximity
   (`row_band_tolerance`, running-centroid), then order left→right within each row.
   This yields a deterministic reading order independent of detector output order.
3. **Shelf assignment** — a detection belongs to the shelf whose `y_range` contains its centre.
4. **SKU-gated greedy matching** — for each slot, only same-SKU detections are candidates.
   Pairs are admissible when `IoU ≥ iou_threshold` **or** `‖centre_slot − centre_det‖₂ ≤
   center_distance_threshold`; the Euclidean rescue recovers correctly-placed products whose
   detected box is tighter than the drawn slot. Pairs are consumed highest-IoU first,
   ties broken by the closer centre.
5. **Verdicts** —
   - `COMPLIANT`: a same-SKU detection matched the slot.
   - `MISPLACED`: no same-SKU match, but another SKU overlaps the slot at `IoU ≥ threshold`.
   - `MISSING`: nothing overlaps the slot.
   - `EXTRA`: a detection consumed by no slot (off-planogram stock).
6. **Roll-up** — see §6 for the metric definitions.

Complexity is `O(S·D)` per shelf (slots × detections); at realistic shelf sizes
(≤ 40 slots, ≤ 100 detections) this is sub-millisecond, so the reported compliance
latency is dominated by detection, not by matching.

## 4a. Freshness classification (Phase 2)

`app/services/freshness.py`, mirroring the detector's lifecycle contract
(singleton + load lock, lazy torch import, explicit failure).

```python
result = get_freshness_service().predict_freshness(bgr_crop)
result.label          # FreshnessLabel.SPOILED
result.probabilities  # {"fresh": 0.02, "ripening": 0.11, "spoiled": 0.87}
```

- **The class list lives in the checkpoint**, not in config. Label order is a
  property of the trained head; reading it from a settings list is how a model
  silently starts reporting "spoiled" for fresh produce after someone reorders
  that list. `models/train_freshness.py` writes `state_dict` + `backbone` +
  `classes` + `input_size` together.
- **Checkpoint loading tries the safe path first.** torch ≥ 2.6 defaults to
  `weights_only=True`; a pickled module needs the unsafe path, which is
  attempted second *and logged*, so an operator knows when arbitrary pickle runs.
- **Dataset class names are coerced, not rejected.** `rotten`, `overripe`,
  `good_quality` map onto the three canonical labels — with compound forms
  (`overripe`, `notfresh`) tested before the bare keywords they contain.
- Batch inference is the default internally: a shelf frame yields one crop per
  detected perishable, and paying per-call overhead N times stalls a scan.

### Datasets — `models/dataset.py`

Public freshness datasets disagree on layout (Kaggle `train/freshapples/`,
Roboflow `train/Fresh/`, flat `fresh/`) and on vocabulary. `FreshnessDataset.discover()`
resolves all three, maps folder names by keyword (overridable with a JSON table),
and **reports what it could not map** instead of silently absorbing it.

It also surfaces the structural problem with this corpus: most public sets are
**binary** (fresh/rotten) and contain no `ripening` class at all. `describe()`
prints a warning so an empty confusion-matrix row is visibly a dataset property,
not a model failure. Without an in-house `ripening` set, the three-class claim
cannot be evidenced — that is a Phase 3 data task, not a modelling one.

## 5. Expiry OCR pipeline

`app/services/ocr_expiry.py` (OCR + variant strategy) → `app/utils/dates.py`
(normalisation + parsing).

### Why a variant sweep, not one preprocessing choice

Expiry codes are the worst text on a package: dot-matrix/inkjet clouds of
disconnected dots, on curved foil, low contrast, often light-on-dark. Any single
preprocessing choice that fixes one case breaks another. The service therefore
runs an ordered list of variants and stops at the first that yields a *parseable
date* above `OCR_EARLY_STOP_CONFIDENCE`:

| Variant | What it fixes |
| --- | --- |
| `raw` | Clean laser/thermal print (upscaled — EasyOCR degrades below ~20px glyphs) |
| `clahe_sharpen` | Faint ink; recovers contrast without blowing highlights |
| `otsu` | Flat, evenly-lit labels |
| `adaptive_close` | **Dot-matrix**: local threshold + 2px morphological close bridges the dot cloud into continuous strokes |
| `otsu_invert` | Light-on-dark stamps, which otherwise read as noise |

Measured on rendered stamps: clean dates resolve on `raw` in ~2.4 s (CPU); a
dot-matrix stamp only resolves after falling through to `clahe_sharpen`. The
report records `variant_usage`, which is the evidence that the sweep earns its
cost.

**A time budget bounds the worst case.** An unreadable crop satisfies no
early-stop rule, so it used to pay the entire sweep (~11.6 s measured) — the
worst latency landing on exactly the frames that produce nothing.
`OCR_TIME_BUDGET_MS` (default 6 s) caps it.

Two candidates are also formed per crop that a naive reader would miss: each OCR
line individually, **and** all lines joined — a date split across
`BEST BEFORE` / `12 09 2026` only parses when concatenated.

### Parsing — `app/utils/dates.py`

1. **Prefix strip** — `EXP`, `EXP.`, `BB`, `BBE`, `BBD`, `USE BY`, `BEST BEFORE`,
   `CONSUME BY`, `VALID UNTIL`, `UBD`.
2. **Glyph repair** — `O→0`, `I/l/|→1`, `S→5`, `B→8`. Alphabetic month tokens
   (`JAN…DEC`) are masked *before* this pass, so `NOV` never becomes `N0V`.
3. **Pattern match**, first hit wins: `iso_ymd`, `compact_ymd`, `dmy_alpha`, `mdy_alpha`,
   `my_alpha`, `numeric_dmy` (ambiguity resolved by `EXPIRY_DAYFIRST`, with the
   opposite order as fallback when the first read is an impossible date), `numeric_my`.
   `numeric_dmy` accepts **whitespace as a separator** (`12 09 2026`): OCR
   routinely loses faint slashes, and requiring punctuation silently drops a
   large share of real reads.
4. **Two-digit year pivot** — `00–79 → 20xx`, `80–99 → 19xx`.
5. **Status** — `expired` (`days_remaining < 0`), `near_expiry`
   (`≤ EXPIRY_NEAR_THRESHOLD_DAYS`), `valid`, or `unreadable` when nothing parses.

The matched pattern name is persisted, which is what makes **per-format parsing precision**
reportable rather than a single aggregate number.

## 6. Evaluation protocol

Harness: `python -m evaluation.benchmark all` → `evaluation/reports/<UTC stamp>/`.
Every report embeds package versions, device and thresholds.

| Pipeline | Metrics | Definition note |
| --- | --- | --- |
| Detection | mAP@0.5, mAP@0.5:0.95, precision, recall, F1, latency mean/median/p95/FPS | torchmetrics `MeanAveragePrecision` over **live** inference on `data/test_images/`; a pure-Python AP@0.5 fallback runs when the COCO backend is absent and records `backend_error` in the report |
| Freshness | Top-1 accuracy, macro/micro/weighted F1, per-class report, confusion matrix (raw + normalised) | scikit-learn, with a dependency-free fallback |
| Expiry OCR | CER, WER, date-parsing precision/recall/F1, per-pattern precision | CER/WER = Levenshtein over chars/words ÷ reference length, averaged per sample |
| Compliance | Spatial alignment accuracy, discrepancy false-positive rate, discrepancy recall, mean IoU | see below |

**Spatial alignment accuracy** = slot verdicts matching the human label ÷ labelled slots.
In the live engine roll-up the same quantity is computed over *filled* slots only, so
availability failures (`MISSING`) do not mask placement quality.

**Discrepancy false-positive rate** = slots flagged non-compliant that a human labelled
compliant ÷ all slots flagged. This is the metric a store manager feels — every false
alarm is a wasted walk to the aisle.

Figures are emitted at 300 dpi PNG **and** vector PDF. Confusion matrices use a
single-hue sequential ramp with in-cell values, so they stay readable in greyscale print
and under colour-vision deficiency.

## 7. Local LLM insight layer (Phase 3)

```
ScanSession / time window
        │  app/services/insight_context.py
        ▼
   InsightContext  ──compile_prompt()──►  PromptBundle(system, user)
        │                                        │
        │  GET /insights/context                 │  GET /insights/prompt
        ▼                                        ▼
   (auditable telemetry)              Ollama POST /api/generate
                                               │  format=json
                                               ▼
                                   LLMInsightPayload (strict Pydantic)
                                               │
                                               ▼  ranked, capped at 3
                                        InsightResponse
```

### Three layers, deliberately separate

`InsightContext` (what we send) → `LLMInsightPayload` (what we accept back) →
`InsightResponse` (what we return). The middle layer is the important one: a 3B
model will happily return prose, wrong types or half the fields, and an
unvalidated dict reaching the dashboard is how a briefing acquires invented
numbers. Validation is strict on types and required fields, lenient on extra keys
— rejecting a good briefing over a stray `"confidence"` field would trade a
correct answer for a fallback.

### Scope

| Scope | Source | Use |
| --- | --- | --- |
| `session` | one `ScanSession` + its audit rows | "explain this scan" |
| `window` | last N hours, optionally one shelf | "how is the store doing" |

### Prompt compilation

`compile_prompt()` is a pure function of `(context, audience)` — regenerable for
the paper without a database or a running model, and exposed verbatim at
`GET /insights/prompt`. Three details that changed the output measurably:

- **A domain glossary.** Without it, llama3.2 read a *phantom* SKU (system 18,
  detected 0) as an "overstock" and recommended restocking for the wrong reason.
  With `phantom / undercount / overcount / misplaced` defined in the system
  prompt, the same telemetry produced "phantom stockout … detected 0, system 18".
- **Empty metric blocks are dropped.** A model shown a wall of zeros writes about
  the zeros.
- **A clean shelf gets different instructions**, explicitly telling the model not
  to manufacture problems and to return an empty action list.

### Degradation is typed, never silent

`generate()` always returns an `InsightResponse` — it never raises and never
returns `None`. Each failure carries a distinct `degraded_reason` because each
needs a different fix:

| Situation | `degraded_reason` | Fix |
| --- | --- | --- |
| Ollama not running | `ollama_unreachable` | `ollama serve` |
| Model not installed | `model_not_found` | `ollama pull <model>` |
| Generation too slow | `timeout` | smaller model / raise `OLLAMA_TIMEOUT_S` |
| Reply not JSON | `invalid_json` | — |
| JSON wrong shape | `schema_validation_failed` | — |

The Phase 0 client collapsed the second case into "unreachable", which sent you
inspecting a server that was running fine. Model **substitution** (configured
model absent, another installed one used) is applied so a fresh machine works,
but flagged via `model_substituted` — silent substitution would poison
reproducibility.

## 7a. Dataset augmentation (Phase 3)

`models/augment_data.py`, OpenCV + numpy (no albumentations dependency; every
transform explicit and seed-reproducible).

**Ripening synthesis.** Public freshness datasets are almost all binary, leaving
the `ripening` class empty. The generator shifts *saturated* pixels through HSV
toward yellow/orange, scales saturation/value, and stipples ripening spots.
Restricting to saturated pixels matters: shifting the whole frame would drag the
background along and the classifier would learn the tray.

**OCR degradation.** Renders known date strings across six formats, then degrades
them in physical order — print (dot-matrix masking) → geometry (rotation,
perspective) → optics (blur) → illumination (vignette) → contrast → sensor noise
→ codec (JPEG). Applying JPEG before noise would produce artefacts no camera
generates. Emits `ground_truth.csv` so `benchmark ocr` scores it immediately, and
`manifest.json` recording every parameter per image.

**Severity tiers are calibrated against measured behaviour, not intuition.** The
first `harsh` profile read **0/6 even with all five preprocessing variants and a
60 s budget** — a rung past the edge of feasibility measures destruction, not
resilience. `harsh` was retuned to sit at the edge; the known-unreadable case
moved to an opt-in `extreme` tier so the ladder still has a floor to cite.

> **Every generated file is `synthetic: true` in its manifest.** Synthetic
> ripening images are derived from fresh ones by a known colour transform, so a
> classifier can learn the transform rather than ripeness, and evaluating on the
> same distribution measures that circularity. Legitimate: pipeline validation,
> augmenting a real set, ablations. Not legitimate: a headline three-class
> accuracy number with no real ripening photographs.

## 8. Reproducibility checklist

- [ ] `.env` for the run is committed alongside the reported numbers.
- [ ] Detector and classifier trained with the default `--seed 42`.
- [ ] Planogram `checksum` recorded with each compliance result.
- [ ] `benchmark_report.json` retained (it carries the environment block).
- [ ] `backend` field checked in the detection/classification results — a `fallback`
      backend means torchmetrics/scikit-learn were missing and the numbers are not final.

## 9. Phase roadmap

| Phase | Scope | State |
| --- | --- | --- |
| **0** | Architecture, schema, contracts, API skeleton, evaluation harness, dashboard shell | done |
| **1** | OpenCV ingestion, YOLOv8 wrapper + NMS, detection→compliance wiring, live detection benchmark | done |
| **2** | Freshness CNN service + dataset loader + training; EasyOCR variant pipeline; upload endpoints; live freshness/OCR benchmark runners | done |
| **3** | Local-LLM insight layer (typed context, prompt compiler, strict validation, typed degradation); dataset augmentation engine (ripening synthesis, graded OCR degradation) | done |
| 4 | **Real data**: annotate shelf frames for SKU classes; photograph real ripening produce and dot-matrix stamps | next |
| 5 | Fine-tune YOLOv8 + freshness CNN on that data; publishable accuracy numbers | |
| 6 | Live camera streaming (WebSocket), alert persistence | |
| 7 | Full benchmark run, ablations (IoU / centre-distance / OCR variants / severity tiers), paper figures | |

### Measured Phase 3 baseline (synthetic stamps, EasyOCR on CPU)

30 generated stamps, 10 per tier, `reference_date=2026-08-14`:

| Severity | CER | WER | Date precision | Date recall |
| --- | --- | --- | --- | --- |
| mild | 0.53 | 0.55 | **1.00** | 0.60 |
| moderate | 0.54 | 0.67 | 0.67 | 0.40 |
| harsh | 0.80 | 1.00 | 0.40 | 0.20 |

Read these as a *resilience curve on synthetic input*, not as OCR accuracy.
Two definitional notes that materially affect the numbers:

- **CER/WER score the full OCR transcript**, not the line the parser selected.
  Scoring the selected line coupled a recogniser metric to the parser — a date
  regex change moved CER — which makes both uninterpretable. Transcript = OCR
  quality; precision/recall = parser quality.
- CER is high partly because EasyOCR emits spurious extra lines on degraded
  images, and the transcript definition counts them. That is the honest reading:
  those lines are real errors that mislead downstream.

### What Phases 1–2 do *not* yet give the paper

Every pipeline is wired, instrumented and tested. **None of the accuracy numbers
are citable yet**, for reasons that are about data, not code:

| Pipeline | Why the current number is not a result |
| --- | --- |
| Detection | Stock COCO-pretrained YOLOv8n detects `bottle`/`banana`/`person`, not SKUs. `data/class_map.json` bridges it so the pipeline runs today. Scoring against labels derived from its own predictions yields mAP = 1.0 by construction. |
| Freshness | Verified against a synthetic colour-separable set (val 0.83 after 3 head-only epochs). That measures the *plumbing*. A real dataset is needed — and most public ones are binary, so `ripening` needs sourcing. |
| OCR | CER/WER = 0.0 on **rendered** text is a property of the renderer, not the engine. Real dot-matrix photographs will be far harsher. |
| Compliance | Runs on hand-authored fixtures; needs labelled shelf frames. |

The harness is built to make that distinction visible rather than convenient:
live runs log `not publication numbers` when they fall back to fixtures, record
the `backend` that computed each metric, and omit accuracy entirely when labels
are absent instead of inventing it.
