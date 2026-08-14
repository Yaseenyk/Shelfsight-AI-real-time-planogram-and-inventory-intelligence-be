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

## 5. Expiry OCR pipeline

`app/utils/dates.py` → `app/services/ocr_expiry.py`.

1. **Prefix strip** — `EXP`, `EXP.`, `BB`, `BBE`, `BBD`, `USE BY`, `BEST BEFORE`,
   `CONSUME BY`, `VALID UNTIL`, `UBD`.
2. **Glyph repair** — `O→0`, `I/l/|→1`, `S→5`, `B→8`. Alphabetic month tokens
   (`JAN…DEC`) are masked *before* this pass, so `NOV` never becomes `N0V`.
3. **Pattern match**, first hit wins: `iso_ymd`, `compact_ymd`, `dmy_alpha`, `mdy_alpha`,
   `my_alpha`, `numeric_dmy` (ambiguity resolved by `EXPIRY_DAYFIRST`, with the
   opposite order as fallback when the first read is an impossible date), `numeric_my`.
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

## 7. Local LLM insight layer

`POST /api/v1/insights/generate` assembles a compact telemetry snapshot
(`GET /insights/context` returns the identical payload for auditability) and calls
Ollama `POST /api/generate` with `stream=false`, `format=json` and a schema-constrained
system prompt. Responses are parsed defensively (fenced/prose-wrapped JSON tolerated).

When Ollama is unreachable the service returns a deterministic rule-based briefing with
`degraded: true` rather than an error — the dashboard never shows an empty panel during
a demo, and the flag keeps generated text distinguishable from templated text in any
figure or transcript included in the paper.

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
| 2 | Dataset collection & annotation; YOLOv8 fine-tuning on real SKU classes; published detection numbers | next |
| 3 | Freshness CNN training; OCR tuning on real packaging; per-format error analysis | |
| 4 | Live camera streaming (WebSocket), alert persistence, Ollama prompt tuning | |
| 5 | Full benchmark run, ablations (IoU / centre-distance thresholds), paper figures | |

### What Phase 1 does *not* yet give the paper

The detector is a **stock COCO-pretrained YOLOv8n**. It detects `bottle`,
`banana`, `person`… — not retail SKUs. `data/class_map.json` bridges the two so
the full pipeline runs end-to-end today, but every detection metric produced
before Phase 2 fine-tuning is an **integration check, not a result**. In
particular, scoring the detector against labels derived from its own predictions
yields mAP = 1.0 by construction; only human-annotated frames in
`data/test_images/labels/` produce a citable number.
