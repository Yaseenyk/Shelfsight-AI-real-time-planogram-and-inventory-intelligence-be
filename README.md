# ShelfSight AI — Backend

Real-time planogram compliance, phantom-inventory detection, freshness classification,
expiry OCR and local-LLM executive insight. FastAPI + SQLAlchemy + SQLite,
YOLOv8 / PyTorch / EasyOCR, Ollama for narrative summaries.

Frontend repo: [Shelfsight-AI-real-time-planogram-and-inventory-intelligence](https://github.com/Yaseenyk/Shelfsight-AI-real-time-planogram-and-inventory-intelligence)

---

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows;  source .venv/bin/activate on Linux/macOS

pip install -r requirements.txt      # API + evaluation (no torch)
pip install -r requirements-ml.txt   # adds torch, ultralytics, easyocr — needed for inference

copy .env.example .env               # cp on Linux/macOS

python -m app.db.init_db --seed      # create SQLite schema + demo catalogue & planogram
uvicorn app.main:app --reload --port 8000
```

- Swagger: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

The API **starts without model weights**. Each ML service reports `is_ready = False`,
logs a warning, and its endpoints return `503` — the catalogue, planogram, parsing and
insight endpoints stay fully usable while models are still training.

### The vision pipeline (Phase 1)

On first start the detector downloads `yolov8n.pt` (~6 MB) into `models/weights/`
and caches it there. Set `DETECTION_ALLOW_DOWNLOAD=false` for air-gapped machines
or for a reproducible published run.

```bash
curl -F "file=@shelf.jpg" -F "planogram_id=PLN-AISLE3-BAY2" \
     http://localhost:8000/api/v1/planogram/verify
```

returns the per-slot verdicts, the raw detections, and the timing breakdown:

```json
{
  "compliance_score": 0.375, "missing_slots": 4, "extra_detections": 4,
  "detection": { "count": 12, "resolved_skus": 9, "unresolved": 3,
                 "suppressed": 1, "inference_ms": 142.1,
                 "class_counts": {"bottle": 7, "banana": 2} },
  "detections": [ { "class_name": "bottle", "sku": "SKU-WATER-500",
                    "confidence": 0.87, "bbox": {"x1": 0.06, ...} } ]
}
```

> **The stock checkpoint detects COCO classes, not SKUs.** `data/class_map.json`
> maps `bottle → SKU-WATER-500` and friends so the pipeline runs end to end
> before fine-tuning. Detection metrics from this checkpoint are an integration
> check, not a publishable result — see Phase 2.

Unreadable uploads return **422** with the reason, a missing model returns **503**,
and a failed frame is recorded as a `FAILED` scan session rather than vanishing.

### Freshness & expiry (Phase 2)

```bash
# Train the classifier on any Kaggle/Roboflow freshness dataset.
# --dry-run prints the folder→label mapping before spending GPU time on it.
python models/train_freshness.py --data-dir data/freshness --dry-run
python models/train_freshness.py --data-dir data/freshness --epochs 25

curl -F "file=@banana.jpg" http://localhost:8000/api/v1/freshness/classify
curl -F "file=@stamp.jpg" -F "reference_date=2026-08-14" \
     http://localhost:8000/api/v1/expiry/extract
```

The OCR endpoint returns the raw text, the winning candidate and which
preprocessing variant produced it:

```json
{
  "best": { "parsed_date": "2026-09-12", "status": "valid",
            "matched_pattern": "numeric_dmy", "days_remaining": 29 },
  "raw_text": "EXP 12/09/2026",
  "variant_used": "raw",
  "variants_tried": ["raw"]
}
```

> **An unreadable stamp is a 200, not an error.** "No date found" is a real audit
> outcome the dashboard must show; only a broken OCR engine produces a 5xx.
> Verdicts: `expired` (< today), `near_expiry` (≤ 7 days), `valid` (> 7 days).

---

## Project layout

```
be/
├─ app/
│  ├─ main.py                 FastAPI entrypoint: CORS, /health, router mounting
│  ├─ core/                   config (env-driven) + logging
│  ├─ db/                     engine/session, declarative base, init_db --seed
│  ├─ models/                 SQLAlchemy ORM (Product, InventoryLog, PlanogramLayout,
│  │                          ComplianceAudit, FreshnessAudit, ExpiryAudit, ScanSession)
│  ├─ schemas/                Pydantic request/response contracts
│  ├─ api/v1/                 inventory · planogram · freshness · expiry · insights
│  ├─ services/               detection (YoloDetector) · class_map · compliance ·
│  │                          freshness · ocr_expiry · llm_client · planogram_store
│  └─ utils/                  vision (OpenCV ingest/preprocess) ·
│                             geometry (IoU, Euclidean, row sorting) · dates (expiry regex)
├─ models/                    dataset loader + training scripts + weights/ (git-ignored)
├─ evaluation/                benchmark.py + metrics/ + reports/
├─ data/                      planogram schema & instances, ground truth, uploads
└─ tests/                     pytest suite (runs without torch)
```

## API surface (`/api/v1`)

| Method | Path | Purpose |
| --- | --- | --- |
| `GET/POST/PATCH` | `/inventory/products[/{sku}]` | Catalogue CRUD (system stock lives here) |
| `POST` | `/inventory/scan` | Reconcile a supplied detection set |
| `POST` | `/inventory/scan/image` | Upload frame → YOLOv8 → reconcile |
| `GET` | `/inventory/summary` \| `/logs` \| `/alerts` | Dashboard tiles, ledger, live alerts |
| `GET/POST/DELETE` | `/planogram/layouts[/{uid}]` | Versioned planogram documents |
| `POST` | **`/planogram/verify`** | **Image → YOLOv8 → SKU resolution → compliance (Phase 1)** |
| `POST` | `/planogram/compliance` | Compliance from a supplied detection set |
| `POST` | `/planogram/compliance/image` | Deprecated alias for `/verify` |
| `GET` | `/planogram/audits[/latest]` | Compliance history |
| `POST` | **`/freshness/classify`** | **Upload → CNN → Fresh / Ripening / Spoiled (Phase 2)** |
| `POST` | `/freshness/classify/batch` | Classify crops already on disk |
| `GET` | `/freshness/audits` \| `/summary` | Spoilage history and rate |
| `POST` | **`/expiry/extract`** | **Upload → EasyOCR → date → validity status (Phase 2)** |
| `POST` | `/expiry/parse` | Text-only date normalisation (no OCR needed) |
| `GET` | `/expiry/audits` \| `/summary` | Expiry history, read rate |
| `GET` | `/insights/status` \| `/context` | Ollama reachability, the exact LLM input |
| `POST` | `/insights/generate` | Executive briefing (falls back if Ollama is down) |

## Evaluation

```bash
python -m evaluation.benchmark all
```

See [`evaluation/README.md`](evaluation/README.md) for suites, input formats and outputs.

## Tests & lint

```bash
pytest -q
ruff check .
```

The suite covers the IoU/Euclidean geometry, the expiry regex parser (including OCR
glyph repair), the compliance engine's four verdict paths and the metric functions —
all without torch installed.

## Notes

- **Coordinates are normalised** (`0..1`, xyxy) everywhere, so planograms and thresholds
  are resolution-independent.
- **SQLite runs in WAL mode** with foreign keys on, so dashboard reads don't block behind
  a long inference request.
- `DETECTION_*`, `COMPLIANCE_*`, `FRESHNESS_*`, `OCR_*` and `OLLAMA_*` are all env-driven;
  commit the `.env` used for a published run so reported numbers stay traceable.
