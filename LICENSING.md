# Licensing and Redistribution

Read this before publishing the code, handing it to a client, or deploying it
anywhere other people can reach.

The short version: **one dependency, Ultralytics YOLOv8, is AGPL-3.0, and its
obligation is triggered by network use rather than by distribution.** Everything
else is permissive. Dataset terms are separate from software terms and must be
checked individually.

> This is an engineering summary written to flag what needs attention, not legal
> advice. Have someone qualified review it before any commercial deployment.

---

## 1. The one that constrains you: Ultralytics YOLOv8

**Licence:** AGPL-3.0
**Used for:** shelf product detection (`ultralytics>=8.2`)

AGPL-3.0 is copyleft, and section 13 extends the obligation to software accessed
over a network. Ordinary GPL obligations are triggered by *distributing* a
binary; AGPL adds that **letting users interact with the software remotely counts
as distribution**.

Concretely, for this project:

| What you do | Obligation |
|---|---|
| Run it locally for research or a viva | None in practice |
| Hand the source to a client who runs it themselves | Supply the AGPL licence text; they inherit the same terms |
| **Deploy it so anyone reaches the dashboard over a network** | **You must offer the complete corresponding source of your entire application under AGPL-3.0** |

That last row is the one that catches people. Because the FastAPI backend calls
Ultralytics in-process, a networked deployment plausibly makes the whole backend
a derivative work. Serving it to even one remote user triggers the obligation.

### Your options

1. **Open-source the project under AGPL-3.0.** Free, immediate, and the natural
   fit for an academic capstone. Recommended.
2. **Buy an Ultralytics Enterprise Licence** if the client needs to keep their
   source closed. Commercial, priced per case.
3. **Replace the detector.** The architecture does not depend on Ultralytics
   specifically — any detector producing normalised `xyxy` boxes works. A
   permissively licensed alternative removes the constraint entirely:

   | Alternative | Licence |
   |---|---|
   | `torchvision` Faster R-CNN / RetinaNet / FCOS | BSD-3-Clause |
   | DETR (Hugging Face `transformers`) | Apache-2.0 |
   | YOLOX | Apache-2.0 |
   | RT-DETR (Baidu original) | Apache-2.0 |

   Cost: retraining, and probably lower CPU throughput than YOLOv8n.

Note this applies to the **library**, independently of the model weights. Using
`yolov8n.pt` pretrained weights carries the same terms.

---

## 2. Everything else in the stack

All permissive. None require you to publish your source.

| Package | Licence | Role |
|---|---|---|
| FastAPI | MIT | API framework |
| Uvicorn | BSD-3-Clause | ASGI server |
| Pydantic / pydantic-settings | MIT | Validation, settings |
| SQLAlchemy | MIT | ORM |
| Alembic | MIT | Migrations |
| httpx | BSD-3-Clause | HTTP client |
| NumPy | BSD-3-Clause | Arrays |
| pandas | BSD-3-Clause | Tabular data |
| scikit-learn | BSD-3-Clause | Metrics |
| Matplotlib | PSF-based (BSD-compatible) | Figures |
| seaborn | BSD-3-Clause | Figures |
| PyTorch / torchvision | BSD-3-Clause | Training and inference |
| torchmetrics | Apache-2.0 | Detection metrics |
| Pillow | MIT-CMU | Image I/O |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | Preprocessing |
| EasyOCR | Apache-2.0 | Expiry-date OCR |
| pycocotools | BSD-2-Clause | COCO mAP |
| Next.js | MIT | Dashboard |
| React | MIT | UI |
| Tailwind CSS | MIT | Styling |
| lucide-react | ISC | Icons |
| Ollama | MIT | Local LLM runtime |

**Attribution still applies.** MIT, BSD and Apache-2.0 all require the copyright
notice and licence text to travel with redistributed code. Shipping a
`THIRD_PARTY_LICENSES` file satisfies this; `pip-licenses` can generate one.

### The LLM weights are licensed separately from Ollama

Ollama itself is MIT, but each model it runs carries its own terms — Llama
models use the Meta Llama Community Licence, which is **not** OSI-approved and
adds conditions (including a monthly-active-user threshold and naming
requirements). Check the licence of whichever model you configure. Mistral and
Qwen variants are typically Apache-2.0 and simpler to redistribute.

---

## 3. Copyleft further down the tree

`pip-licenses` over the installed environment found four more copyleft packages
that no hand-written dependency list would surface, because none are direct
dependencies — they arrive transitively, mostly through EasyOCR. None is as
demanding as AGPL, but they should be recorded rather than discovered later.

| Package | Licence | Arrives via | What it means here |
|---|---|---|---|
| `ultralytics-thop` | AGPL-3.0+ | ultralytics | Same obligation as YOLOv8; disappears with the same swap |
| `text-unidecode` | Artistic **or** GPL-2.0+ | transitive | Dual-licensed, so **choose the Artistic License** and the GPL obligation never applies. Record the choice. |
| `pi_heif` | LGPL-3.0 | pillow-heif → EasyOCR | LGPL copyleft covers the library itself, not code that imports it. Using it unmodified is fine; publish any modifications you make to it. |
| `python-bidi` | LGPL | EasyOCR | Same as above. |
| `tqdm` | MPL-2.0 AND MIT | many | MPL is file-level copyleft — modified MPL files must be published, your own files are unaffected. |
| `certifi` | MPL-2.0 | httpx, requests | As above. Unmodified in practice. |

The practical conclusion is unchanged: **AGPL-3.0 from Ultralytics is the only
licence that constrains how you may license your own application.** The LGPL and
MPL entries constrain only modifications to those libraries, and `text-unidecode`
offers a permissive alternative you simply elect.

Regenerate this inventory after any dependency change:

```bash
python -m piplicenses --format=markdown --with-urls --order=license \
    > THIRD_PARTY_LICENSES.md
```

## 4. Datasets — verify these individually

Dataset licences are **not** software licences and are frequently more
restrictive. Several forbid commercial use outright, which does not prevent
publishing a paper but does prevent a client deploying commercially.

| Dataset | Used for | Status |
|---|---|---|
| **SKU-110K** | detector training and evaluation | Released for research with the CVPR 2019 paper. **Verify the current terms before any commercial use.** Cite Goldman et al., *Precise Detection in Densely Packed Scenes*, CVPR 2019. |
| **Fruits fresh and rotten for classification** (Kaggle, `sriramr`) | freshness training | Check the dataset page. Kaggle licences vary per upload and are often CC BY-NC-SA (non-commercial). |
| **Banana Ripeness Classification** (Kaggle, `shahriar26s`) | freshness `ripening` class | Same — verify on the page. |

**Nobody has yet confirmed these terms by reading the source pages.** That check
requires a human opening each page and is the single outstanding licensing task
in this project.

Two things that follow regardless:

- **Trained weights may inherit dataset terms.** A model trained on a
  non-commercial dataset is widely treated as a derivative of it. If the client
  intends commercial deployment, retraining on permissively licensed or
  self-collected data is the clean path.
- **Cite the datasets in the paper.** Required by their terms and by academic
  convention.

---

## 5. Recommended position for this project

For an academic capstone handed to a student client:

1. **Publish under AGPL-3.0.** It is the only licence compatible with the
   YOLOv8 dependency without buying a commercial licence, it is free, and it is
   entirely normal for published research code.
2. **Ship a `THIRD_PARTY_LICENSES` file** generated with `pip-licenses`.
3. **Have a human verify the three dataset pages** and record what they say.
4. **Tell the client plainly** that commercial deployment requires either an
   Ultralytics Enterprise Licence or a detector swap, *and* dataset terms that
   permit it. Both are solvable, but they are decisions to make deliberately
   rather than discover later.

## 6. If you swap the detector

Only the detection service is affected. `app/services/detection.py` is the sole
Ultralytics call site; the compliance, freshness, OCR and insight stages consume
normalised bounding boxes and do not care what produced them. Removing
`ultralytics` from `requirements-ml.txt` and reimplementing that one module
against a BSD or Apache detector makes the entire project permissively licensed.
