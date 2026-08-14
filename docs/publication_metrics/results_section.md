# Results and Performance

*Generated from the metric artefacts by `tools/generate_results_section.py`.*
*Every figure is reproducible; none is transcribed by hand.*

### A. Shelf Product Detection

The detector was fine-tuned from YOLOv8n on a 2000-image subset of the SKU-110K training split (8185 images available, sampled by uniform stride) and evaluated on the **complete official test split of 2920 images**, so the figures below are directly comparable with published SKU-110K baselines.

| Metric | Value |
|---|---|
| mAP@0.5 | 0.8462 |
| mAP@0.5:0.95 | 0.4917 |
| Precision | 0.8551 |
| Recall | 0.7874 |
| F1 | 0.8199 |
| Inference latency (CPU) | 103.1 ms |
| Throughput | 9.7 FPS |

Evaluation used an input resolution of 640 px on CPU. The task is single-class product localisation: the system establishes presence, count and position, and SKU identity is resolved downstream by the class-mapping stage rather than by the detector.

### B. Produce Freshness Classification

MobileNetV2 with a frozen backbone was trained to classify produce as fresh, ripening or spoiled, and evaluated on a held-out split of 4061 images. The corpus comprises 27,077 images resolving to 20,984 independent observations after near-duplicate clustering; partitioning was performed over those components rather than over images (18954/4062/4061).

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.9520 |
| Macro F1 | 0.9440 |
| Weighted F1 | 0.9526 |
| Inference latency (CPU) | 16.4 ms (p95 18.4 ms) |
| Throughput | 60.8 FPS |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| fresh | 0.9517 | 0.9761 | 0.9637 | 1212 |
| ripening | 0.8430 | 0.9900 | 0.9106 | 602 |
| spoiled | 0.9886 | 0.9288 | 0.9578 | 2247 |

Performance is limited by the *ripening* class, whose precision of 0.8430 trails *spoiled* at 0.9886. This is consistent with ripening being an intermediate state whose visual boundary with its neighbours is genuinely gradual rather than sharp.

### C. Partition Integrity

Both corpora initially produced held-out figures that measured memorisation rather than generalisation, and neither failure raised any error — the only symptom was implausibly strong performance.

| Corpus | Fault | Evidence | Effect on the reported metric |
|---|---|---|---|
| Detection | partitioned per video frame across three clips of one shelf | all 60 test frames within 7 frames of a training frame; 37 immediately adjacent | mAP@0.5:0.95 of 0.980, discarded |
| Freshness | exact-hash deduplication then image-level shuffling over a pre-augmented source | 30.2\% of held-out images within 5 dHash bits of a training image; 177 exact perceptual duplicates | top-1 inflated from 0.9520 to 0.9606 |

After repartitioning, same-class near-duplicate overlap between the held-out and training splits is **0.00\%**. The 0.86-point reduction in top-1 accuracy is the contribution of leakage, isolated by retraining with identical architecture, hyper-parameters and random seed.
