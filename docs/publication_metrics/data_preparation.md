# Data Preparation and Partition Integrity

Draft material for the IEEE submission. Every figure here is measured and
reproducible from the tooling in `tools/`; nothing is estimated.

## Motivation

Both datasets initially available to this work produced held-out accuracies
that were, on inspection, artefacts of how the partitions were constructed
rather than measurements of generalisation. Because the failure mode is silent
— it inflates the reported metric without raising any error — we treat
partition integrity as an explicit, verified stage of the pipeline rather than
an assumed property of the source data.

## 1. Detection: frame-level leakage in video-derived data

The initial detection corpus comprised 603 images distributed as 452/91/60
across train/validation/test. Filename inspection revealed that all images were
frames decoded from **three** source videos of a single shelf, and that the
partition had been drawn at the level of the individual frame.

We quantified the consequence by computing, for every test frame, the frame-index
distance to the nearest training frame drawn from the same source video:

| Distance to nearest training frame | Test images |
|---|---|
| 1 frame | 37 |
| 2–3 frames | 14 |
| 4–7 frames | 9 |
| ≥ 8 frames | 0 |

All 60 test frames lie within seven frames — under a quarter-second of video —
of an image used in training, and 37 are immediately adjacent. The test
partition is therefore not an independent sample: at 25–30 fps, adjacent frames
are near-identical observations of the same physical scene.

A YOLOv8n detector fine-tuned on this partition attained mAP@0.5 = 0.9905 and
mAP@0.5:0.95 = 0.980 on the corresponding test split. The latter figure is the
diagnostic one: near-perfect performance under strict localisation thresholds is
not consistent with published results on retail shelf detection, and is better
explained by memorisation of near-duplicate frames. **These figures are reported
here only as evidence of the failure mode and are excluded from our results.**

### Remedy

The corpus was replaced with a subset of **SKU-110K** [Goldman et al., CVPR
2019], selected because (i) its images are independent photographs of distinct
shelves and stores rather than frames sampled from video, so adjacent-frame
leakage cannot arise by construction, and (ii) it carries a published
train/validation/test partition against which other work is measured.

To remain trainable on commodity CPU hardware we sample 2,000 images from the
official training split by uniform stride — not by prefix, which would
over-represent whichever stores appear earliest in filename order — and 400
images from the official validation split. **The official test split is retained
in full (2,920 images)** so that reported detection performance is directly
comparable with published SKU-110K baselines. Split boundaries are encoded in
the source filenames, making cross-boundary contamination structurally
impossible.

## 2. Freshness: augmentation-sibling leakage under exact-hash deduplication

The freshness corpus pools 27,077 images across three classes. The curation
stage removed 13,599 byte-identical files by exact content hash and then drew a
70/15/15 partition by shuffling at the level of the individual image.

Exact-hash deduplication is insufficient here. The dominant source is
distributed in pre-augmented form: rotations, flips and re-encodings of a
smaller set of base photographs are stored as separate files. These siblings
differ byte-for-byte, survive exact-hash deduplication, and are then dispersed
across partitions by the shuffle.

We measured the resulting contamination using a 64-bit difference hash (dHash),
which encodes horizontal luminance gradients and is therefore stable under
re-encoding and mild rescaling while remaining sensitive to genuinely different
content. For each held-out image we computed the minimum Hamming distance to any
training image:

| Distance to nearest training image | Test (4,060) | Validation (4,060) |
|---|---|---|
| 0 bits (exact perceptual duplicate) | 177 (4.4%) | 154 (3.8%) |
| ≤ 5 bits (near-identical) | **1,225 (30.2%)** | 1,158 (28.5%) |
| ≤ 10 bits | 3,076 (75.8%) | 3,074 (75.7%) |
| median distance | 8 bits | 8 bits |

Approximately 30% of each held-out partition consisted of near-duplicates of
training images. The 177 exact perceptual duplicates are particularly
diagnostic: these files passed byte-level deduplication yet are visually
identical, confirming that re-encoded variants were present.

### Remedy: connected-component partitioning

We replace image-level assignment with **cluster-level** assignment. Images
within a Hamming radius of 5 bits are linked; the connected components of the
resulting graph are treated as the atomic unit of partitioning, so a base
photograph and all of its augmented descendants are necessarily assigned to the
same split. Components are allocated largest-first to whichever split is
furthest below its target proportion, which keeps realised ratios close to the
70/15/15 target despite highly variable component sizes. Clustering is performed
per class and comparison is restricted to unique hash values, which keeps the
pairwise step tractable.

Clustering reveals that the corpus contains substantially fewer independent
observations than images:

| Class | Images | Components | Absorbed as near-duplicates | Largest component |
|---|---|---|---|---|
| fresh | 8,083 | 6,061 | 2,022 (25.0%) | 96 |
| ripening | 4,015 | 3,213 | 802 (20.0%) | 35 |
| spoiled | 14,979 | 11,710 | 3,269 (21.8%) | 98 |
| **Total** | **27,077** | **20,984** | **6,093 (22.5%)** | — |

Only 20,984 of 27,077 images (77.5%) are distinct observations, and a single
photograph appears in as many as 98 augmented variants. The resulting partition
is 18,954/4,062/4,061.

### Verification

Partition integrity is verified rather than assumed, and training is gated on
that verification. Re-running the leakage measurement against the new partition
yields **0.00% same-class near-duplicate overlap** for both the validation and
test splits, reduced from 30.2%.

### Measured effect on the reported metric

Retraining the identical architecture, hyper-parameters and seed on the
repartitioned corpus isolates the contribution of leakage to the previously
reported figure:

| | Random image-level split | Cluster split |
|---|---|---|
| Same-class near-duplicates in test | 30.2% | **0.00%** |
| Held-out top-1 accuracy | 0.9606 | **0.9520** |
| Held-out macro F1 | 0.9543 | **0.9440** |

The 0.86-point drop in top-1 is not a regression — it is the portion of the
original figure attributable to the model recognising training images it had
already seen. Only the second column is a generalisation estimate.

The per-class breakdown is more informative than the headline. Under the clean
partition, `ripening` precision falls to 0.843 against 0.952 and 0.989 for
`fresh` and `spoiled`, exposing it as the limiting class. The contaminated
partition masked this: augmented siblings of `ripening` training images appeared
in its test set, and the class appeared to perform comparably to the others.
Partition integrity therefore affects not only the magnitude of the reported
metric but which conclusions can be drawn from it.

### Cross-class matches are not leakage

Sixty cross-class matches within the 5-bit radius remain and are *not* leakage.
dHash operates on 9×8 luminance gradients and is invariant to colour, so a fresh
and a spoiled specimen photographed in the same pose and framing yield nearly
identical hashes while carrying different labels. Such pairs cannot inflate
accuracy — they constitute a harder discrimination problem, not an easier one.
Restricting the leakage criterion to same-class matches is therefore the correct
measurement; the pooled figure over-reports contamination.

## Reproducibility

| Stage | Tool |
|---|---|
| Cluster partitioning | `tools/cluster_split.py` |
| SKU-110K subsetting | `tools/subset_sku110k.py` |
| Holdout evaluation | `tools/evaluate_holdout.py` |
| Figures and tables | `tools/generate_publication_figures.py` |

Partition manifests recording hash function, radius, component counts and
per-class allocations are emitted alongside the data
(`cluster_split_manifest.json`, `subset_manifest.json`).

## Limitation

Perceptual hashing detects near-duplicates, not all forms of dependence. Two
distinct photographs of the *same physical specimen* under different lighting or
viewpoint may exceed the 5-bit radius and be assigned to different splits. The
reported figures should therefore be read as an upper bound on partition
integrity under this criterion. The radius is exposed as a parameter
(`--threshold`) so that sensitivity to this choice can be examined.
