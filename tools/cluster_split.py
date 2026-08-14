"""Re-split a classification dataset by perceptual near-duplicate cluster.

Why this exists
---------------
`dataset_curator.merge` deduplicates by exact file hash and then shuffles at the
*image* level. That is sound only when every image is an independent
observation. It is not: the dominant freshness source is a pre-augmented Kaggle
set in which rotations and flips of one base photograph are stored as separate
files. Those siblings differ byte-for-byte, survive exact-hash dedup, and the
shuffle scatters them across train/val/test -- so a third of the held-out set
was a near-duplicate of something the model had already seen, and the reported
accuracy measured memorisation as much as generalisation.

The fix is to make the *cluster*, not the image, the unit of assignment. Images
within `--threshold` Hamming bits of each other are linked; connected components
of that graph are the clusters; whole clusters are assigned to a single split.
A base photograph and all of its augmented children therefore always land
together, and the held-out set is genuinely unseen.

dHash is the right hash for this: it encodes horizontal gradient structure, so
it is stable under re-encoding, mild scaling and JPEG noise, but moves under
genuinely different content. Comparison is done on *unique* hashes -- exact
collisions are collapsed first -- which is what keeps an O(n^2) pairing
tractable at this dataset's size.

Usage
-----
    python -m tools.cluster_split \\
        --src data/freshness --out data/freshness_clean \\
        --val-split 0.15 --test-split 0.15 --threshold 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

#: Splits read from the source tree. The source is already-split output from
#: `dataset_curator merge`; we pool it all back together before re-splitting,
#: because the existing boundaries are exactly what we do not trust.
SOURCE_SPLITS = ("train", "val", "test")
HASH_WORKERS = 16
#: <=5 bits differing out of 64 is the conventional dHash "visually
#: near-identical" threshold. Exposed as a flag so the sensitivity of the split
#: to this choice can be reported in the paper.
DEFAULT_THRESHOLD = 5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ClassPlan:
    """Per-class assignment outcome, kept for the manifest."""

    label: str
    images: int = 0
    clusters: int = 0
    largest_cluster: int = 0
    counts: Dict[str, int] = field(default_factory=dict)


def dhash(path: Path) -> Optional[int]:
    """64-bit difference hash, or None when the file cannot be decoded."""
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            px = np.asarray(small, dtype=np.int16)
    except Exception:  # noqa: BLE001 - a corrupt image must not abort the split
        return None
    bits = px[:, 1:] > px[:, :-1]
    return int(np.packbits(bits.flatten()).view(">u8")[0])


def collect(src: Path, label: str) -> List[Path]:
    """Every image of one class, pooled across the source's existing splits."""
    found: List[Path] = []
    for split in SOURCE_SPLITS:
        directory = src / split / label
        if not directory.is_dir():
            continue
        found.extend(
            p
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    return found


def hash_all(paths: Sequence[Path]) -> Tuple[List[Path], np.ndarray, int]:
    """Hash in parallel; drop unreadable files. I/O-bound, so threads suffice."""
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as pool:
        hashes = list(pool.map(dhash, paths))
    kept = [(p, h) for p, h in zip(paths, hashes) if h is not None]
    unreadable = len(paths) - len(kept)
    return (
        [p for p, _ in kept],
        np.array([h for _, h in kept], dtype=np.uint64),
        unreadable,
    )


class UnionFind:
    """Disjoint-set over cluster representatives, with path compression."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression, iterative
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def cluster(hashes: np.ndarray, threshold: int, chunk: int = 512) -> np.ndarray:
    """Connected components over the "within `threshold` bits" graph.

    Exact-equal hashes are collapsed to unique values first: augmented siblings
    frequently hash identically, and pairing uniques instead of raw images is
    what keeps the quadratic step affordable.
    """
    uniq, inverse = np.unique(hashes, return_inverse=True)
    union = UnionFind(len(uniq))
    for start in range(0, len(uniq), chunk):
        block = uniq[start : start + chunk, None]
        # Compare each block only against uniques at or after its own start, so
        # every pair is visited once rather than twice.
        tail = uniq[None, start:]
        xor = np.bitwise_xor(block, tail).view(np.uint8)
        dist = np.unpackbits(xor.reshape(-1, 8), axis=1).sum(axis=1)
        dist = dist.reshape(block.shape[0], -1)
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            union.union(start + r, start + c)
    roots = np.array([union.find(i) for i in range(len(uniq))], dtype=np.int64)
    return roots[inverse]


def assign(cluster_ids: np.ndarray, val_split: float, test_split: float) -> Dict[int, str]:
    """Map each cluster to a split, largest-first, to hit the target ratios.

    Greedy largest-first matters: one huge cluster placed late would overshoot
    whichever split it landed in. Placing big clusters while the deficits are
    still large keeps the realised ratios close to the targets.
    """
    sizes: Dict[int, int] = defaultdict(int)
    for cid in cluster_ids.tolist():
        sizes[cid] += 1
    total = int(cluster_ids.size)
    targets = {
        "val": total * val_split,
        "test": total * test_split,
        "train": total * (1.0 - val_split - test_split),
    }
    placed = {"train": 0, "val": 0, "test": 0}
    assignment: Dict[int, str] = {}
    # Deterministic: size descending, cluster id ascending as the tiebreak.
    for cid, size in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        split = max(targets, key=lambda s: targets[s] - placed[s])
        assignment[cid] = split
        placed[split] += size
    return assignment


def run(
    src: Path,
    out: Path,
    labels: Sequence[str],
    val_split: float,
    test_split: float,
    threshold: int,
    move: bool = False,
) -> Dict[str, object]:
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing directory: {out}")
    for split in SOURCE_SPLITS:
        for label in labels:
            (out / split / label).mkdir(parents=True, exist_ok=True)

    plans: List[ClassPlan] = []
    for label in labels:
        paths = collect(src, label)
        if not paths:
            raise SystemExit(f"no images found for class {label!r} under {src}")
        print(f"[{label}] {len(paths)} images -> hashing", flush=True)
        paths, hashes, unreadable = hash_all(paths)
        if unreadable:
            print(f"[{label}]   {unreadable} unreadable, excluded", flush=True)

        cluster_ids = cluster(hashes, threshold)
        plan = ClassPlan(label=label, images=len(paths))
        sizes: Dict[int, int] = defaultdict(int)
        for cid in cluster_ids.tolist():
            sizes[cid] += 1
        plan.clusters = len(sizes)
        plan.largest_cluster = max(sizes.values())
        print(
            f"[{label}]   {plan.clusters} clusters "
            f"(largest {plan.largest_cluster}, "
            f"{len(paths) - plan.clusters} images absorbed as duplicates)",
            flush=True,
        )

        assignment = assign(cluster_ids, val_split, test_split)
        counts: Dict[str, int] = defaultdict(int)
        for index, (path, cid) in enumerate(zip(paths, cluster_ids.tolist())):
            split = assignment[cid]
            target = out / split / label / f"{label}_{index:05d}{path.suffix.lower()}"
            if move:
                shutil.move(str(path), target)
            else:
                shutil.copy2(path, target)
            counts[split] += 1
        plan.counts = dict(counts)
        print(f"[{label}]   {dict(counts)}", flush=True)
        plans.append(plan)

    totals: Dict[str, int] = defaultdict(int)
    for plan in plans:
        for split, n in plan.counts.items():
            totals[split] += n

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "method": "dhash-connected-component cluster split",
        "hash": "dhash-64bit",
        "threshold_bits": threshold,
        "val_split": val_split,
        "test_split": test_split,
        "totals": dict(totals),
        "per_class": {
            p.label: {
                "images": p.images,
                "clusters": p.clusters,
                "largest_cluster": p.largest_cluster,
                "counts": p.counts,
            }
            for p in plans
        },
    }
    (out / "cluster_split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/freshness")
    parser.add_argument("--out", default="data/freshness_clean")
    parser.add_argument("--labels", default="fresh,ripening,spoiled")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--move",
        action="store_true",
        help="move instead of copy (saves disk, destroys the source layout)",
    )
    args = parser.parse_args(argv)

    manifest = run(
        src=Path(args.src),
        out=Path(args.out),
        labels=tuple(x.strip() for x in args.labels.split(",") if x.strip()),
        val_split=args.val_split,
        test_split=args.test_split,
        threshold=args.threshold,
        move=args.move,
    )
    print("\n=== TOTALS ===")
    for split, n in sorted(manifest["totals"].items()):  # type: ignore[union-attr]
        print(f"  {split}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
