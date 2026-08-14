"""Build a CPU-trainable subset of SKU-110K that respects the official splits.

SKU-110K is the canonical dense retail-shelf benchmark: ~11.7k supermarket
photographs, one class ("object"), roughly 140 annotated products per image.
Two properties make it the right replacement for the video-frame dataset that
leaked:

* the images are independent photographs of different shelves and stores, not
  frames sampled from a handful of videos, so adjacent-frame leakage cannot
  occur by construction; and
* the train/val/test partition is the published one, and here it is encoded in
  the filenames themselves (`train_0.jpg`, `val_0.jpg`, `test_0.jpg`), so a
  subset cannot silently cross a boundary.

The full training split is far too large for CPU fine-tuning (8,185 dense
images is on the order of a day), so we sample a subset of the *training* split
only. Sampling is by uniform stride rather than by taking the first N, because
the filenames are ordered and a prefix would over-represent whichever stores
were captured first. Validation is likewise subsampled to keep per-epoch
overhead down; the test split is left whole so the reported number is directly
comparable to published SKU-110K baselines.

Output is Ultralytics list-file format -- a text file of image paths per split
-- so nothing is copied and the 4.7GB source stays untouched. Ultralytics finds
each label by swapping `/images/` for `/labels/` and the suffix for `.txt`,
which the source layout already satisfies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def stride_sample(items: Sequence[Path], count: int) -> List[Path]:
    """Evenly spaced sample preserving order; returns all items if count >= len."""
    total = len(items)
    if count >= total:
        return list(items)
    if count <= 0:
        return []
    # Spread indices across the whole range so the sample spans every region of
    # the ordered filename space rather than clustering at the start.
    step = total / count
    picked = [items[min(int(i * step), total - 1)] for i in range(count)]
    # int() collisions are possible at the tail; de-duplicate while keeping order.
    seen: set = set()
    unique = [p for p in picked if not (p in seen or seen.add(p))]
    index = 0
    while len(unique) < count and index < total:
        candidate = items[index]
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
        index += 1
    return unique


def list_split(root: Path, split: str) -> List[Path]:
    directory = root / "images" / split
    if not directory.is_dir():
        raise SystemExit(f"missing split directory: {directory}")
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def verify_labels(images: Sequence[Path]) -> tuple[int, int]:
    """Count images whose label file is missing or empty.

    An image with no label file is treated by Ultralytics as a background image
    with zero objects. A handful is harmless; a large number would quietly
    destroy the training signal, so the caller reports the tally.
    """
    missing = 0
    empty = 0
    for image in images:
        label = Path(str(image).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))
        label = label.with_suffix(".txt")
        if not label.is_file():
            missing += 1
        elif label.stat().st_size == 0:
            empty += 1
    return missing, empty


def write_list(path: Path, images: Sequence[Path]) -> None:
    """Write one image path per line using NATIVE separators.

    Ultralytics locates each label with `img2label_paths`, which swaps the
    literal substring `f"{os.sep}images{os.sep}"` for the labels equivalent. On
    Windows that is `\\images\\`, so a list written with forward slashes matches
    nothing: every label resolves to a path that does not exist, Ultralytics
    treats all images as unlabelled background, and training silently produces a
    model that has never seen a box. Emitting native separators is what makes
    the swap fire.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(p.resolve()) for p in images) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, help="SKU-110K v1.0 root (has images/ and labels/)"
    )
    parser.add_argument("--out", default="models/datasets/sku110k_subset")
    parser.add_argument("--yaml", default="models/datasets/sku110k.yaml")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--val", type=int, default=400)
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="0 keeps the FULL official test split (recommended for comparability)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out)
    counts = {"train": args.train, "val": args.val, "test": args.test}

    summary = {}
    for split in ("train", "val", "test"):
        every = list_split(root, split)
        wanted = counts[split]
        chosen = every if wanted <= 0 else stride_sample(every, wanted)
        missing, empty = verify_labels(chosen)
        write_list(out / f"{split}.txt", chosen)
        summary[split] = {
            "available": len(every),
            "selected": len(chosen),
            "missing_labels": missing,
            "empty_labels": empty,
        }
        print(
            f"{split:5s}: {len(chosen):5d} / {len(every):5d} selected"
            f"  (missing labels {missing}, empty {empty})"
        )
        if missing > len(chosen) * 0.05:
            raise SystemExit(
                f"FATAL: {missing} of {len(chosen)} {split} images lack labels - "
                "the layout assumption is wrong, refusing to write a broken dataset"
            )

    config = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": str((out / "train.txt").resolve()).replace("\\", "/"),
        "val": str((out / "val.txt").resolve()).replace("\\", "/"),
        "test": str((out / "test.txt").resolve()).replace("\\", "/"),
        "nc": 1,
        "names": ["product"],
    }
    yaml_path = Path(args.yaml)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out / "subset_manifest.json").write_text(
        json.dumps(
            {
                "source": str(root),
                "dataset": "SKU-110K (resized 640)",
                "split_policy": "official train/val/test preserved; subset sampled "
                "by uniform stride within the training split only",
                "splits": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {yaml_path}")
    print(yaml_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
