"""Normalise a downloaded YOLO dataset into a trainable Ultralytics config.

Roboflow-style exports ship a `data.yaml` whose split paths are written as
`../train/images`, assuming the YAML sits one level *below* the dataset root.
When the export is unpacked so that the YAML sits *beside* `train/`, those paths
resolve outside the dataset and Ultralytics fails with an unhelpful
"Dataset images not found" error.

This writes a corrected YAML with absolute paths next to the original, leaving
the download untouched:

    python tools/prepare_detection_yaml.py --dataset <dir> --out models/datasets/shelf.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

SPLIT_KEYS = ("train", "val", "test")

#: Exports disagree on the validation folder's name. Getting this wrong is not a
#: cosmetic failure: the fallback is "validate on train", which silently reports
#: memorisation as accuracy.
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation", "eval"),
    "test": ("test", "testing"),
}


def find_dataset_yaml(root: Path) -> Optional[Path]:
    """Locate the export's own data.yaml, wherever it was unpacked."""
    candidates = sorted(root.rglob("data.yaml")) or sorted(root.rglob("*.yaml"))
    return candidates[0] if candidates else None


def resolve_split_dir(dataset_root: Path, declared: str, split: str) -> Optional[Path]:
    """Find the real directory for a split, ignoring a broken declared path."""
    # Try the declared path first — it is correct in well-formed exports.
    for base in (dataset_root, dataset_root.parent):
        candidate = (base / declared).resolve()
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate

    # Fall back to the conventional layout under any accepted folder name.
    for alias in SPLIT_ALIASES.get(split, (split,)):
        for pattern in (f"{alias}/images", alias):
            candidate = dataset_root / pattern
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate

    for alias in SPLIT_ALIASES.get(split, (split,)):
        matches = [p for p in dataset_root.rglob(f"{alias}/images") if p.is_dir()]
        if matches:
            return matches[0]
    return None


def build_config(dataset_dir: Path, out_path: Path) -> Dict[str, Any]:
    source_yaml = find_dataset_yaml(dataset_dir)
    if source_yaml is None:
        raise FileNotFoundError(f"No data.yaml found under {dataset_dir}")

    original = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = source_yaml.parent
    logger.info("Source config: %s", source_yaml)

    names: List[str] = original.get("names") or []
    if isinstance(names, dict):  # some exports use {0: name, ...}
        names = [names[key] for key in sorted(names)]

    config: Dict[str, Any] = {"names": names, "nc": original.get("nc", len(names))}

    for split in SPLIT_KEYS:
        declared = original.get(split)
        if not declared:
            continue
        resolved = resolve_split_dir(dataset_root, str(declared), split)
        if resolved is None:
            logger.warning("Could not resolve the %r split (declared %r)", split, declared)
            continue
        count = len(
            [p for p in resolved.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        logger.info("%-5s -> %s (%d images)", split, resolved, count)
        # Absolute paths: immune to where Ultralytics is invoked from.
        config[split] = str(resolved).replace("\\", "/")

    if "train" not in config:
        raise ValueError(f"No usable train split found under {dataset_dir}")
    if "val" not in config:
        # Refuse rather than silently validate on train: the resulting mAP would
        # measure memorisation and would be indistinguishable from a real score.
        raise ValueError(
            f"No validation split found under {dataset_dir}. Validating on train "
            "would report memorisation as accuracy, so no config was written. "
            "Point --dataset at an export that includes val/ (or valid/)."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    logger.info("Wrote %s (%d classes)", out_path, config["nc"])
    return config


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Fix a YOLO dataset config for training")
    parser.add_argument("--dataset", required=True, help="downloaded dataset directory")
    parser.add_argument(
        "--out", default="models/datasets/shelf.yaml", help="corrected config to write"
    )
    args = parser.parse_args(argv)

    try:
        config = build_config(Path(args.dataset), Path(args.out))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    print(yaml.safe_dump(config, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
