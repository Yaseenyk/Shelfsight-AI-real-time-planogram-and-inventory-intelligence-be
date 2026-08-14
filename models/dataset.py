"""Freshness dataset loading for real-world folder layouts.

Public fruit/vegetable freshness datasets do not agree on anything. The common
shapes this module handles:

    Kaggle "Fruits fresh and rotten for classification"
        dataset/train/freshapples/*.png
        dataset/train/rottenbanana/*.png
        dataset/test/...

    Roboflow classification export
        dataset/train/Fresh/*.jpg
        dataset/valid/Overripe/*.jpg
        dataset/test/Rotten/*.jpg

    Flat, no split
        dataset/fresh/*.jpg
        dataset/spoiled/*.jpg

Two problems follow from that. First, **folder names are not our labels**:
`freshapples`, `Fresh`, `good_quality` all mean `fresh`; `rottenbanana`,
`Rotten`, `decayed` all mean `spoiled`. Second, **most public sets are binary**
(fresh/rotten) while our taxonomy has three classes — the `ripening` class is
simply absent, and pretending otherwise produces a model that can never predict
it and a confusion matrix with an empty row.

`FreshnessDataset.discover()` resolves the first with keyword mapping (plus an
explicit JSON override) and *reports* the second rather than hiding it.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

CANONICAL_CLASSES: Tuple[str, ...] = ("fresh", "ripening", "spoiled")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_DIR_NAMES = ("train", "training")
VAL_DIR_NAMES = ("val", "valid", "validation", "test", "testing", "eval")

#: Keyword → canonical label. Order matters: `overripe` must beat `ripe`, and
#: `not_fresh` must beat `fresh`, so negations and compounds are listed first.
LABEL_RULES: Tuple[Tuple[str, str], ...] = (
    ("notfresh", "spoiled"),
    ("not_fresh", "spoiled"),
    ("overripe", "spoiled"),
    ("veryripe", "spoiled"),
    ("rotten", "spoiled"),
    ("spoil", "spoiled"),
    ("decay", "spoiled"),
    ("mould", "spoiled"),
    ("mold", "spoiled"),
    ("stale", "spoiled"),
    ("expired", "spoiled"),
    ("bad", "spoiled"),
    ("unripe", "fresh"),  # before "ripe": unripe stock is sellable, not turning
    ("ripening", "ripening"),
    ("semiripe", "ripening"),
    ("semi_ripe", "ripening"),
    ("halfripe", "ripening"),
    ("turning", "ripening"),
    ("aging", "ripening"),
    ("ripe", "ripening"),
    ("mid", "ripening"),
    ("fresh", "fresh"),
    ("good", "fresh"),
    ("healthy", "fresh"),
    ("unripe", "fresh"),
    ("new", "fresh"),
)


class DatasetError(RuntimeError):
    """The dataset directory cannot be turned into a usable training set."""


@dataclass
class ClassFolder:
    """One source folder and the canonical label it was mapped to."""

    path: Path
    folder_name: str
    label: str
    count: int
    split: str


@dataclass
class FreshnessDataset:
    """A resolved dataset: file lists per split, plus the mapping audit trail."""

    root: Path
    classes: List[str]
    train: List[Tuple[Path, int]] = field(default_factory=list)
    val: List[Tuple[Path, int]] = field(default_factory=list)
    folders: List[ClassFolder] = field(default_factory=list)
    unmapped: List[str] = field(default_factory=list)

    @property
    def class_to_index(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.classes)}

    def distribution(self, split: str = "train") -> Dict[str, int]:
        items = self.train if split == "train" else self.val
        counts = Counter(self.classes[index] for _path, index in items)
        return {name: counts.get(name, 0) for name in self.classes}

    def describe(self) -> str:
        lines = [
            f"Dataset: {self.root}",
            f"  classes : {self.classes}",
            f"  train   : {len(self.train)} images {self.distribution('train')}",
            f"  val     : {len(self.val)} images {self.distribution('val')}",
            "  folder mapping:",
        ]
        for folder in self.folders:
            lines.append(
                f"    [{folder.split:<5}] {folder.folder_name:<28} -> "
                f"{folder.label:<9} ({folder.count} images)"
            )
        if self.unmapped:
            lines.append(f"  UNMAPPED (ignored): {sorted(set(self.unmapped))}")
        missing = [name for name, count in self.distribution("train").items() if count == 0]
        if missing:
            lines.append(
                f"  WARNING: no training images for {missing}. The model cannot learn "
                "these classes; the confusion matrix will contain empty rows."
            )
        return "\n".join(lines)

    @classmethod
    def discover(
        cls,
        root: Path,
        class_map_file: Optional[Path] = None,
        val_split: float = 0.2,
        classes: Sequence[str] = CANONICAL_CLASSES,
        seed: int = 42,
    ) -> "FreshnessDataset":
        """Walk `root`, map folder names to labels and build the split lists."""
        root = Path(root)
        if not root.exists():
            raise DatasetError(f"Dataset directory not found: {root}")

        overrides = _load_overrides(class_map_file)
        dataset = cls(root=root, classes=list(classes))
        index = dataset.class_to_index

        split_dirs = _find_split_dirs(root)
        for split, directory in split_dirs:
            for folder in sorted(p for p in directory.iterdir() if p.is_dir()):
                label = map_folder_to_label(folder.name, overrides)
                images = _list_images(folder)
                if label is None:
                    if images:
                        dataset.unmapped.append(folder.name)
                        logger.warning(
                            "Folder '%s' (%d images) did not map to any class — ignored. "
                            "Add it to the class-map JSON to include it.",
                            folder.name,
                            len(images),
                        )
                    continue
                if label not in index:
                    dataset.unmapped.append(folder.name)
                    continue
                if not images:
                    continue

                dataset.folders.append(
                    ClassFolder(
                        path=folder,
                        folder_name=folder.name,
                        label=label,
                        count=len(images),
                        split=split,
                    )
                )
                target = dataset.train if split == "train" else dataset.val
                target.extend((image, index[label]) for image in images)

        if not dataset.train and not dataset.val:
            raise DatasetError(
                f"No labelled images found under {root}. Expected class folders such as "
                "train/freshapples/ or fresh/ — see models/dataset.py for the layouts handled."
            )

        # No validation split in the dataset: carve one out deterministically so
        # the reported accuracy is not measured on training images.
        if not dataset.val and 0.0 < val_split < 1.0:
            dataset.train, dataset.val = _stratified_split(dataset.train, val_split, seed)
            logger.info(
                "No validation folder found — held out %.0f%% of training data (seed=%d)",
                val_split * 100,
                seed,
            )
        return dataset


def map_folder_to_label(
    folder_name: str, overrides: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Map a source folder name onto a canonical class, or None if unknown."""
    raw = str(folder_name).strip()
    key = raw.lower().replace("-", "").replace("_", "").replace(" ", "")

    if overrides:
        for candidate in (raw, raw.lower(), key):
            if candidate in overrides:
                return overrides[candidate]

    if key in CANONICAL_CLASSES:
        return key
    for token, label in LABEL_RULES:
        if token.replace("_", "") in key:
            return label
    return None


def _load_overrides(path: Optional[Path]) -> Dict[str, str]:
    """Optional explicit `{folder_name: canonical_label}` JSON."""
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("Class-map file %s not found — using keyword rules only", file_path)
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Invalid class-map file %s: %s", file_path, exc)
        return {}

    raw = payload.get("mapping", payload) if isinstance(payload, dict) else {}
    overrides = {str(k): str(v).lower() for k, v in raw.items() if isinstance(v, str)}
    # Index by several spellings so callers need not match our normalisation.
    expanded: Dict[str, str] = {}
    for key, value in overrides.items():
        expanded[key] = value
        expanded[key.lower()] = value
        expanded[key.lower().replace("-", "").replace("_", "").replace(" ", "")] = value
    return expanded


def _find_split_dirs(root: Path) -> List[Tuple[str, Path]]:
    """Locate train/val directories, or treat `root` itself as a flat train set."""
    found: List[Tuple[str, Path]] = []
    children = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}

    for name in TRAIN_DIR_NAMES:
        if name in children:
            found.append(("train", children[name]))
            break
    for name in VAL_DIR_NAMES:
        if name in children:
            found.append(("val", children[name]))
            break

    if not found:
        found.append(("train", root))
    return found


def _list_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _stratified_split(
    items: List[Tuple[Path, int]], val_split: float, seed: int
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Per-class split, so a rare class does not vanish from validation."""
    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)

    rng = random.Random(seed)
    train: List[Tuple[Path, int]] = []
    val: List[Tuple[Path, int]] = []
    for _label, group in sorted(by_class.items()):
        ordered = sorted(group, key=lambda pair: str(pair[0]))
        rng.shuffle(ordered)
        cut = max(1, int(len(ordered) * val_split)) if len(ordered) > 1 else 0
        val.extend(ordered[:cut])
        train.extend(ordered[cut:])
    return train, val


def build_torch_dataset(items: Sequence[Tuple[Path, int]], transform: Any) -> Any:
    """Wrap `(path, label)` pairs in a `torch.utils.data.Dataset`.

    Reads through `app.utils.vision.read_image_file`, which is the same ingestion
    path the API uses — so training never sees images the API would reject, and
    a corrupt file is skipped with a warning instead of killing an epoch.
    """
    import torch  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    from app.utils.vision import ImageDecodeError, read_image_file, to_rgb  # noqa: PLC0415

    class _FreshnessTorchDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.items = list(items)
            self.transform = transform
            self.skipped = 0

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):  # noqa: ANN201
            path, label = self.items[index]
            try:
                array = to_rgb(read_image_file(path))
            except ImageDecodeError as exc:
                # One unreadable JPEG must not abort a 20-epoch run.
                self.skipped += 1
                logger.warning("Skipping unreadable image %s: %s", path, exc)
                neighbour = (index + 1) % len(self.items)
                if neighbour == index:
                    raise DatasetError(f"No readable images in dataset: {exc}") from exc
                return self.__getitem__(neighbour)
            return self.transform(PILImage.fromarray(array)), label

    return _FreshnessTorchDataset()
