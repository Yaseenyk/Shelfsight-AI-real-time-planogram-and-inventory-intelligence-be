"""Real-dataset acquisition and sanitisation.

Fetches public datasets for the detection and freshness pipelines and merges them
into the strict layouts the rest of the system expects.

    python tools/dataset_curator.py list
    python tools/dataset_curator.py fetch freshness --dry-run
    python tools/dataset_curator.py fetch freshness --api-key $ROBOFLOW_API_KEY
    python tools/dataset_curator.py merge --sources data/raw/a data/raw/b \
        --out data/freshness --val-split 0.2

Sources are **declarative** (`DATASET_REGISTRY`), not discovered at runtime. That
is a deliberate limitation worth understanding:

> The `roboflow` package has no dataset *search* API. Downloading requires the
> exact `workspace/project/version` triple plus an API key; there is no
> supported way to turn the string "grocery shelf" into a download. Universe
> search is a web UI. Pretending otherwise would produce a script that fails at
> the client's viva with an opaque error.

So the registry pins known identifiers, `list` prints them with their Universe
URLs for verification, and `--dry-run` shows exactly what would be fetched
without needing a key. Add your own entries — or point `merge` at any folders you
downloaded by hand, which needs no API key at all.

Every run writes `curation_manifest.json`: sources, licences, per-class counts,
duplicate images removed, and the split. That file is what makes the dataset
section of the paper reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from models.dataset import CANONICAL_CLASSES, map_folder_to_label

logger = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_RAW_DIR = settings.DATA_DIR / "raw"
DEFAULT_FRESHNESS_OUT = settings.DATA_DIR / "freshness"
DEFAULT_DETECTION_OUT = settings.DATA_DIR / "detection"


@dataclass
class DatasetSource:
    """One fetchable dataset, pinned to an exact version."""

    key: str
    pipeline: str  # "freshness" | "detection"
    provider: str  # "roboflow" | "kaggle" | "manual"
    description: str
    licence: str
    url: str
    # Roboflow coordinates
    workspace: Optional[str] = None
    project: Optional[str] = None
    version: Optional[int] = None
    fmt: str = "folder"  # "folder" for classification, "yolov8" for detection
    # Kaggle coordinates
    kaggle_ref: Optional[str] = None
    #: Which canonical classes this source is expected to contribute.
    provides: Tuple[str, ...] = ()

    @property
    def target_dir(self) -> Path:
        return DEFAULT_RAW_DIR / self.key


#: Pinned public datasets. **Verify each before citing it**: Universe projects can
#: be renamed, re-versioned or taken down, and this registry cannot detect that
#: without network access. `list` prints the URL for exactly that purpose.
DATASET_REGISTRY: Dict[str, DatasetSource] = {
    "fresh-rotten-kaggle": DatasetSource(
        key="fresh-rotten-kaggle",
        pipeline="freshness",
        provider="kaggle",
        description="Fruits fresh/rotten classification (apples, bananas, oranges). "
        "Binary — supplies fresh + spoiled, no ripening class.",
        licence="CC BY-SA 4.0 (verify on the dataset page)",
        url="https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification",
        kaggle_ref="sriramr/fruits-fresh-and-rotten-for-classification",
        provides=("fresh", "spoiled"),
    ),
    "ripeness-roboflow": DatasetSource(
        key="ripeness-roboflow",
        pipeline="freshness",
        provider="roboflow",
        description="Fruit ripeness stages (unripe/ripe/overripe) — the source of "
        "the `ripening` class the binary sets omit.",
        licence="see project page (Universe projects vary: CC BY 4.0 / MIT / other)",
        url="https://universe.roboflow.com/search?q=fruit+ripeness",
        workspace="ripeness-detection",
        project="fruit-ripeness-stages",
        version=1,
        fmt="folder",
        provides=("fresh", "ripening", "spoiled"),
    ),
    "grocery-shelf-roboflow": DatasetSource(
        key="grocery-shelf-roboflow",
        pipeline="detection",
        provider="roboflow",
        description="Retail shelf product detection, YOLOv8 format.",
        licence="see project page",
        url="https://universe.roboflow.com/search?q=grocery+shelf",
        workspace="retail-shelf",
        project="grocery-shelf-detection",
        version=1,
        fmt="yolov8",
    ),
    "sku110k-manual": DatasetSource(
        key="sku110k-manual",
        pipeline="detection",
        provider="manual",
        description="SKU-110K dense retail shelves (11k images). Large; download "
        "manually, then point `merge`/`--images-dir` at it.",
        licence="research use — see the paper's terms",
        url="https://github.com/eg4000/SKU110K_CVPR19",
    ),
}


# ----------------------------------------------------------------- fetching --
def fetch_roboflow(source: DatasetSource, api_key: str, overwrite: bool = False) -> Path:
    """Download a pinned Roboflow version into `data/raw/<key>/`."""
    try:
        from roboflow import Roboflow  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise RuntimeError(
            "The roboflow package is not installed — pip install roboflow"
        ) from exc

    target = source.target_dir
    if target.exists() and any(target.iterdir()) and not overwrite:
        logger.info("%s already present at %s — skipping (use --overwrite)", source.key, target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading %s/%s v%s (%s) from Roboflow…",
        source.workspace,
        source.project,
        source.version,
        source.fmt,
    )
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(source.workspace).project(source.project)
    dataset = project.version(source.version).download(source.fmt, location=str(target))
    return Path(getattr(dataset, "location", target))


def fetch_kaggle(source: DatasetSource, overwrite: bool = False) -> Path:
    """Download a Kaggle dataset via kagglehub (uses ~/.kaggle/kaggle.json)."""
    try:
        import kagglehub  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed — pip install kagglehub, then place your "
            "credentials in ~/.kaggle/kaggle.json"
        ) from exc

    target = source.target_dir
    if target.exists() and any(target.iterdir()) and not overwrite:
        logger.info("%s already present at %s — skipping (use --overwrite)", source.key, target)
        return target

    logger.info("Downloading %s from Kaggle…", source.kaggle_ref)
    downloaded = Path(kagglehub.dataset_download(source.kaggle_ref))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(downloaded, target)
    return target


def fetch(
    source: DatasetSource,
    api_key: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Optional[Path]:
    """Fetch one source, or explain precisely why it cannot be fetched."""
    if dry_run:
        logger.info(
            "[dry-run] %s (%s) -> %s\n            %s",
            source.key,
            source.provider,
            source.target_dir,
            source.url,
        )
        return None

    if source.provider == "manual":
        logger.warning(
            "%s must be downloaded by hand (%s), then unpacked into %s",
            source.key,
            source.url,
            source.target_dir,
        )
        return None

    if source.provider == "roboflow":
        if not api_key:
            logger.error(
                "%s needs a Roboflow API key. Pass --api-key or set ROBOFLOW_API_KEY "
                "(free account: https://app.roboflow.com/settings/api)",
                source.key,
            )
            return None
        return fetch_roboflow(source, api_key, overwrite)

    if source.provider == "kaggle":
        return fetch_kaggle(source, overwrite)

    logger.error("Unknown provider %r for %s", source.provider, source.key)
    return None


# ------------------------------------------------------------ sanitisation --
@dataclass
class MergeStats:
    """What the merge actually did — persisted into the manifest."""

    sources: List[str] = field(default_factory=list)
    per_class: Dict[str, int] = field(default_factory=dict)
    per_source: Dict[str, Dict[str, int]] = field(default_factory=dict)
    duplicates_removed: int = 0
    unmapped_folders: List[str] = field(default_factory=list)
    train_count: int = 0
    val_count: int = 0
    missing_classes: List[str] = field(default_factory=list)


def _image_hash(path: Path, chunk_size: int = 65536) -> str:
    """Content hash, used to drop the same photo appearing in several sources."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_labelled_images(
    root: Path, class_map: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, List[Path]], List[str]]:
    """Walk a dataset and group images by canonical class.

    Handles the layouts public sets actually use: `train/freshapples/*.jpg`,
    `Fresh/*.jpg`, and nested `*/images/*.jpg`. Folder names are resolved with
    the same keyword mapping the training loader uses, so curation and training
    can never disagree about what `rottenbanana` means.
    """
    grouped: Dict[str, List[Path]] = {name: [] for name in CANONICAL_CLASSES}
    unmapped: List[str] = []

    if not root.exists():
        return grouped, unmapped

    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        images = [
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not images:
            continue

        label = map_folder_to_label(folder.name, class_map)
        if label is None or label not in grouped:
            unmapped.append(folder.name)
            continue
        grouped[label].extend(images)

    return grouped, unmapped


def merge_datasets(
    sources: Sequence[Path],
    out_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
    class_map: Optional[Dict[str, str]] = None,
    copy_files: bool = True,
) -> MergeStats:
    """Merge several raw datasets into the strict 3-class train/val structure.

        out_dir/train/{fresh,ripening,spoiled}/
        out_dir/val/{fresh,ripening,spoiled}/

    Deduplicates by content hash across sources — public fruit datasets overlap,
    and the same photo landing in both train and val silently inflates accuracy.
    """
    import random  # noqa: PLC0415 - stdlib, kept local to the function

    stats = MergeStats(sources=[str(p) for p in sources])
    seen_hashes: set[str] = set()
    pooled: Dict[str, List[Path]] = {name: [] for name in CANONICAL_CLASSES}

    for source in sources:
        grouped, unmapped = collect_labelled_images(Path(source), class_map)
        stats.unmapped_folders.extend(unmapped)
        source_counts: Dict[str, int] = {}

        for label, images in grouped.items():
            kept = 0
            for image in images:
                try:
                    digest = _image_hash(image)
                except OSError as exc:
                    logger.warning("Unreadable file %s: %s", image, exc)
                    continue
                if digest in seen_hashes:
                    stats.duplicates_removed += 1
                    continue
                seen_hashes.add(digest)
                pooled[label].append(image)
                kept += 1
            if kept:
                source_counts[label] = kept
        stats.per_source[str(source)] = source_counts

    rng = random.Random(seed)
    for split in ("train", "val"):
        for label in CANONICAL_CLASSES:
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    for label, images in pooled.items():
        stats.per_class[label] = len(images)
        if not images:
            stats.missing_classes.append(label)
            continue

        ordered = sorted(images, key=str)
        rng.shuffle(ordered)
        cut = int(len(ordered) * val_split) if len(ordered) > 1 else 0
        for index, image in enumerate(ordered):
            split = "val" if index < cut else "train"
            destination = out_dir / split / label / f"{label}_{index:05d}{image.suffix.lower()}"
            if copy_files:
                shutil.copy2(image, destination)
            else:
                destination.write_bytes(b"")  # placeholder for --dry-run inspection
            if split == "train":
                stats.train_count += 1
            else:
                stats.val_count += 1

    if stats.missing_classes:
        logger.warning(
            "No images for %s. The classifier cannot learn these classes; "
            "synthesise with models/augment_data.py or acquire real data.",
            stats.missing_classes,
        )
    return stats


def write_manifest(out_dir: Path, stats: MergeStats, extra: Optional[Dict] = None) -> Path:
    """Record provenance so the paper's dataset section is reproducible."""
    manifest = {
        "curated_at": datetime.now(timezone.utc).isoformat(),
        "classes": list(CANONICAL_CLASSES),
        **asdict(stats),
    }
    if extra:
        manifest.update(extra)
    path = out_dir / "curation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# --------------------------------------------------------------------- CLI --
def cmd_list(_args: argparse.Namespace) -> int:
    print(f"{'key':<24} {'pipeline':<10} {'provider':<10} provides")
    print("-" * 88)
    for source in DATASET_REGISTRY.values():
        provides = ",".join(source.provides) if source.provides else "-"
        print(f"{source.key:<24} {source.pipeline:<10} {source.provider:<10} {provides}")
        print(f"{'':<24} {source.description}")
        print(f"{'':<24} licence: {source.licence}")
        print(f"{'':<24} {source.url}\n")
    print(
        "Verify each project page before citing it — Universe datasets can be\n"
        "renamed, re-versioned or withdrawn, which this registry cannot detect."
    )
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    selected = [
        source
        for source in DATASET_REGISTRY.values()
        if args.pipeline in ("all", source.pipeline)
        and (not args.only or source.key in args.only)
    ]
    if not selected:
        logger.error("No registry entries match pipeline=%s only=%s", args.pipeline, args.only)
        return 1

    fetched: List[Path] = []
    for source in selected:
        try:
            location = fetch(source, args.api_key, args.overwrite, args.dry_run)
        except RuntimeError as exc:
            logger.error("%s: %s", source.key, exc)
            continue
        if location:
            fetched.append(location)

    if args.dry_run:
        print(json.dumps({"would_fetch": [s.key for s in selected]}, indent=2))
    else:
        print(json.dumps({"fetched": [str(p) for p in fetched]}, indent=2))
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    sources = [Path(p) for p in args.sources] if args.sources else sorted(
        p for p in DEFAULT_RAW_DIR.glob("*") if p.is_dir()
    )
    if not sources:
        logger.error(
            "No sources. Pass --sources, or fetch into %s first.", DEFAULT_RAW_DIR
        )
        return 1

    class_map = None
    if args.class_map:
        class_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
        class_map = class_map.get("mapping", class_map)

    out_dir = Path(args.out or DEFAULT_FRESHNESS_OUT)
    stats = merge_datasets(
        sources,
        out_dir,
        val_split=args.val_split,
        seed=args.seed,
        class_map=class_map,
    )
    write_manifest(out_dir, stats, extra={"val_split": args.val_split, "seed": args.seed})

    print(
        json.dumps(
            {
                "out": str(out_dir),
                "train": stats.train_count,
                "val": stats.val_count,
                "per_class": stats.per_class,
                "duplicates_removed": stats.duplicates_removed,
                "missing_classes": stats.missing_classes,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset_curator", description="Fetch and sanitise real datasets"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="show the pinned dataset registry")
    listing.set_defaults(func=cmd_list)

    fetching = sub.add_parser("fetch", help="download registry datasets")
    fetching.add_argument(
        "pipeline", choices=["freshness", "detection", "all"], nargs="?", default="all"
    )
    fetching.add_argument("--only", nargs="*", help="restrict to these registry keys")
    fetching.add_argument("--api-key", default=None, help="Roboflow API key")
    fetching.add_argument("--overwrite", action="store_true")
    fetching.add_argument(
        "--dry-run", action="store_true", help="print what would be fetched, download nothing"
    )
    fetching.set_defaults(func=cmd_fetch)

    merging = sub.add_parser("merge", help="merge raw datasets into the 3-class layout")
    merging.add_argument("--sources", nargs="*", help="raw dataset dirs (default: data/raw/*)")
    merging.add_argument("--out", help=f"output root (default: {DEFAULT_FRESHNESS_OUT})")
    merging.add_argument("--val-split", type=float, default=0.2)
    merging.add_argument("--seed", type=int, default=42)
    merging.add_argument("--class-map", help="JSON overriding folder→label mapping")
    merging.set_defaults(func=cmd_merge)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.command == "fetch" and not args.api_key:
        import os  # noqa: PLC0415

        args.api_key = os.environ.get("ROBOFLOW_API_KEY")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
