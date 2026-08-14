"""Package everything a client needs into one archive.

The repository alone is not a working handover. Trained weights are gitignored
-- binaries do not belong in version control -- so a fresh clone starts with no
models at all, and Ultralytics quietly downloads the untrained COCO baseline in
their place. The system then starts, reports itself healthy, and detects people
and cars instead of shelf products. Someone receiving only a git URL would not
discover this until a demo.

This produces a single archive containing the weights, the configuration that
points at them, and the documents that explain what to do, with a manifest
recording exactly what went in and how big each piece is.

    python -m tools.package_handover --out dist/shelfsight-handover.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

#: Files whose absence makes the handover non-functional rather than merely
#: incomplete. Packaging stops rather than shipping a bundle that cannot run.
REQUIRED_DOCS = (
    "READ_ME_FIRST.md",
    "START.bat",
    "STOP.bat",
    "scripts/start_all.ps1",
    ".env.example",
)

OPTIONAL_DOCS = (
    "LICENSING.md",
    "THIRD_PARTY_LICENSES.md",
    "README.md",
    "Makefile",
    "docs/publication_metrics/data_preparation.md",
)

WEIGHT_SUFFIXES = {".pt", ".onnx"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_weights(env_file: Path) -> Dict[str, str]:
    """Read the weight paths the application will actually load.

    Deliberately parsed from the config rather than globbed from disk: the point
    is to ship what `.env` names, so a mismatch between the two is caught here
    instead of at the client's first upload.
    """
    wanted: Dict[str, str] = {}
    if not env_file.is_file():
        return wanted
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() in {"DETECTION_WEIGHTS", "FRESHNESS_WEIGHTS"}:
            wanted[key.strip()] = value.strip()
    return wanted


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dist/shelfsight-handover.zip")
    parser.add_argument(
        "--env",
        default=".env.example",
        help="config naming the weights to ship (default: .env.example)",
    )
    parser.add_argument(
        "--allow-missing-weights",
        action="store_true",
        help="package anyway when a configured weight file is absent",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    missing_docs = [name for name in REQUIRED_DOCS if not (ROOT / name).is_file()]
    if missing_docs:
        print("FATAL: required files are missing:", file=sys.stderr)
        for name in missing_docs:
            print(f"  {name}", file=sys.stderr)
        return 1

    wanted = configured_weights(ROOT / args.env)
    if not wanted:
        print(f"FATAL: no weight paths found in {args.env}", file=sys.stderr)
        return 1

    resolved: Dict[str, Path] = {}
    absent: List[str] = []
    for key, relative in wanted.items():
        # Config uses forward slashes; Path normalises them on either platform.
        path = ROOT / Path(relative)
        if path.is_file():
            resolved[key] = path
        else:
            absent.append(f"{key} -> {relative}")

    if absent and not args.allow_missing_weights:
        print("FATAL: configured weights are missing:", file=sys.stderr)
        for item in absent:
            print(f"  {item}", file=sys.stderr)
        print(
            "\nTrain them first, or pass --allow-missing-weights to package a\n"
            "bundle the client cannot run vision features with.",
            file=sys.stderr,
        )
        return 1

    manifest: Dict[str, object] = {
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "weights": {},
        "documents": [],
        "missing_weights": absent,
    }

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for key, path in resolved.items():
            arc = f"models/weights/{path.name}"
            archive.write(path, arc)
            size_mb = round(path.stat().st_size / 1e6, 2)
            manifest["weights"][key] = {  # type: ignore[index]
                "file": arc,
                "size_mb": size_mb,
                "sha256": sha256(path),
            }
            print(f"  + {arc}  ({size_mb} MB)")

        # Sibling exports of the same checkpoints -- ONNX and TorchScript are
        # what the container actually serves, so shipping only the .pt would
        # leave the runtime falling back to eager PyTorch.
        for extra in sorted((ROOT / "models" / "weights").glob("*")):
            if extra.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            arc = f"models/weights/{extra.name}"
            if arc in {v["file"] for v in manifest["weights"].values()}:  # type: ignore[union-attr]
                continue
            archive.write(extra, arc)
            print(f"  + {arc}  ({round(extra.stat().st_size / 1e6, 2)} MB)")

        for name in REQUIRED_DOCS + OPTIONAL_DOCS:
            path = ROOT / name
            if path.is_file():
                archive.write(path, name)
                manifest["documents"].append(name)  # type: ignore[union-attr]
                print(f"  + {name}")

        archive.writestr("HANDOVER_MANIFEST.json", json.dumps(manifest, indent=2))

    size_mb = round(out.stat().st_size / 1e6, 2)
    print(f"\nwrote {out}  ({size_mb} MB)")
    if absent:
        print("\nWARNING: packaged without these weights:")
        for item in absent:
            print(f"  {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
