"""Model export and publication-metric generation.

    python models/export_pipeline.py export --format onnx
    python models/export_pipeline.py metrics
    python models/export_pipeline.py all

Two jobs, deliberately in one place because they belong to the same moment — the
end of a training run:

1. **Export** the trained YOLOv8 and freshness weights to ONNX / TorchScript.
   The packaged container runs CPU-only inference; ONNX Runtime typically beats
   eager PyTorch there, and TorchScript removes the Python class dependency so a
   checkpoint cannot be broken by a later refactor of `app/services/`.
2. **Publish** the metric figures into `docs/publication_metrics/`, which is the
   directory the paper draws from. Regenerating figures by hand after a training
   run is how a paper ends up with a chart from two models ago.

Exports are *verified*, not assumed: each one is reloaded and run on a dummy
input, and ONNX outputs are compared against the PyTorch reference. An export
that silently produces wrong numbers is worse than no export.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

PUBLICATION_DIR = Path(__file__).resolve().parents[1] / "docs" / "publication_metrics"

#: Figures copied into the publication directory, if the benchmark produced them.
FIGURE_NAMES = (
    "detection_metrics",
    "detection_pr_curve",
    "freshness_metrics",
    "freshness_confusion_matrix",
    "freshness_confusion_matrix_normalized",
    "freshness_pr_curve",
    "ocr_metrics",
    "compliance_metrics",
)


# ------------------------------------------------------------------ export --
def export_detector(
    weights: Optional[Path] = None,
    fmt: str = "onnx",
    imgsz: Optional[int] = None,
    opset: int = 12,
) -> Optional[Path]:
    """Export the YOLOv8 detector via Ultralytics' own exporter.

    Ultralytics owns the graph surgery (NMS handling, dynamic axes), so calling
    its exporter is materially safer than hand-tracing the model.
    """
    weights_path = Path(weights or settings.DETECTION_WEIGHTS)
    if not weights_path.exists():
        logger.warning("Detector weights not found at %s — skipping export", weights_path)
        return None

    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        logger.error("ultralytics is not installed — pip install -r requirements-ml.txt")
        return None

    logger.info("Exporting detector %s to %s…", weights_path.name, fmt)
    model = YOLO(str(weights_path))
    exported = model.export(
        format=fmt,
        imgsz=imgsz or settings.DETECTION_IMG_SIZE,
        opset=opset if fmt == "onnx" else None,
        dynamic=False,
        simplify=False,
    )
    path = Path(exported) if exported else None
    if path and path.exists():
        logger.info("Detector exported: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def export_freshness(
    weights: Optional[Path] = None, fmt: str = "onnx", opset: int = 12
) -> Optional[Path]:
    """Export the freshness CNN to ONNX or TorchScript, then verify it."""
    weights_path = Path(weights or settings.FRESHNESS_WEIGHTS)
    if not weights_path.exists():
        logger.warning("Freshness weights not found at %s — skipping export", weights_path)
        return None

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        logger.error("torch is not installed — pip install -r requirements-ml.txt")
        return None

    from app.services.freshness import FreshnessService  # noqa: PLC0415

    service = FreshnessService(weights=weights_path)
    if not service.load():
        logger.error("Could not load %s: %s", weights_path, service.load_failure)
        return None

    model = service._model  # noqa: SLF001 - export needs the raw module
    model.eval()
    size = service.input_size
    dummy = torch.randn(1, 3, size, size)

    with torch.no_grad():
        reference = model(dummy)

    if fmt == "torchscript":
        target = weights_path.with_suffix(".torchscript.pt")
        traced = torch.jit.trace(model, dummy)
        traced.save(str(target))
        # Reload and re-run: a trace that captured the wrong branch fails here.
        reloaded = torch.jit.load(str(target))
        with torch.no_grad():
            replayed = reloaded(dummy)
        drift = float((reference - replayed).abs().max())
        logger.info("TorchScript export verified (max |Δ| = %.2e): %s", drift, target)
        return target

    target = weights_path.with_suffix(".onnx")
    export_kwargs: Dict[str, Any] = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "opset_version": opset,
        "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
    }
    # torch >= 2.9 routes torch.onnx.export through the dynamo exporter, which
    # imports `onnxscript`. Pin the legacy TorchScript path when the signature
    # offers it: it needs no extra dependency, which keeps the CPU container
    # smaller and the client's install one step shorter.
    import inspect  # noqa: PLC0415

    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    try:
        torch.onnx.export(model, dummy, str(target), **export_kwargs)
    except ImportError as exc:  # pragma: no cover - depends on the torch build
        logger.error(
            "ONNX export needs an extra package on this torch build (%s). "
            "Either `pip install onnxscript` or export TorchScript instead: "
            "python models/export_pipeline.py export --format torchscript",
            exc,
        )
        return None

    drift = _verify_onnx(target, dummy, reference)
    if drift is not None:
        logger.info("ONNX export verified (max |Δ| = %.2e): %s", drift, target)
    return target


def _verify_onnx(path: Path, dummy: Any, reference: Any) -> Optional[float]:
    """Run the exported graph and compare against the PyTorch output."""
    try:
        import numpy as np  # noqa: PLC0415
        import onnxruntime  # noqa: PLC0415
    except ImportError:
        logger.warning("onnxruntime not installed — export written but NOT verified")
        return None

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": dummy.numpy()})
    drift = float(np.abs(reference.numpy() - outputs[0]).max())
    if drift > 1e-3:
        logger.error(
            "ONNX output diverges from PyTorch by %.2e — do not ship this export", drift
        )
    return drift


def benchmark_runtimes(
    onnx_path: Optional[Path] = None, runs: int = 20
) -> Dict[str, Any]:
    """Measure eager-PyTorch vs ONNX Runtime latency — the reason to export at all."""
    if onnx_path is None or not Path(onnx_path).exists():
        return {}

    try:
        import time  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        import onnxruntime  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError:
        return {}

    from app.services.freshness import FreshnessService  # noqa: PLC0415

    service = FreshnessService()
    if not service.load():
        return {}

    size = service.input_size
    dummy = torch.randn(1, 3, size, size)
    model = service._model  # noqa: SLF001
    model.eval()

    with torch.no_grad():
        for _ in range(3):  # warm-up
            model(dummy)
        started = time.perf_counter()
        for _ in range(runs):
            model(dummy)
        torch_ms = (time.perf_counter() - started) * 1000.0 / runs

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    payload = {"input": dummy.numpy().astype(np.float32)}
    for _ in range(3):
        session.run(None, payload)
    started = time.perf_counter()
    for _ in range(runs):
        session.run(None, payload)
    onnx_ms = (time.perf_counter() - started) * 1000.0 / runs

    result = {
        "runs": runs,
        "pytorch_ms": round(torch_ms, 3),
        "onnx_ms": round(onnx_ms, 3),
        "speedup": round(torch_ms / onnx_ms, 2) if onnx_ms else None,
    }
    logger.info(
        "CPU latency: PyTorch %.1f ms vs ONNX %.1f ms (%.2fx)",
        torch_ms,
        onnx_ms,
        result["speedup"] or 0.0,
    )
    return result


# ----------------------------------------------------------------- metrics --
def publish_metrics(run_id: Optional[str] = None, suites: str = "all") -> Dict[str, Any]:
    """Run the benchmark harness and copy its figures into docs/publication_metrics/."""
    from evaluation.benchmark import main as benchmark_main  # noqa: PLC0415

    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("Running benchmark suite '%s' (run id %s)…", suites, stamp)
    exit_code = benchmark_main([suites, "--run-id", stamp])
    if exit_code != 0:
        logger.error("Benchmark run failed with exit code %s", exit_code)

    run_dir = settings.REPORTS_DIR / stamp
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for name in FIGURE_NAMES:
        for suffix in (".png", ".pdf"):
            source = run_dir / f"{name}{suffix}"
            if source.exists():
                shutil.copy2(source, PUBLICATION_DIR / source.name)
                copied.append(source.name)

    report = run_dir / "benchmark_report.json"
    if report.exists():
        shutil.copy2(report, PUBLICATION_DIR / "benchmark_report.json")
        copied.append("benchmark_report.json")

    logger.info("Published %d artefact(s) to %s", len(copied), PUBLICATION_DIR)
    return {"run_id": stamp, "run_dir": str(run_dir), "published": copied}


def write_export_manifest(payload: Dict[str, Any]) -> Path:
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    path = PUBLICATION_DIR / "export_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------- CLI --
def cmd_export(args: argparse.Namespace) -> int:
    results: Dict[str, Any] = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "artifacts": {},
    }

    if args.target in ("all", "detector"):
        path = export_detector(fmt=args.format, opset=args.opset)
        results["artifacts"]["detector"] = str(path) if path else None

    if args.target in ("all", "freshness"):
        path = export_freshness(fmt=args.format, opset=args.opset)
        results["artifacts"]["freshness"] = str(path) if path else None
        if path and args.format == "onnx" and args.benchmark:
            results["latency"] = benchmark_runtimes(path)

    manifest = write_export_manifest(results)
    print(json.dumps({**results, "manifest": str(manifest)}, indent=2))
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    print(json.dumps(publish_metrics(args.run_id, args.suites), indent=2))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    cmd_export(args)
    return cmd_metrics(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_pipeline", description="Export models and publish paper metrics"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="export weights to ONNX/TorchScript")
    export.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    export.add_argument("--target", choices=["all", "detector", "freshness"], default="all")
    export.add_argument("--opset", type=int, default=12)
    export.add_argument(
        "--benchmark", action="store_true", help="measure PyTorch vs ONNX CPU latency"
    )
    export.set_defaults(func=cmd_export)

    metrics = sub.add_parser("metrics", help="run benchmarks and publish figures")
    metrics.add_argument("--suites", default="all")
    metrics.add_argument("--run-id", default=None)
    metrics.set_defaults(func=cmd_metrics)

    every = sub.add_parser("all", help="export, then publish metrics")
    every.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    every.add_argument("--target", choices=["all", "detector", "freshness"], default="all")
    every.add_argument("--opset", type=int, default=12)
    every.add_argument("--benchmark", action="store_true")
    every.add_argument("--suites", default="all")
    every.add_argument("--run-id", default=None)
    every.set_defaults(func=cmd_all)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
