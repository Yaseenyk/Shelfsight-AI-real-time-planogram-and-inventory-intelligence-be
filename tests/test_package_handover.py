"""Handover packaging: refuse to ship a bundle that cannot work.

The failure this guards against is quiet. Trained weights are gitignored, so a
recipient given only the repository starts the system, sees it report healthy,
and gets nonsense results because Ultralytics substituted the COCO baseline.
Packaging is therefore allowed to fail loudly but never allowed to produce a
bundle that looks complete and is not.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tools import package_handover
from tools.package_handover import configured_weights, main, sha256


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A miniature project tree standing in for the repository root."""
    root = tmp_path / "be"
    (root / "scripts").mkdir(parents=True)
    (root / "models" / "weights").mkdir(parents=True)
    (root / "docs" / "publication_metrics").mkdir(parents=True)

    for name in ("READ_ME_FIRST.md", "START.bat", "STOP.bat"):
        (root / name).write_text(f"# {name}", encoding="utf-8")
    (root / "scripts" / "start_all.ps1").write_text("# launcher", encoding="utf-8")
    (root / ".env.example").write_text(
        "DETECTION_WEIGHTS=models/weights/detector.pt\n"
        "FRESHNESS_WEIGHTS=models/weights/freshness.pt\n",
        encoding="utf-8",
    )
    (root / "LICENSING.md").write_text("# licensing", encoding="utf-8")

    monkeypatch.setattr(package_handover, "ROOT", root)
    return root


def _add_weights(root: Path, *names: str) -> None:
    for name in names:
        (root / "models" / "weights" / name).write_bytes(name.encode() * 64)


# ------------------------------------------------------- configured_weights --
def test_configured_weights_reads_both_keys(project: Path):
    found = configured_weights(project / ".env.example")
    assert found == {
        "DETECTION_WEIGHTS": "models/weights/detector.pt",
        "FRESHNESS_WEIGHTS": "models/weights/freshness.pt",
    }


def test_configured_weights_ignores_comments_and_blanks(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "# DETECTION_WEIGHTS=commented/out.pt\n"
        "\n"
        "FRESHNESS_WEIGHTS=models/weights/real.pt\n"
        "UNRELATED=value\n",
        encoding="utf-8",
    )
    assert configured_weights(env) == {"FRESHNESS_WEIGHTS": "models/weights/real.pt"}


def test_configured_weights_on_a_missing_file_is_empty(tmp_path: Path):
    assert configured_weights(tmp_path / "nope.env") == {}


# ------------------------------------------------------------------ sha256 --
def test_sha256_is_stable_and_content_sensitive(tmp_path: Path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"identical")
    b = tmp_path / "b.bin"
    b.write_bytes(b"identical")
    c = tmp_path / "c.bin"
    c.write_bytes(b"different")

    assert sha256(a) == sha256(b)
    assert sha256(a) != sha256(c)


# -------------------------------------------------------------------- main --
def test_refuses_when_a_configured_weight_is_absent(project: Path, tmp_path: Path):
    _add_weights(project, "freshness.pt")  # detector.pt deliberately absent

    code = main(["--out", str(tmp_path / "bundle.zip")])
    assert code == 1
    assert not (tmp_path / "bundle.zip").exists(), "no archive on refusal"


def test_packages_when_every_weight_is_present(project: Path, tmp_path: Path):
    _add_weights(project, "detector.pt", "freshness.pt")
    out = tmp_path / "bundle.zip"

    assert main(["--out", str(out)]) == 0
    assert out.is_file()

    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        assert "models/weights/detector.pt" in names
        assert "models/weights/freshness.pt" in names
        assert "READ_ME_FIRST.md" in names
        assert "START.bat" in names
        assert "HANDOVER_MANIFEST.json" in names


def test_manifest_records_a_checksum_per_weight(project: Path, tmp_path: Path):
    _add_weights(project, "detector.pt", "freshness.pt")
    out = tmp_path / "bundle.zip"
    main(["--out", str(out)])

    with zipfile.ZipFile(out) as archive:
        manifest = json.loads(archive.read("HANDOVER_MANIFEST.json"))

    assert set(manifest["weights"]) == {"DETECTION_WEIGHTS", "FRESHNESS_WEIGHTS"}
    for entry in manifest["weights"].values():
        assert len(entry["sha256"]) == 64
        assert entry["size_mb"] >= 0
    assert manifest["missing_weights"] == []


def test_allow_missing_weights_packages_and_records_the_gap(project: Path, tmp_path: Path):
    _add_weights(project, "freshness.pt")
    out = tmp_path / "bundle.zip"

    assert main(["--out", str(out), "--allow-missing-weights"]) == 0
    with zipfile.ZipFile(out) as archive:
        manifest = json.loads(archive.read("HANDOVER_MANIFEST.json"))

    # The gap must be recorded, not silently tolerated.
    assert any("DETECTION_WEIGHTS" in item for item in manifest["missing_weights"])


def test_includes_sibling_onnx_exports(project: Path, tmp_path: Path):
    """The container serves ONNX, so shipping only .pt would degrade it."""
    _add_weights(project, "detector.pt", "freshness.pt", "freshness.onnx")
    out = tmp_path / "bundle.zip"
    main(["--out", str(out)])

    with zipfile.ZipFile(out) as archive:
        assert "models/weights/freshness.onnx" in archive.namelist()


def test_refuses_when_a_required_document_is_missing(project: Path, tmp_path: Path):
    _add_weights(project, "detector.pt", "freshness.pt")
    (project / "START.bat").unlink()

    assert main(["--out", str(tmp_path / "bundle.zip")]) == 1


def test_refuses_when_the_config_names_no_weights(project: Path, tmp_path: Path):
    _add_weights(project, "detector.pt", "freshness.pt")
    (project / ".env.example").write_text("APP_NAME=ShelfSight\n", encoding="utf-8")

    assert main(["--out", str(tmp_path / "bundle.zip")]) == 1


def test_does_not_ship_superseded_checkpoints(project: Path, tmp_path: Path):
    """models/weights accumulates old models; only the configured ones ship.

    A blanket glob would hand the client the freshness model trained on the
    contaminated partition beside the current one, with nothing to tell them
    apart.
    """
    _add_weights(project, "detector.pt", "freshness.pt")
    _add_weights(project, "freshness_OLD_contaminated.pt", "detector_v1_leaky.onnx")
    out = tmp_path / "bundle.zip"

    assert main(["--out", str(out)]) == 0
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()

    assert "models/weights/freshness_OLD_contaminated.pt" not in names
    assert "models/weights/detector_v1_leaky.onnx" not in names
    assert "models/weights/detector.pt" in names
    assert "models/weights/freshness.pt" in names


def test_ships_torchscript_and_onnx_of_configured_weights(project: Path, tmp_path: Path):
    _add_weights(project, "detector.pt", "freshness.pt", "freshness.onnx")
    (project / "models" / "weights" / "freshness.torchscript.pt").write_bytes(b"ts" * 32)
    out = tmp_path / "bundle.zip"

    main(["--out", str(out)])
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()

    assert "models/weights/freshness.onnx" in names
    assert "models/weights/freshness.torchscript.pt" in names
