from __future__ import annotations

import json

from app.schemas.common import BoundingBox, Detection
from app.services.class_map import ClassMap, resolve_detections


def _detection(class_name: str, sku=None) -> Detection:  # noqa: ANN001
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2),
        sku=sku,
    )


def test_lookup_is_case_insensitive():
    mapping = ClassMap({"Bottle": "SKU-WATER-500"})
    assert mapping.resolve("bottle") == "SKU-WATER-500"
    assert mapping.resolve("  BOTTLE ") == "SKU-WATER-500"
    assert mapping.resolve("banana") is None


def test_from_file_reads_mapping_block(tmp_path):  # noqa: ANN001
    path = tmp_path / "class_map.json"
    path.write_text(
        json.dumps({"detector": "yolov8n", "mapping": {"banana": "SKU-BANANA-1KG"}}),
        encoding="utf-8",
    )
    assert ClassMap.from_file(path).resolve("banana") == "SKU-BANANA-1KG"


def test_from_file_survives_missing_and_invalid_files(tmp_path):  # noqa: ANN001
    assert len(ClassMap.from_file(tmp_path / "absent.json")) == 0

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert len(ClassMap.from_file(broken)) == 0


def test_resolve_populates_sku_without_mutating_input():
    detections = [_detection("bottle"), _detection("banana")]
    resolved = resolve_detections(detections, class_map=ClassMap({"bottle": "SKU-WATER-500"}))

    assert [d.sku for d in resolved] == ["SKU-WATER-500", None]
    assert [d.sku for d in detections] == [None, None]  # originals untouched


def test_resolve_keeps_an_existing_sku():
    resolved = resolve_detections(
        [_detection("bottle", sku="SKU-OVERRIDE")],
        class_map=ClassMap({"bottle": "SKU-WATER-500"}),
    )
    assert resolved[0].sku == "SKU-OVERRIDE"


def test_unmapped_detections_are_kept_not_dropped():
    # An unrecognised object must still reach the compliance engine, where it
    # counts as EXTRA — dropping it would hide a real shelf finding.
    resolved = resolve_detections([_detection("skateboard")], class_map=ClassMap({}))
    assert len(resolved) == 1 and resolved[0].sku is None
