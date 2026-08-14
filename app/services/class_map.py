"""Detector class → catalogue SKU resolution.

The compliance engine matches a planogram slot only against a detection of the
*same SKU*, and the inventory reconciler counts facings per SKU. A raw detector
emits class names from its training set (`bottle`, `banana`, … for a
COCO-pretrained YOLOv8n), so something has to bridge the two vocabularies.

Two resolution paths, checked in order:

1. **`data/class_map.json`** — an explicit `detector class name → SKU` table.
   This is what makes a stock COCO checkpoint usable end-to-end before the
   fine-tuned model exists.
2. **`Product.detection_class_name`** — the catalogue's own mapping, which takes
   over once the detector is fine-tuned on real SKU classes.

Unmapped detections keep `sku = None`. They are *not* discarded: they still
count as `EXTRA` in a compliance audit, which is exactly right — an unrecognised
object on the shelf is a finding, not a non-event.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.product import Product
from app.schemas.common import Detection

logger = get_logger(__name__)


class ClassMap:
    """Case-insensitive detector-class → SKU lookup, loaded from JSON."""

    def __init__(self, mapping: Optional[Dict[str, str]] = None, source: str = "memory") -> None:
        self._mapping: Dict[str, str] = {
            str(key).strip().lower(): str(value) for key, value in (mapping or {}).items()
        }
        self.source = source

    def __len__(self) -> int:
        return len(self._mapping)

    def __contains__(self, class_name: str) -> bool:
        return str(class_name).strip().lower() in self._mapping

    @property
    def mapping(self) -> Dict[str, str]:
        return dict(self._mapping)

    def resolve(self, class_name: str) -> Optional[str]:
        return self._mapping.get(str(class_name).strip().lower())

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "ClassMap":
        """Load `data/class_map.json`; a missing or invalid file yields an empty map."""
        file_path = Path(path or settings.DETECTION_CLASS_MAP)
        if not file_path.exists():
            logger.info("No class map at %s — falling back to catalogue mapping", file_path)
            return cls(source=str(file_path))

        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Invalid class map %s: %s", file_path, exc)
            return cls(source=str(file_path))

        raw = payload.get("mapping", payload) if isinstance(payload, dict) else {}
        if not isinstance(raw, dict):
            logger.error("Class map %s must contain an object under 'mapping'", file_path)
            return cls(source=str(file_path))

        mapping = {k: v for k, v in raw.items() if isinstance(v, str) and v}
        logger.info("Loaded %d class→SKU mappings from %s", len(mapping), file_path.name)
        return cls(mapping, source=str(file_path))


_class_map: Optional[ClassMap] = None
_lock = Lock()


def get_class_map(reload: bool = False) -> ClassMap:
    """Process-wide class map singleton."""
    global _class_map
    if _class_map is None or reload:
        with _lock:
            if _class_map is None or reload:
                _class_map = ClassMap.from_file()
    return _class_map


def catalogue_mapping(db: Session) -> Dict[str, str]:
    """`detection_class_name → sku` straight from the product catalogue."""
    stmt = select(Product).where(Product.detection_class_name.is_not(None))
    return {
        str(product.detection_class_name).strip().lower(): product.sku
        for product in db.execute(stmt).scalars().all()
    }


def resolve_detections(
    detections: Sequence[Detection],
    db: Optional[Session] = None,
    class_map: Optional[ClassMap] = None,
) -> List[Detection]:
    """Return copies of `detections` with `sku` populated where resolvable.

    The catalogue wins over the JSON file: once a SKU is trained into the
    detector, its own `detection_class_name` is the authoritative mapping.
    """
    if not detections:
        return []

    file_map = (class_map or get_class_map()).mapping
    db_map = catalogue_mapping(db) if db is not None else {}

    resolved: List[Detection] = []
    unmapped: set[str] = set()
    for detection in detections:
        key = detection.class_name.strip().lower()
        sku = detection.sku or db_map.get(key) or file_map.get(key)
        if sku is None:
            unmapped.add(detection.class_name)
        resolved.append(detection.model_copy(update={"sku": sku}))

    if unmapped:
        logger.debug("Unmapped detector classes (counted as EXTRA): %s", sorted(unmapped))
    return resolved
