"""Persistence helpers for versioned planogram documents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.planogram import PlanogramLayout
from app.schemas.planogram import PlanogramDocument

logger = get_logger(__name__)


def checksum(document: PlanogramDocument) -> str:
    """Stable SHA-256 over the canonical JSON — cite this in the paper's methods."""
    payload = json.dumps(document.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def upsert(db: Session, document: PlanogramDocument, is_active: bool = True) -> PlanogramLayout:
    existing = db.execute(
        select(PlanogramLayout).where(PlanogramLayout.planogram_uid == document.planogram_id)
    ).scalar_one_or_none()

    layout = existing or PlanogramLayout(planogram_uid=document.planogram_id)
    layout.name = document.name
    layout.version = document.version
    layout.store_id = document.store_id
    layout.aisle = document.aisle
    layout.bay = document.bay
    layout.shelf_count = len(document.shelves)
    layout.slot_count = document.slot_count
    layout.iou_threshold = document.tolerances.iou_threshold
    layout.center_distance_threshold = document.tolerances.center_distance_threshold
    layout.is_active = is_active
    layout.checksum = checksum(document)
    layout.layout_json = document.model_dump(mode="json")

    if existing is None:
        db.add(layout)
    db.flush()
    return layout


def get_active(db: Session, shelf_id: Optional[str] = None) -> Optional[PlanogramLayout]:
    """Active layout, optionally the one that actually contains `shelf_id`."""
    stmt = select(PlanogramLayout).where(PlanogramLayout.is_active.is_(True))
    layouts = db.execute(stmt.order_by(PlanogramLayout.updated_at.desc())).scalars().all()
    if not layouts:
        return None
    if shelf_id is None:
        return layouts[0]
    for layout in layouts:
        shelves = (layout.layout_json or {}).get("shelves", [])
        if any(shelf.get("shelf_id") == shelf_id for shelf in shelves):
            return layout
    return layouts[0]


def to_document(layout: PlanogramLayout) -> PlanogramDocument:
    return PlanogramDocument.model_validate(layout.layout_json)


def expected_skus(db: Session, shelf_id: Optional[str] = None) -> List[str]:
    """SKUs the active planogram says belong on a shelf (or on the whole bay).

    This is what makes phantom inventory detectable on a scoped shelf scan: an
    expected SKU that produced no detection is a phantom, and without the
    planogram there is nothing to compare an empty slot against.
    """
    layout = get_active(db, shelf_id)
    if layout is None:
        return []

    skus: List[str] = []
    for shelf in (layout.layout_json or {}).get("shelves", []):
        if shelf_id and shelf.get("shelf_id") != shelf_id:
            continue
        for row in shelf.get("rows", []):
            for slot in row.get("slots", []):
                sku = slot.get("sku")
                if sku and sku not in skus:
                    skus.append(sku)
    return skus


def load_from_disk(path: Optional[Path] = None) -> List[PlanogramDocument]:
    """Read every `*.json` planogram in `data/planograms/`."""
    directory = Path(path or settings.PLANOGRAM_DIR)
    if not directory.exists():
        return []
    documents: List[PlanogramDocument] = []
    for file in sorted(directory.glob("*.json")):
        try:
            documents.append(PlanogramDocument.model_validate_json(file.read_text("utf-8")))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop startup
            logger.error("Invalid planogram %s: %s", file.name, exc)
    return documents
