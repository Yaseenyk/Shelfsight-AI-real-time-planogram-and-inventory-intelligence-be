"""First-boot database seeding.

Runs from the FastAPI lifespan. The system must be *usable* the moment the
container starts — an empty catalogue means every scan reports zero SKUs and the
dashboard looks broken rather than empty, which is the worst first impression a
handover can make.

Two rules:

- **Idempotent.** Seeding checks for existing rows and never overwrites operator
  edits. A restart must not resurrect a product someone deliberately deleted or
  reset a stock count someone corrected.
- **Only when empty.** `seed_if_empty()` is the startup path and does nothing
  once the catalogue exists. `seed(force=True)` is the explicit CLI path.

The catalogue maps SKUs to **COCO class names**, matching `data/class_map.json`,
so a stock YOLOv8n detects real products out of the box. Once the detector is
fine-tuned on shelf SKUs, `detection_class_name` is what changes.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.planogram import PlanogramLayout
from app.models.product import Product
from app.services import planogram_store

logger = get_logger(__name__)

#: Seed catalogue. `detection_class_name` values are COCO-80 classes so the
#: pretrained detector resolves them via data/class_map.json on first boot.
SEED_PRODUCTS: List[Dict[str, Any]] = [
    {
        "sku": "SKU-WATER-500",
        "name": "Spring Water 500ml",
        "category": "beverages",
        "brand": "Blue Peak",
        "detection_class_name": "bottle",
        "unit_price": 0.90,
        "system_stock": 18,
        "reorder_threshold": 8,
    },
    {
        "sku": "SKU-COLA-330",
        "name": "Cola Classic 330ml",
        "category": "beverages",
        "brand": "Fizzco",
        "detection_class_name": "cup",
        "unit_price": 1.20,
        "system_stock": 12,
        "reorder_threshold": 6,
    },
    {
        "sku": "SKU-CHIPS-150",
        "name": "Salted Chips 150g",
        "category": "snacks",
        "brand": "Crispy",
        "detection_class_name": "sandwich",
        "unit_price": 2.40,
        "system_stock": 9,
        "reorder_threshold": 4,
    },
    {
        "sku": "SKU-MILK-1L",
        "name": "Whole Milk 1L",
        "category": "dairy",
        "brand": "Meadow",
        "detection_class_name": "vase",  # closest COCO carton-like class
        "unit_price": 1.55,
        "system_stock": 10,
        "reorder_threshold": 5,
        "is_perishable": True,
        "shelf_life_days": 10,
    },
    {
        "sku": "SKU-BANANA-1KG",
        "name": "Bananas 1kg",
        "category": "produce",
        "brand": "FarmFresh",
        "detection_class_name": "banana",
        "unit_price": 1.80,
        "system_stock": 7,
        "reorder_threshold": 3,
        "is_perishable": True,
        "shelf_life_days": 6,
    },
    {
        "sku": "SKU-APPLE-1KG",
        "name": "Gala Apples 1kg",
        "category": "produce",
        "brand": "FarmFresh",
        "detection_class_name": "apple",
        "unit_price": 2.10,
        "system_stock": 11,
        "reorder_threshold": 5,
        "is_perishable": True,
        "shelf_life_days": 14,
    },
    {
        "sku": "SKU-ORANGE-1KG",
        "name": "Navel Oranges 1kg",
        "category": "produce",
        "brand": "FarmFresh",
        "detection_class_name": "orange",
        "unit_price": 2.30,
        "system_stock": 8,
        "reorder_threshold": 4,
        "is_perishable": True,
        "shelf_life_days": 12,
    },
]


def catalogue_is_empty(db: Session) -> bool:
    return db.execute(select(func.count()).select_from(Product)).scalar_one() == 0


def planograms_are_empty(db: Session) -> bool:
    return db.execute(select(func.count()).select_from(PlanogramLayout)).scalar_one() == 0


def seed_products(db: Session, force: bool = False) -> int:
    """Insert missing catalogue rows. Returns how many were added."""
    existing = {
        sku for (sku,) in db.execute(select(Product.sku)).all()
    }
    added = 0
    for payload in SEED_PRODUCTS:
        if payload["sku"] in existing and not force:
            continue
        if payload["sku"] in existing:
            continue  # `force` re-seeds planograms, never overwrites product edits
        db.add(Product(**payload))
        added += 1
    if added:
        db.flush()
    return added


def seed_planograms(db: Session) -> int:
    """Upsert every planogram JSON in `data/planograms/`. Returns how many."""
    documents = planogram_store.load_from_disk()
    for document in documents:
        planogram_store.upsert(db, document, is_active=True)
    if documents:
        db.flush()
    return len(documents)


def seed(db: Session, force: bool = False) -> Dict[str, int]:
    """Seed catalogue + planograms, returning what was inserted."""
    products = seed_products(db, force=force)
    planograms = seed_planograms(db)
    db.commit()

    if products or planograms:
        logger.info("Seeded %d product(s) and %d planogram(s)", products, planograms)
    return {"products": products, "planograms": planograms}


def seed_if_empty(db: Session) -> Dict[str, int]:
    """Startup path: seed only a fresh database, and never touch a populated one."""
    needs_products = catalogue_is_empty(db)
    needs_planograms = planograms_are_empty(db)

    if not needs_products and not needs_planograms:
        logger.debug("Database already populated — skipping seed")
        return {"products": 0, "planograms": 0}

    logger.info(
        "First boot detected (products=%s, planograms=%s) — seeding",
        "empty" if needs_products else "present",
        "empty" if needs_planograms else "present",
    )

    products = seed_products(db) if needs_products else 0
    planograms = seed_planograms(db) if needs_planograms else 0
    db.commit()

    logger.info(
        "Seed complete: %d product(s), %d planogram(s). The API is usable now.",
        products,
        planograms,
    )
    return {"products": products, "planograms": planograms}
