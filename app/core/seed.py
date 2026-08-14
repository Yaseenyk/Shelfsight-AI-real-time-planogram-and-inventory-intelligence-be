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

#: Seed catalogue for an Indian kirana / modern-trade shelf.
#:
#: Two things are deliberate here:
#: - **Prices are MRP in rupees.** A store manager shown "$41.40 at risk" on a
#:   shelf priced in rupees misreads the exposure by roughly 85x, and every
#:   value-at-risk figure in the dashboard and the LLM briefing derives from
#:   `unit_price`.
#: - **`detection_class_name` stays a COCO-80 class.** A Bisleri bottle is still
#:   a `bottle` to the pretrained detector, so the seeded demo works on first
#:   boot without a fine-tuned Indian SKU model.
SEED_PRODUCTS: List[Dict[str, Any]] = [
    {
        "sku": "SKU-WATER-1L",
        "name": "Packaged Drinking Water 1L",
        "category": "beverages",
        "brand": "Bisleri",
        "detection_class_name": "bottle",
        "unit_price": 20.0,
        "system_stock": 18,
        "reorder_threshold": 8,
    },
    {
        "sku": "SKU-COLA-750",
        "name": "Cola 750ml",
        "category": "beverages",
        "brand": "Thums Up",
        "detection_class_name": "cup",
        "unit_price": 40.0,
        "system_stock": 12,
        "reorder_threshold": 6,
    },
    {
        "sku": "SKU-CHIPS-52",
        "name": "Potato Chips 52g",
        "category": "snacks",
        "brand": "Lay's",
        "detection_class_name": "sandwich",
        "unit_price": 20.0,
        "system_stock": 9,
        "reorder_threshold": 4,
    },
    {
        "sku": "SKU-MILK-500",
        "name": "Toned Milk 500ml",
        "category": "dairy",
        "brand": "Amul",
        "detection_class_name": "vase",  # closest COCO carton/pouch-like class
        "unit_price": 28.0,
        "system_stock": 10,
        "reorder_threshold": 5,
        "is_perishable": True,
        # Indian ambient conditions shorten dairy shelf life relative to the
        # cold chain assumed by most Western datasets.
        "shelf_life_days": 3,
    },
    {
        "sku": "SKU-BANANA-1KG",
        "name": "Robusta Banana 1kg",
        "category": "produce",
        "brand": "Fresh Produce",
        "detection_class_name": "banana",
        "unit_price": 60.0,
        "system_stock": 7,
        "reorder_threshold": 3,
        "is_perishable": True,
        "shelf_life_days": 5,
    },
    {
        "sku": "SKU-APPLE-1KG",
        "name": "Shimla Apple 1kg",
        "category": "produce",
        "brand": "Fresh Produce",
        "detection_class_name": "apple",
        "unit_price": 180.0,
        "system_stock": 11,
        "reorder_threshold": 5,
        "is_perishable": True,
        "shelf_life_days": 14,
    },
    {
        "sku": "SKU-ORANGE-1KG",
        "name": "Nagpur Orange 1kg",
        "category": "produce",
        "brand": "Fresh Produce",
        "detection_class_name": "orange",
        "unit_price": 120.0,
        "system_stock": 8,
        "reorder_threshold": 4,
        "is_perishable": True,
        "shelf_life_days": 10,
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
