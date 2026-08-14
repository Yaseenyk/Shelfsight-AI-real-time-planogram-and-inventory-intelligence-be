"""Schema creation plus an optional demo seed.

`python -m app.db.init_db --seed` gives a working dashboard (products, a
planogram and one reconciliation) without any camera or weights present.
"""

from __future__ import annotations

import argparse
from typing import List

from sqlalchemy import select

from app.core.logging import configure_logging, get_logger
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.models import (  # noqa: F401 - imported for metadata registration
    ComplianceAudit,
    ExpiryAudit,
    FreshnessAudit,
    InventoryLog,
    PlanogramLayout,
    Product,
    ScanSession,
)
from app.services import planogram_store

logger = get_logger(__name__)

DEMO_PRODUCTS: List[dict] = [
    {
        "sku": "SKU-COLA-330",
        "name": "Cola Classic 330ml",
        "category": "beverages",
        "brand": "Fizzco",
        "detection_class_name": "cola_can",
        "unit_price": 1.20,
        "system_stock": 12,
        "reorder_threshold": 6,
    },
    {
        "sku": "SKU-WATER-500",
        "name": "Spring Water 500ml",
        "category": "beverages",
        "brand": "Blue Peak",
        "detection_class_name": "water_bottle",
        "unit_price": 0.90,
        "system_stock": 18,
        "reorder_threshold": 8,
    },
    {
        "sku": "SKU-CHIPS-150",
        "name": "Salted Chips 150g",
        "category": "snacks",
        "brand": "Crispy",
        "detection_class_name": "chips_bag",
        "unit_price": 2.40,
        "system_stock": 9,
        "reorder_threshold": 4,
    },
    {
        "sku": "SKU-BANANA-1KG",
        "name": "Bananas 1kg",
        "category": "produce",
        "brand": "FarmFresh",
        "detection_class_name": "banana",
        "unit_price": 1.80,
        "system_stock": 7,
        "is_perishable": True,
        "shelf_life_days": 6,
    },
    {
        "sku": "SKU-MILK-1L",
        "name": "Whole Milk 1L",
        "category": "dairy",
        "brand": "Meadow",
        "detection_class_name": "milk_carton",
        "unit_price": 1.55,
        "system_stock": 10,
        "is_perishable": True,
        "shelf_life_days": 10,
    },
]


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured at %s", engine.url)


def seed() -> None:
    """Idempotent demo data: products + every planogram JSON in data/planograms."""
    with SessionLocal() as db:
        for payload in DEMO_PRODUCTS:
            exists = db.execute(
                select(Product).where(Product.sku == payload["sku"])
            ).scalar_one_or_none()
            if exists is None:
                db.add(Product(**payload))
        db.flush()

        for document in planogram_store.load_from_disk():
            planogram_store.upsert(db, document, is_active=True)
            logger.info("Seeded planogram %s", document.planogram_id)

        db.commit()
    logger.info("Seed complete: %d products", len(DEMO_PRODUCTS))


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Initialise the ShelfSight database")
    parser.add_argument("--seed", action="store_true", help="insert demo catalogue + planograms")
    args = parser.parse_args()

    init_db()
    if args.seed:
        seed()
