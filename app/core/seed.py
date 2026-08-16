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
from app.models.enums import UserRole
from app.models.planogram import PlanogramLayout
from app.models.product import Product
from app.models.user import User
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
        "barcode": "8901234500017",
        #: A 52g crisp packet hangs many more facings than a 1kg bag of fruit.
        "units_per_row": 24,
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
        "barcode": "8901234500024",
        #: A 52g crisp packet hangs many more facings than a 1kg bag of fruit.
        "units_per_row": 18,
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
        "barcode": "8901234500031",
        #: A 52g crisp packet hangs many more facings than a 1kg bag of fruit.
        "units_per_row": 45,
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
        "barcode": "8901234500048",
        #: A 52g crisp packet hangs many more facings than a 1kg bag of fruit.
        "units_per_row": 30,
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
        "barcode": "8901234500055",
        #: A 52g crisp packet hangs many more facings than a 1kg bag of fruit.
        "units_per_row": 12,
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
        "barcode": "8901234500062",
        #: A 52g crisp packet hangs many more facings than a 1kg bag of fruit.
        "units_per_row": 14,
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
        "barcode": "8901234500079",
        #: A 52g crisp packet hangs many more facings than a 1kg bag of fruit.
        "units_per_row": 14,
        "is_perishable": True,
        "shelf_life_days": 10,
    },
]


def catalogue_is_empty(db: Session) -> bool:
    return db.execute(select(func.count()).select_from(Product)).scalar_one() == 0


def users_are_empty(db: Session) -> bool:
    return db.execute(select(func.count()).select_from(User)).scalar_one() == 0


def planograms_are_empty(db: Session) -> bool:
    return db.execute(select(func.count()).select_from(PlanogramLayout)).scalar_one() == 0


def seed_products(db: Session, force: bool = False) -> int:
    """Insert missing catalogue rows. Returns how many were added.

    Insert-only, and called only when the catalogue is empty. An operator who
    deletes a product they do not stock must not find it back after a restart,
    so "row absent" cannot on its own mean "row should be added".
    """
    existing = {sku for (sku,) in db.execute(select(Product.sku)).all()}
    added = 0
    for payload in SEED_PRODUCTS:
        if payload["sku"] in existing:
            continue
        db.add(Product(**payload))
        added += 1
    if added:
        db.flush()
    return added


def backfill_catalogue_fields(db: Session) -> int:
    """Fill in catalogue facts on existing rows that have never been set.

    Separate from seeding on purpose. Seeding inserts rows and must not
    resurrect deleted ones; this only ever *updates* a row that is already
    there, so it is safe to run on every boot -- which is what a deployment
    upgraded after a new field was added actually needs.

    Touches only fields the operator cannot have chosen: a null barcode, or a
    units_per_row still at its column default. Prices and stock levels are
    operator data and are never written.
    """
    by_sku = {row.sku: row for row in db.execute(select(Product)).scalars()}
    changed = 0
    for payload in SEED_PRODUCTS:
        row = by_sku.get(payload["sku"])
        if row is None:
            continue
        touched = False
        if row.barcode is None and payload.get("barcode"):
            row.barcode = payload["barcode"]
            touched = True
        if payload.get("units_per_row") and row.units_per_row in (None, 20):
            row.units_per_row = payload["units_per_row"]
            touched = True
        changed += int(touched)
    if changed:
        db.flush()
        logger.info("Backfilled catalogue fields on %d existing product(s)", changed)
    return changed


def seed_planograms(db: Session) -> int:
    """Upsert every planogram JSON in `data/planograms/`. Returns how many."""
    documents = planogram_store.load_from_disk()
    for document in documents:
        planogram_store.upsert(db, document, is_active=True)
    if documents:
        db.flush()
    return len(documents)


#: Demo accounts, one per role.
#:
#: These PINs are published in the README and are therefore public. They exist
#: so the system is usable the moment it starts, which matters for a handover
#: and a viva; they are not a security posture. `seed_users` refuses to touch an
#: account whose PIN has been changed, so a real deployment can replace them and
#: never see them come back.
SEED_USERS: List[Dict[str, Any]] = [
    {
        "username": "manager",
        "name": "Asha Menon",
        "role": UserRole.MANAGER,
        "pin": "1001",
    },
    {
        "username": "coordinator",
        "name": "Ravi Kumar",
        "role": UserRole.COORDINATOR,
        "pin": "2002",
    },
    {
        "username": "staff",
        "name": "Priya Nair",
        "role": UserRole.STAFF,
        "pin": "3003",
    },
]


def seed_users(db: Session) -> int:
    """Create the demo accounts, skipping any username that already exists.

    Never updates an existing row: if an operator changed the manager PIN, a
    restart must not silently reset it to the published one.
    """
    existing = {username for (username,) in db.execute(select(User.username)).all()}
    created = 0
    for payload in SEED_USERS:
        if payload["username"] in existing:
            continue
        user = User(
            username=payload["username"],
            name=payload["name"],
            role=payload["role"],
        )
        user.set_pin(payload["pin"])
        db.add(user)
        created += 1
    if created:
        db.flush()
    return created


def seed(db: Session, force: bool = False) -> Dict[str, int]:
    """Seed catalogue + planograms, returning what was inserted."""
    products = seed_products(db, force=force)
    planograms = seed_planograms(db)
    users = seed_users(db)
    db.commit()

    if products or planograms or users:
        logger.info(
            "Seeded %d product(s), %d planogram(s), %d user(s)", products, planograms, users
        )
    return {"products": products, "planograms": planograms, "users": users}


def seed_if_empty(db: Session) -> Dict[str, int]:
    """Startup path: insert only what is absent, backfill what was never set.

    Three different questions, deliberately answered separately:

    * **Insert** runs only when a table is empty, so deleting a seeded row is a
      decision the operator gets to keep.
    * **Backfill** runs every boot, because it only updates rows that already
      exist and is how a long-running deployment picks up a field added after
      it was installed.
    * **Planograms** stay gated: upserting whole documents is heavy and only
      wanted on a fresh database.
    """
    products = seed_products(db) if catalogue_is_empty(db) else 0
    # An empty users table means nobody can sign in at all, which is different
    # from an operator having removed one account among several.
    users = seed_users(db) if users_are_empty(db) else 0
    planograms = seed_planograms(db) if planograms_are_empty(db) else 0
    backfilled = backfill_catalogue_fields(db)
    db.commit()

    if products or users or planograms:
        logger.info(
            "Seed complete: %d product(s), %d planogram(s), %d user(s). The API is usable now.",
            products,
            planograms,
            users,
        )
    elif backfilled:
        logger.info("Catalogue already present; %d row(s) backfilled", backfilled)

    return {"products": products, "planograms": planograms, "users": users}
