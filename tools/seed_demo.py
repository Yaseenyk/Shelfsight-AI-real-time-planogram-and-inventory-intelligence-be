"""Populate a shop that demonstrates every use case.

An empty system demonstrates nothing: a reviewer clicks through blank screens
and has to be told what would happen. This builds a shop that is already mid-
shift, with each situation the system exists to handle visible somewhere.

Deliberately constructed, not random. Each row below is placed to show one
thing:

* a **full row** -- the healthy case, so "low" has something to contrast with;
* a **row at its buffer** -- an open restock task waiting for the coordinator;
* a **row already assigned** -- work in progress, showing the coordinator step;
* a **row with two batches** -- the older one at the front, which is the whole
  point of FEFO and is invisible with only one batch;
* a **row with a batch expiring this week** -- the case the expiry feature is for;
* an **empty allocated row** -- allocated but never filled, which is a different
  problem from "sold out" and looks identical if you only count units.

    python -m tools.seed_demo            # add to whatever is there
    python -m tools.seed_demo --reset    # clear demo shelves first
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from typing import Dict, List, Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import configure_logging, get_logger
from app.db.base import utcnow
from app.db.session import SessionLocal
from app.models.enums import DiscrepancyType, RestockStatus, ScanStatus, Severity, UserRole
from app.models.inventory import InventoryLog
from app.models.product import Product
from app.models.scan import ScanSession
from app.models.shelf import Batch, RestockTask, Shelf
from app.models.user import User
from app.services import shelf_manager, stock

logger = get_logger(__name__)

#: Shelves to build. Codes are prefixed DEMO- so --reset can find them again
#: without touching anything a real user created.
SHELVES = [
    {
        "code": "DEMO-AISLE1",
        "name": "Aisle 1 — Drinks & Snacks",
        "location": "Front of store",
        "rows": 4,
    },
    {
        "code": "DEMO-AISLE2",
        "name": "Aisle 2 — Dairy & Fresh",
        "location": "Back wall, chilled",
        "rows": 3,
    },
]


def _products(db: Session) -> Dict[str, Product]:
    return {p.sku: p for p in db.execute(select(Product)).scalars()}


def _batch(
    db: Session,
    product: Product,
    *,
    code: str,
    days_to_expiry: int,
    quantity: int,
    receiver: Optional[User],
) -> Batch:
    batch = Batch(
        batch_code=code,
        product_id=product.id,
        expiry_date=date.today() + timedelta(days=days_to_expiry),
        quantity_received=quantity,
        quantity_remaining=quantity,
        received_at=utcnow() - timedelta(days=max(0, 30 - days_to_expiry) // 4),
        received_by_id=receiver.id if receiver else None,
    )
    db.add(batch)
    db.flush()
    return batch


def reset(db: Session) -> int:
    """Remove only what this script created."""
    shelves = db.execute(select(Shelf).where(Shelf.code.like("DEMO-%"))).scalars().all()
    for shelf in shelves:
        db.delete(shelf)
    db.execute(select(Batch).where(Batch.batch_code.like("DEMO-%")))
    for batch in db.execute(select(Batch).where(Batch.batch_code.like("DEMO-%"))).scalars():
        db.delete(batch)
    db.commit()
    return len(shelves)


def seed_shelf_check_alerts(db: Session, products: Dict[str, Product]) -> int:
    """Give the vision dashboard something to show before anyone uploads a photo.

    The shelf-check screen is driven by inventory logs, which only exist once a
    frame has been analysed. On a freshly reset system it therefore reads
    "everything looks good" -- true, but it demonstrates nothing, and a reviewer
    cannot tell a healthy shop from an empty database.

    These rows describe the same shop the shelves do: a few products the stock
    record claims are present and the camera did not find.
    """
    if db.execute(select(InventoryLog)).first():
        return 0

    session = ScanSession(
        session_uid=f"demo-{uuid4().hex[:12]}",
        shelf_id="DEMO-AISLE1",
        store_id="DEMO-STORE",
        status=ScanStatus.COMPLETED,
    )
    db.add(session)
    db.flush()

    #: (sku index, what the system thinks, what the camera saw)
    SITUATIONS = [
        (0, 18, 0),  # phantom: stock record says 18, shelf is bare
        (1, 12, 4),  # running low
        (2, 9, 0),  # phantom
        (3, 10, 14),  # more on the shelf than recorded
    ]
    skus = list(products)
    created = 0
    for sku_index, system_count, detected in SITUATIONS:
        product = products[skus[sku_index % len(skus)]]
        gap = detected - system_count
        if detected == 0 and system_count > 0:
            kind, severity = DiscrepancyType.PHANTOM, Severity.CRITICAL
        elif detected < system_count:
            kind, severity = DiscrepancyType.UNDERCOUNT, Severity.WARNING
        elif detected > system_count:
            kind, severity = DiscrepancyType.OVERCOUNT, Severity.WARNING
        else:
            kind, severity = DiscrepancyType.MATCH, Severity.INFO

        db.add(
            InventoryLog(
                session_id=session.id,
                product_id=product.id,
                detected_count=detected,
                system_count=system_count,
                discrepancy=gap,
                discrepancy_type=kind,
                severity=severity,
                shelf_id="DEMO-AISLE1",
                # Value impact is derived by the service from unit price, so it
                # is not stored here.
                mean_confidence=0.82 if detected else None,
            )
        )
        created += 1
    db.flush()
    return created


def build(db: Session) -> Dict[str, int]:
    products = _products(db)
    if not products:
        raise SystemExit("No products in the catalogue. Start the API once to seed it, then retry.")

    manager = db.execute(select(User).where(User.role == UserRole.MANAGER)).scalars().first()
    coordinator = (
        db.execute(select(User).where(User.role == UserRole.COORDINATOR)).scalars().first()
    )
    staff = db.execute(select(User).where(User.role == UserRole.STAFF)).scalars().first()

    skus = list(products)
    created_shelves: List[Shelf] = []
    for spec in SHELVES:
        if db.execute(select(Shelf).where(Shelf.code == spec["code"])).scalar_one_or_none():
            continue
        created_shelves.append(
            shelf_manager.create_shelf(
                db,
                code=spec["code"],
                name=spec["name"],
                row_count=spec["rows"],
                location=spec["location"],
                actor=manager,
            )
        )
    if not created_shelves:
        logger.info("Demo shelves already present — nothing to build")
        return {"shelves": 0, "rows": 0, "batches": 0, "tasks": 0}

    rows = [row for shelf in created_shelves for row in shelf.rows]

    #: (sku index, capacity, refill point, how full to leave it, extra batch?)
    #: The last two are what create the demonstrable situations.
    PLAN = [
        (0, 50, 12, 50, False),  # full and healthy
        (1, 40, 10, 10, False),  # sitting exactly on its buffer -> open task
        (2, 30, 8, 6, False),  # below buffer -> a second task, to assign
        (3, 24, 6, 18, True),  # two batches, older at the front
        (4, 20, 5, 4, True),  # low AND a batch expiring within the week
        (5, 36, 9, 36, False),  # full
        (6, 15, 4, 0, False),  # allocated but never filled
    ]

    batches = 0
    for index, row in enumerate(rows):
        if index >= len(PLAN):
            break
        sku_index, capacity, buffer_at, fill_to, split = PLAN[index]
        product = products[skus[sku_index % len(skus)]]

        shelf_manager.allocate_row(
            db,
            row_id=row.id,
            product_id=product.id,
            capacity=capacity,
            buffer_threshold=buffer_at,
            slotting_fee=float((index + 1) * 500),
            actor=manager,
        )
        if fill_to <= 0:
            continue

        if split:
            # Older batch first and smaller, so the front of the shelf is the
            # one that expires soonest -- the behaviour FEFO exists to produce.
            older = _batch(
                db,
                product,
                code=f"DEMO-B{index}A",
                days_to_expiry=5,
                quantity=fill_to // 3 or 1,
                receiver=coordinator,
            )
            newer = _batch(
                db,
                product,
                code=f"DEMO-B{index}B",
                days_to_expiry=90,
                quantity=fill_to - (fill_to // 3 or 1),
                receiver=coordinator,
            )
            batches += 2
            # Placed newest first on purpose: the service must sort it to the
            # back by itself, which is the thing worth demonstrating.
            stock.place_on_row(
                db, row_id=row.id, batch_id=newer.id, quantity=newer.quantity_remaining, actor=staff
            )
            stock.place_on_row(
                db, row_id=row.id, batch_id=older.id, quantity=older.quantity_remaining, actor=staff
            )
        else:
            batch = _batch(
                db,
                product,
                code=f"DEMO-B{index}",
                days_to_expiry=120,
                quantity=fill_to,
                receiver=coordinator,
            )
            batches += 1
            stock.place_on_row(db, row_id=row.id, batch_id=batch.id, quantity=fill_to, actor=staff)

    # Raise tasks for everything now sitting at or below its buffer, then put one
    # of them in the assigned state so the coordinator step is visible too.
    tasks = 0
    for shelf in created_shelves:
        for row in shelf.rows:
            fresh = db.get(type(row), row.id)
            if stock.raise_restock_if_needed(db, fresh):
                tasks += 1
    db.commit()

    open_tasks = (
        db.execute(select(RestockTask).where(RestockTask.status == RestockStatus.OPEN))
        .scalars()
        .all()
    )
    if open_tasks and coordinator and staff:
        first = open_tasks[0]
        first.status = RestockStatus.ASSIGNED
        first.assigned_to_id = staff.id
        first.assigned_by_id = coordinator.id
        db.commit()

    # Reserve stock, so the refill jobs have somewhere to be refilled from.
    # Without it the story breaks in the middle: staff are told a shelf is low
    # and the stockroom is empty, which demonstrates half a workflow.
    reserve = 0
    for index, row in enumerate(rows):
        if index >= len(PLAN):
            break
        allocation = row.allocation
        if allocation is None:
            continue
        # Two consignments per product with different dates, so the placement
        # screen has a real FEFO choice to make rather than one obvious option.
        for suffix, days, qty in (("R1", 21, 20), ("R2", 150, 40)):
            _batch(
                db,
                allocation.product,
                code=f"DEMO-{allocation.product.sku[-6:]}-{suffix}",
                days_to_expiry=days,
                quantity=qty,
                receiver=coordinator,
            )
            reserve += 1

    alerts = seed_shelf_check_alerts(db, products)
    db.commit()

    return {
        "alerts": alerts,
        "shelves": len(created_shelves),
        "rows": len(rows),
        "batches": batches + reserve,
        "tasks": tasks,
    }


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="remove DEMO- shelves first")
    args = parser.parse_args(argv)

    db = SessionLocal()
    try:
        if args.reset:
            removed = reset(db)
            print(f"removed {removed} demo shelf/shelves")
        summary = build(db)
        print(
            f"built {summary['shelves']} shelves, {summary['rows']} rows, "
            f"{summary['batches']} batches, {summary['tasks']} restock task(s)"
        )
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
