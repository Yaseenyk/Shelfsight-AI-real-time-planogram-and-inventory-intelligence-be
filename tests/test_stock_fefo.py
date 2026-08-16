"""First Expired First Out: ordering on placement, and consumption from the front.

The bug these guard against produced no error. Placements were re-queried
through a row already in the session's identity map, so the freshly-added
placement was missing from the cached collection, every placement kept sequence
0, and the shelf was no longer ordered by expiry. Each individual step
succeeded; only the outcome was wrong -- which for this feature means stock
expiring on the shelf behind newer stock, the exact loss it exists to prevent.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

import app.models  # noqa: F401 - registers every table
from app.core.seed import seed_products, seed_users
from app.db.base import Base, utcnow
from app.models.product import Product
from app.models.shelf import Batch, Placement, RestockTask, ShelfRow
from app.services import shelf_manager, stock


@pytest.fixture
def db() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    seed_products(session)
    seed_users(session)
    session.commit()
    yield session
    session.close()


@pytest.fixture
def row(db: Session):
    product = db.execute(select(Product)).scalars().first()
    shelf = shelf_manager.create_shelf(db, code="T1", name="Test bay", row_count=1)
    target = shelf.rows[0]
    shelf_manager.allocate_row(
        db, row_id=target.id, product_id=product.id, capacity=50, buffer_threshold=5
    )
    return db.get(ShelfRow, target.id)


def _batch(db: Session, product_id: int, code: str, days: int, qty: int) -> Batch:
    batch = Batch(
        batch_code=code,
        product_id=product_id,
        expiry_date=date.today() + timedelta(days=days),
        quantity_received=qty,
        quantity_remaining=qty,
        received_at=utcnow(),
    )
    db.add(batch)
    db.commit()
    return batch


def _sequence(db: Session, row_id: int) -> list[str]:
    """Batch codes front-to-back."""
    fresh = db.execute(
        select(ShelfRow)
        .where(ShelfRow.id == row_id)
        .options(selectinload(ShelfRow.placements).selectinload(Placement.batch))
        .execution_options(populate_existing=True)
    ).scalar_one()
    return [p.batch.batch_code for p in sorted(fresh.placements, key=lambda p: p.sequence)]


def test_soonest_expiry_goes_to_the_front(db: Session, row):
    """Placed newest-first on purpose: the service must reorder it itself."""
    product_id = row.allocation.product_id
    late = _batch(db, product_id, "LATE", days=90, qty=10)
    soon = _batch(db, product_id, "SOON", days=5, qty=10)

    stock.place_on_row(db, row_id=row.id, batch_id=late.id, quantity=10)
    stock.place_on_row(db, row_id=row.id, batch_id=soon.id, quantity=10)

    assert _sequence(db, row.id) == ["SOON", "LATE"]


def test_every_placement_gets_a_distinct_sequence(db: Session, row):
    """The regression itself: all placements silently kept sequence 0."""
    product_id = row.allocation.product_id
    for index, days in enumerate((60, 10, 30)):
        batch = _batch(db, product_id, f"B{index}", days=days, qty=5)
        stock.place_on_row(db, row_id=row.id, batch_id=batch.id, quantity=5)

    fresh = db.execute(
        select(ShelfRow)
        .where(ShelfRow.id == row.id)
        .options(selectinload(ShelfRow.placements))
        .execution_options(populate_existing=True)
    ).scalar_one()
    sequences = sorted(p.sequence for p in fresh.placements)
    assert sequences == [0, 1, 2], "each placement needs its own front-to-back position"


def test_a_sale_consumes_the_soonest_expiring_first(db: Session, row):
    product_id = row.allocation.product_id
    soon = _batch(db, product_id, "SOON", days=5, qty=10)
    late = _batch(db, product_id, "LATE", days=90, qty=10)
    stock.place_on_row(db, row_id=row.id, batch_id=soon.id, quantity=10)
    stock.place_on_row(db, row_id=row.id, batch_id=late.id, quantity=10)

    fresh = db.get(ShelfRow, row.id)
    stock.sell_from_row(db, row=stock._load_row(db, fresh.id), quantity=10)

    assert _sequence(db, row.id) == ["LATE"], "the soonest-expiring batch should be gone"


def test_capacity_is_enforced(db: Session, row):
    product_id = row.allocation.product_id
    batch = _batch(db, product_id, "BIG", days=30, qty=100)
    with pytest.raises(stock.StockError, match="fit"):
        stock.place_on_row(db, row_id=row.id, batch_id=batch.id, quantity=60)


def test_a_batch_of_another_product_is_refused(db: Session, row):
    other = db.execute(select(Product)).scalars().all()[1]
    batch = _batch(db, other.id, "WRONG", days=30, qty=5)
    with pytest.raises(stock.StockError, match="different product"):
        stock.place_on_row(db, row_id=row.id, batch_id=batch.id, quantity=5)


def test_dropping_to_the_buffer_raises_one_task(db: Session, row):
    product_id = row.allocation.product_id
    batch = _batch(db, product_id, "B", days=30, qty=20)
    stock.place_on_row(db, row_id=row.id, batch_id=batch.id, quantity=20)

    # Capacity 50, buffer 5: selling 16 leaves 4, below the buffer.
    stock.sell_from_row(db, row=stock._load_row(db, row.id), quantity=16)
    tasks = db.execute(select(RestockTask).where(RestockTask.row_id == row.id)).scalars().all()
    assert len(tasks) == 1

    # A further sale must not pile up duplicates of the same job.
    stock.sell_from_row(db, row=stock._load_row(db, row.id), quantity=1)
    tasks = db.execute(select(RestockTask).where(RestockTask.row_id == row.id)).scalars().all()
    assert len(tasks) == 1, "one open task per row, however many sales follow"


def test_refilling_closes_the_task(db: Session, row):
    """Someone refilling without touching the task list is the normal case."""
    product_id = row.allocation.product_id
    first = _batch(db, product_id, "B1", days=30, qty=20)
    stock.place_on_row(db, row_id=row.id, batch_id=first.id, quantity=20)
    stock.sell_from_row(db, row=stock._load_row(db, row.id), quantity=17)

    task = db.execute(select(RestockTask).where(RestockTask.row_id == row.id)).scalar_one()
    assert task.status.value == "open"

    second = _batch(db, product_id, "B2", days=120, qty=30)
    stock.place_on_row(db, row_id=row.id, batch_id=second.id, quantity=30)

    db.refresh(task)
    assert task.status.value == "done"
