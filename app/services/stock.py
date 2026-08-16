"""Moving stock: onto a shelf, off it at checkout, and the restock it triggers.

Placement is First Expired First Out. When units go onto a row they are ordered
by expiry date, soonest at the front, and a sale always takes from the front.
The reason a shop does this is not that it is clever -- it is that the
alternative is throwing stock away because the older batch sat behind the newer
one until it expired.

Nothing here edits a running total. Shelf quantity is the sum of placements, and
every change writes a StockMovement, so a disagreement between the system and
the shelf can be traced to an event rather than guessed at.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.logging import get_logger
from app.db.base import utcnow
from app.models.enums import MovementType, RestockStatus
from app.models.product import Product
from app.models.shelf import Batch, Placement, RestockTask, RowAllocation, ShelfRow, StockMovement
from app.models.user import User

logger = get_logger(__name__)


class StockError(Exception):
    """A stock rule was broken. The message is safe to show a user."""


def _load_row(db: Session, row_id: int) -> ShelfRow:
    """Load a row with its placements, forcing a refresh of what is cached.

    `populate_existing` is load-bearing. Re-querying an object already in the
    session's identity map returns the cached instance *and leaves its loaded
    collections untouched*, so a placement added moments earlier is absent from
    `row.placements`. The resequence then sorts a stale list, every placement
    keeps sequence 0, and front-of-shelf ordering silently stops working -- with
    no error anywhere, because each individual step succeeded.
    """
    row = db.execute(
        select(ShelfRow)
        .where(ShelfRow.id == row_id)
        .options(
            selectinload(ShelfRow.placements).selectinload(Placement.batch),
            selectinload(ShelfRow.allocation).selectinload(RowAllocation.product),
        )
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if row is None:
        raise StockError("That row does not exist.")
    return row


def _resequence(row: ShelfRow) -> None:
    """Re-number placements front-to-back by expiry, soonest first.

    Called after every change rather than only on insert: removing the front
    batch must promote the next one, and a gap in the sequence would make
    "which is at the front" ambiguous.
    """
    ordered = sorted(row.placements, key=lambda p: (p.batch.expiry_date, p.batch_id))
    for index, placement in enumerate(ordered):
        placement.sequence = index


def place_on_row(
    db: Session,
    *,
    row_id: int,
    batch_id: int,
    quantity: int,
    actor: Optional[User] = None,
) -> ShelfRow:
    """Put units from a batch onto a row, keeping the row in FEFO order."""
    if quantity <= 0:
        raise StockError("Enter how many units you are putting on the shelf.")

    row = _load_row(db, row_id)
    if row.allocation is None:
        raise StockError("This row has not been given to a product yet.")

    batch = db.get(Batch, batch_id)
    if batch is None:
        raise StockError("That batch does not exist.")
    if batch.product_id != row.allocation.product_id:
        raise StockError(
            f"This row is for {row.allocation.product.name}. That batch is a different product."
        )
    if batch.quantity_remaining < quantity:
        raise StockError(
            f"Only {batch.quantity_remaining} unit(s) of that batch are left in the stockroom."
        )

    space = row.allocation.capacity - row.on_shelf
    if quantity > space:
        raise StockError(
            f"Only {space} more unit(s) fit on this row (capacity {row.allocation.capacity})."
        )

    existing = next((p for p in row.placements if p.batch_id == batch_id), None)
    if existing:
        existing.quantity += quantity
    else:
        db.add(Placement(row_id=row.id, batch_id=batch_id, quantity=quantity, sequence=0))

    batch.quantity_remaining -= quantity
    db.add(
        StockMovement(
            row_id=row.id,
            batch_id=batch_id,
            product_id=batch.product_id,
            movement_type=MovementType.PLACED,
            quantity=quantity,
            actor_id=actor.id if actor else None,
        )
    )
    db.flush()

    row = _load_row(db, row_id)
    _resequence(row)
    _close_satisfied_tasks(db, row)
    db.commit()
    return _load_row(db, row_id)


def sell_from_row(
    db: Session,
    *,
    row: ShelfRow,
    quantity: int = 1,
    actor: Optional[User] = None,
) -> List[Batch]:
    """Take units off the front of a row. Returns the batches they came from.

    Consuming from the front is what makes FEFO actually work: ordering the
    shelf achieves nothing if a sale removes from wherever is convenient.
    """
    if quantity <= 0:
        raise StockError("Quantity must be at least 1.")
    if row.on_shelf < quantity:
        raise StockError(f"Only {row.on_shelf} unit(s) are on this row.")

    taken: List[Batch] = []
    remaining = quantity
    for placement in sorted(row.placements, key=lambda p: p.sequence):
        if remaining <= 0:
            break
        take = min(placement.quantity, remaining)
        placement.quantity -= take
        remaining -= take
        taken.append(placement.batch)
        db.add(
            StockMovement(
                row_id=row.id,
                batch_id=placement.batch_id,
                product_id=placement.batch.product_id,
                movement_type=MovementType.SOLD,
                quantity=-take,
                actor_id=actor.id if actor else None,
            )
        )

    # An emptied placement is deleted rather than left at zero, so "what is on
    # this row" never includes rows of nothing.
    for placement in list(row.placements):
        if placement.quantity <= 0:
            db.delete(placement)

    db.flush()
    fresh = _load_row(db, row.id)
    _resequence(fresh)
    raise_restock_if_needed(db, fresh)
    db.commit()
    return taken


def sell_by_barcode(
    db: Session,
    *,
    barcode: str,
    quantity: int = 1,
    actor: Optional[User] = None,
) -> dict:
    """Checkout scan: find where this product lives and take it off that row.

    When a product occupies more than one row, the row whose front batch expires
    soonest is served first, so FEFO holds across the whole shop rather than
    only within a row.
    """
    product = db.execute(
        select(Product).where(Product.barcode == barcode.strip())
    ).scalar_one_or_none()
    if product is None:
        raise StockError(f"No product in the catalogue has the barcode {barcode}.")

    candidates = [
        row
        for row in db.execute(
            select(ShelfRow)
            .join(RowAllocation)
            .where(RowAllocation.product_id == product.id)
            .options(
                selectinload(ShelfRow.placements).selectinload(Placement.batch),
                selectinload(ShelfRow.allocation).selectinload(RowAllocation.product),
                selectinload(ShelfRow.shelf),
            )
        )
        .scalars()
        .unique()
        if row.on_shelf > 0
    ]
    if not candidates:
        raise StockError(f"{product.name} is not on any shelf right now.")

    def front_expiry(row: ShelfRow):
        return min(p.batch.expiry_date for p in row.placements)

    row = sorted(candidates, key=front_expiry)[0]
    batches = sell_from_row(db, row=row, quantity=quantity, actor=actor)
    fresh = _load_row(db, row.id)

    return {
        "product": product,
        "row": fresh,
        "batches": batches,
        "remaining_on_row": fresh.on_shelf,
        "needs_restock": fresh.needs_restock,
    }


def raise_restock_if_needed(db: Session, row: ShelfRow) -> Optional[RestockTask]:
    """Open a restock task when a row falls to its buffer.

    One open task per row: a shelf that keeps selling would otherwise raise a
    task per sale and bury the coordinator in duplicates of the same job.
    """
    if row.allocation is None or not row.needs_restock:
        return None

    existing = db.execute(
        select(RestockTask).where(
            RestockTask.row_id == row.id,
            RestockTask.status.in_([RestockStatus.OPEN, RestockStatus.ASSIGNED]),
        )
    ).scalar_one_or_none()
    if existing:
        return existing

    task = RestockTask(
        row_id=row.id,
        status=RestockStatus.OPEN,
        quantity_at_trigger=row.on_shelf,
        units_needed=max(1, row.allocation.capacity - row.on_shelf),
    )
    db.add(task)
    db.flush()
    logger.info(
        "Restock raised for row %s (%s): %d on shelf, needs %d",
        row.id,
        row.allocation.product.sku,
        row.on_shelf,
        task.units_needed,
    )
    return task


def _close_satisfied_tasks(db: Session, row: ShelfRow) -> None:
    """Clear open tasks once a row is back above its buffer.

    Someone refilling a shelf without touching the task list is the normal case,
    not an exception, so the work closing itself is correct.
    """
    if row.needs_restock:
        return
    tasks = (
        db.execute(
            select(RestockTask).where(
                RestockTask.row_id == row.id,
                RestockTask.status.in_([RestockStatus.OPEN, RestockStatus.ASSIGNED]),
            )
        )
        .scalars()
        .all()
    )
    for task in tasks:
        task.status = RestockStatus.DONE
        task.completed_at = utcnow()


def receive_batch(
    db: Session,
    *,
    product_id: int,
    quantity: int,
    expiry: date,
    batch_code: Optional[str] = None,
    actor: Optional[User] = None,
) -> Batch:
    """Book a delivery into the stockroom.

    Expiry is required rather than optional. A batch without one cannot be
    ordered against another, so accepting it would put stock on a shelf that
    the front-to-back rule silently cannot place.
    """
    if quantity <= 0:
        raise StockError("Enter how many units arrived.")
    product = db.get(Product, product_id)
    if product is None:
        raise StockError("That product is not in the catalogue.")

    code = (batch_code or "").strip() or f"B{product.sku[-6:]}-{utcnow():%y%m%d%H%M%S}"
    if db.execute(select(Batch).where(Batch.batch_code == code)).scalar_one_or_none():
        raise StockError(f"A batch with the code {code} already exists.")

    batch = Batch(
        batch_code=code,
        product_id=product_id,
        expiry_date=expiry,
        quantity_received=quantity,
        quantity_remaining=quantity,
        received_at=utcnow(),
        received_by_id=actor.id if actor else None,
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    logger.info("Received %d x %s, expires %s", quantity, product.sku, expiry)
    return batch


def open_tasks(db: Session) -> List[RestockTask]:
    """Restock work still outstanding, most urgent first.

    Ordered by how empty the row is rather than by age: a shelf with nothing on
    it is losing sales now, while one just under its buffer is not.
    """
    tasks = (
        db.execute(
            select(RestockTask)
            .where(RestockTask.status.in_([RestockStatus.OPEN, RestockStatus.ASSIGNED]))
            .options(
                selectinload(RestockTask.row)
                .selectinload(ShelfRow.allocation)
                .selectinload(RowAllocation.product),
                selectinload(RestockTask.row).selectinload(ShelfRow.placements),
                selectinload(RestockTask.row).selectinload(ShelfRow.shelf),
                selectinload(RestockTask.assignee),
            )
        )
        .scalars()
        .unique()
        .all()
    )
    return sorted(tasks, key=lambda t: (t.row.on_shelf, -t.units_needed))


def assign_task(db: Session, *, task_id: int, assignee_id: int, actor: User) -> RestockTask:
    task = db.get(RestockTask, task_id)
    if task is None:
        raise StockError("That job does not exist.")
    if task.status in (RestockStatus.DONE, RestockStatus.CANCELLED):
        raise StockError("That job is already finished.")
    assignee = db.get(User, assignee_id)
    if assignee is None:
        raise StockError("That person is not on the staff list.")

    task.assigned_to_id = assignee_id
    task.assigned_by_id = actor.id
    task.status = RestockStatus.ASSIGNED
    db.commit()
    db.refresh(task)
    return task


def complete_task(db: Session, *, task_id: int, actor: User) -> RestockTask:
    """Mark a refill done.

    Does not itself move stock: the units were placed through `place_on_row`,
    which already closed the task if the row came back above its buffer. This
    is for the case where someone finished the job on the floor and is
    reporting it, so the queue reflects reality either way round.
    """
    task = db.get(RestockTask, task_id)
    if task is None:
        raise StockError("That job does not exist.")
    if task.status is RestockStatus.DONE:
        return task

    task.status = RestockStatus.DONE
    task.completed_by_id = actor.id
    task.completed_at = utcnow()
    db.commit()
    db.refresh(task)
    return task
