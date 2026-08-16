"""Creating shelves, laying out their rows, and allocating rows to products.

The rules that matter live here rather than in the API layer, so they hold
whoever calls them:

* a row holds exactly one product, because that is what a shelf row is;
* capacity is a property of the row-and-product pairing, not of either alone;
* a row cannot be re-allocated while stock is still sitting on it, because the
  units would silently become units of something else.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.logging import get_logger
from app.models.product import Product
from app.models.shelf import RowAllocation, Shelf, ShelfRow
from app.models.user import User

logger = get_logger(__name__)

MAX_ROWS = 12
MAX_CAPACITY = 999


class ShelfError(Exception):
    """A rule was broken. The message is written to be shown to the user."""


def list_shelves(db: Session) -> List[Shelf]:
    return list(
        db.execute(
            select(Shelf)
            .options(
                selectinload(Shelf.rows)
                .selectinload(ShelfRow.allocation)
                .selectinload(RowAllocation.product),
                selectinload(Shelf.rows).selectinload(ShelfRow.placements),
            )
            .order_by(Shelf.code)
        )
        .scalars()
        .unique()
    )


def get_shelf(db: Session, shelf_id: int) -> Shelf:
    shelf = db.execute(
        select(Shelf)
        .where(Shelf.id == shelf_id)
        .options(
            selectinload(Shelf.rows)
            .selectinload(ShelfRow.allocation)
            .selectinload(RowAllocation.product),
            selectinload(Shelf.rows).selectinload(ShelfRow.placements),
        )
    ).scalar_one_or_none()
    if shelf is None:
        raise ShelfError("That shelf does not exist.")
    return shelf


def create_shelf(
    db: Session,
    *,
    code: str,
    name: str,
    row_count: int,
    location: Optional[str] = None,
    actor: Optional[User] = None,
) -> Shelf:
    """Create a shelf and its empty rows in one step.

    Rows are created up front rather than added one at a time: a physical shelf
    arrives with a fixed number of shelves in it, and asking the manager to
    press "add row" five times models the software rather than the object.
    """
    code = code.strip().upper()
    if not code:
        raise ShelfError("Give the shelf a short code, like AISLE3-BAY2.")
    if not 1 <= row_count <= MAX_ROWS:
        raise ShelfError(f"A shelf can have between 1 and {MAX_ROWS} rows.")

    exists = db.execute(
        select(func.count()).select_from(Shelf).where(Shelf.code == code)
    ).scalar_one()
    if exists:
        raise ShelfError(f"A shelf with the code {code} already exists.")

    shelf = Shelf(
        code=code,
        name=name.strip() or code,
        location=location.strip() if location else None,
        created_by_id=actor.id if actor else None,
    )
    # Position 1 is the top row, as a person reads a shelf.
    shelf.rows = [
        ShelfRow(position=index, label=f"Row {index}") for index in range(1, row_count + 1)
    ]
    db.add(shelf)
    db.commit()
    db.refresh(shelf)
    logger.info("Shelf %s created with %d rows", shelf.code, row_count)
    return shelf


def suggest_capacity(product: Product) -> int:
    """How many of this product a manager should expect to fit in one row.

    Only a starting point, shown in the allocation form so the number is not
    invented from nothing. It comes from the product because pack size is what
    drives it -- fifty small packs or ten large ones in the same physical row.
    """
    return max(1, product.units_per_row or 1)


def allocate_row(
    db: Session,
    *,
    row_id: int,
    product_id: int,
    capacity: Optional[int] = None,
    buffer_threshold: Optional[int] = None,
    slotting_fee: float = 0.0,
    actor: Optional[User] = None,
) -> RowAllocation:
    """Give a row to a product, or change what it is given to."""
    row = db.execute(
        select(ShelfRow)
        .where(ShelfRow.id == row_id)
        .options(selectinload(ShelfRow.placements), selectinload(ShelfRow.allocation))
    ).scalar_one_or_none()
    if row is None:
        raise ShelfError("That row does not exist.")

    product = db.get(Product, product_id)
    if product is None:
        raise ShelfError("That product is not in the catalogue.")

    # Re-allocating an occupied row would quietly turn units of one product into
    # units of another. Make the manager clear it first.
    if row.allocation and row.allocation.product_id != product_id and row.on_shelf > 0:
        raise ShelfError(
            f"This row still holds {row.on_shelf} unit(s). Clear the row before giving it to a different product."
        )

    resolved_capacity = capacity if capacity is not None else suggest_capacity(product)
    if not 1 <= resolved_capacity <= MAX_CAPACITY:
        raise ShelfError(f"Capacity must be between 1 and {MAX_CAPACITY}.")

    resolved_buffer = (
        buffer_threshold if buffer_threshold is not None else max(1, resolved_capacity // 5)
    )
    if resolved_buffer < 0:
        raise ShelfError("The refill point cannot be negative.")
    if resolved_buffer >= resolved_capacity:
        # A threshold at or above capacity means the row is "low" the moment it
        # is filled, so it would raise a restock task that can never be cleared.
        raise ShelfError(
            f"The refill point ({resolved_buffer}) must be below the capacity ({resolved_capacity}), "
            "otherwise the row is never considered full."
        )

    if row.allocation:
        allocation = row.allocation
        allocation.product_id = product_id
        allocation.capacity = resolved_capacity
        allocation.buffer_threshold = resolved_buffer
        allocation.slotting_fee = slotting_fee
        allocation.allocated_by_id = actor.id if actor else None
    else:
        allocation = RowAllocation(
            row_id=row.id,
            product_id=product_id,
            capacity=resolved_capacity,
            buffer_threshold=resolved_buffer,
            slotting_fee=slotting_fee,
            allocated_by_id=actor.id if actor else None,
        )
        db.add(allocation)

    db.commit()
    db.refresh(allocation)
    logger.info(
        "Row %s allocated to %s (capacity %d, refill at %d)",
        row_id,
        product.sku,
        resolved_capacity,
        resolved_buffer,
    )
    return allocation


def clear_allocation(db: Session, row_id: int) -> None:
    row = db.execute(
        select(ShelfRow)
        .where(ShelfRow.id == row_id)
        .options(selectinload(ShelfRow.placements), selectinload(ShelfRow.allocation))
    ).scalar_one_or_none()
    if row is None:
        raise ShelfError("That row does not exist.")
    if row.on_shelf > 0:
        raise ShelfError(f"This row still holds {row.on_shelf} unit(s). Take them off first.")
    if row.allocation:
        db.delete(row.allocation)
        db.commit()


def delete_shelf(db: Session, shelf_id: int) -> None:
    shelf = get_shelf(db, shelf_id)
    occupied = sum(row.on_shelf for row in shelf.rows)
    if occupied:
        raise ShelfError(f"This shelf still holds {occupied} unit(s). Empty it first.")
    db.delete(shelf)
    db.commit()
