"""Physical shelves, the rows on them, and what is actually sitting there.

The structure mirrors the real thing:

    Shelf ── Row ── Allocation (which product this row is sold to)
                 └─ Placement (the individual batches physically on it)

Two decisions worth stating.

**Capacity belongs to the allocation, not the row.** A row holds a different
number of units depending on what is in it -- fifty 10-rupee Maggi packs or ten
50-rupee ones -- so "how many fit" is a property of the pairing, set when the
manager allocates the row.

**Shelf quantity is derived, never stored.** What is on a row is the sum of its
placements, and every change is a StockMovement row. Storing a running total
invites it to drift from reality with no way to find out where; deriving it
means any disagreement can be traced to a specific event.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import MovementType, RestockStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.product import Product
    from app.models.user import User


class Shelf(Base, TimestampMixin):
    __tablename__ = "shelves"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    #: Free text: "Aisle 3, left side". Shop staff navigate by description, and
    #: no coordinate system survives contact with a real shop.
    location: Mapped[Optional[str]] = mapped_column(String(255))
    created_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))

    rows: Mapped[List["ShelfRow"]] = relationship(
        back_populates="shelf",
        cascade="all, delete-orphan",
        order_by="ShelfRow.position",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Shelf {self.code} rows={len(self.rows)}>"


class ShelfRow(Base, TimestampMixin):
    """One horizontal row. Position 1 is the top row, as a person reads it."""

    __tablename__ = "shelf_rows"
    __table_args__ = (UniqueConstraint("shelf_id", "position", name="uq_shelf_row_position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shelf_id: Mapped[int] = mapped_column(
        ForeignKey("shelves.id", ondelete="CASCADE"), index=True, nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(64))

    shelf: Mapped["Shelf"] = relationship(back_populates="rows")
    allocation: Mapped[Optional["RowAllocation"]] = relationship(
        back_populates="row", cascade="all, delete-orphan", uselist=False
    )
    placements: Mapped[List["Placement"]] = relationship(
        back_populates="row",
        cascade="all, delete-orphan",
        order_by="Placement.sequence",
    )

    @property
    def on_shelf(self) -> int:
        """Units physically present, derived from the placements."""
        return sum(placement.quantity for placement in self.placements)

    @property
    def needs_restock(self) -> bool:
        allocation = self.allocation
        return allocation is not None and self.on_shelf <= allocation.buffer_threshold

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ShelfRow {self.shelf_id}/{self.position}>"


class RowAllocation(Base, TimestampMixin):
    """Which product a row is given to, and on what terms.

    Brands pay for placement, so this is a commercial record as much as a
    physical one: `slotting_fee` is what the supplier paid for the row.
    """

    __tablename__ = "row_allocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    row_id: Mapped[int] = mapped_column(
        ForeignKey("shelf_rows.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True, nullable=False
    )

    #: How many units of *this* product fit in *this* row. Set per allocation
    #: because it depends on the pack size, not on the row alone.
    capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Refill when the count falls to this. The manager sets it per row: a fast
    #: seller needs a bigger cushion than a slow one.
    buffer_threshold: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    slotting_fee: Mapped[float] = mapped_column(default=0.0, nullable=False)

    allocated_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))

    row: Mapped["ShelfRow"] = relationship(back_populates="allocation")
    product: Mapped["Product"] = relationship()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RowAllocation row={self.row_id} product={self.product_id} cap={self.capacity}>"


class Batch(Base, TimestampMixin):
    """A delivery of one product with one expiry date.

    Expiry lives here rather than on the product because it is a property of the
    consignment: the same SKU arrives repeatedly with different dates, and
    front-of-shelf ordering depends on telling those consignments apart.
    """

    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True, nullable=False
    )
    expiry_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    quantity_received: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Left in the stockroom, i.e. not yet placed on any shelf.
    quantity_remaining: Mapped[int] = mapped_column(Integer, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))

    product: Mapped["Product"] = relationship()
    placements: Mapped[List["Placement"]] = relationship(back_populates="batch")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Batch {self.batch_code} exp={self.expiry_date} left={self.quantity_remaining}>"


class Placement(Base, TimestampMixin):
    """Units of one batch sitting on one row.

    `sequence` is the front-to-back order: 0 is the front of the shelf, the
    first thing a customer reaches. Ordering by soonest expiry is First Expired
    First Out -- the reason a shop does it is that the alternative is throwing
    stock away, not that it is clever.
    """

    __tablename__ = "placements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    row_id: Mapped[int] = mapped_column(
        ForeignKey("shelf_rows.id", ondelete="CASCADE"), index=True, nullable=False
    )
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), index=True, nullable=False
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    row: Mapped["ShelfRow"] = relationship(back_populates="placements")
    batch: Mapped["Batch"] = relationship(back_populates="placements")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Placement row={self.row_id} batch={self.batch_id} qty={self.quantity}>"


class StockMovement(Base, TimestampMixin):
    """Every change to what is on a shelf. Append-only; the audit trail."""

    __tablename__ = "stock_movements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    row_id: Mapped[Optional[int]] = mapped_column(ForeignKey("shelf_rows.id"), index=True)
    batch_id: Mapped[Optional[int]] = mapped_column(ForeignKey("batches.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True, nullable=False)
    movement_type: Mapped[MovementType] = mapped_column(
        SAEnum(MovementType, native_enum=False, length=16), nullable=False, index=True
    )
    #: Signed: positive puts units on the shelf, negative takes them off.
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    actor_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    note: Mapped[Optional[str]] = mapped_column(String(255))

    def __repr__(self) -> str:  # pragma: no cover
        return f"<StockMovement {self.movement_type.value} qty={self.quantity}>"


class RestockTask(Base, TimestampMixin):
    """Raised when a row drops to its buffer; cleared when someone refills it.

    A row of work rather than a notification, because "tell staff to refill
    this" needs to survive being missed: it stays open until confirmed, and the
    coordinator can see what has been sitting unassigned.
    """

    __tablename__ = "restock_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    row_id: Mapped[int] = mapped_column(
        ForeignKey("shelf_rows.id", ondelete="CASCADE"), index=True, nullable=False
    )
    status: Mapped[RestockStatus] = mapped_column(
        SAEnum(RestockStatus, native_enum=False, length=16),
        default=RestockStatus.OPEN,
        nullable=False,
        index=True,
    )
    #: Snapshot at the moment the threshold was crossed, so the task still makes
    #: sense after the row has changed.
    quantity_at_trigger: Mapped[int] = mapped_column(Integer, nullable=False)
    units_needed: Mapped[int] = mapped_column(Integer, nullable=False)

    assigned_to_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), index=True)
    assigned_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    completed_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    row: Mapped["ShelfRow"] = relationship()
    assignee: Mapped[Optional["User"]] = relationship(
        back_populates="assigned_tasks", foreign_keys=[assigned_to_id]
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RestockTask row={self.row_id} {self.status.value}>"
