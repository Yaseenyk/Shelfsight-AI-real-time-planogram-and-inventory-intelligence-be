"""Day-to-day operations: receiving stock, filling rows, selling, restocking.

Grouped in one router because they are one workflow -- a batch arrives, goes on
a shelf, sells, and raises the refill that starts it again -- and splitting them
across files would hide that.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_db
from app.api.v1.auth import require_user, requires
from app.models.enums import UserRole
from app.models.shelf import Batch, RowAllocation, ShelfRow
from app.models.user import User
from app.services import stock

router = APIRouter(tags=["operations"])


# ------------------------------------------------------------------ schemas --
class BatchRead(BaseModel):
    id: int
    batch_code: str
    product_id: int
    product_name: str
    sku: str
    expiry_date: date
    quantity_received: int
    quantity_remaining: int
    days_to_expiry: int


class BatchCreate(BaseModel):
    product_id: int
    quantity: int = Field(..., ge=1)
    expiry_date: date
    batch_code: Optional[str] = None


class PlaceRequest(BaseModel):
    row_id: int
    batch_id: int
    quantity: int = Field(..., ge=1)


class ScanRequest(BaseModel):
    barcode: str
    quantity: int = Field(default=1, ge=1)


class ScanResponse(BaseModel):
    product_name: str
    sku: str
    shelf_code: str
    row_position: int
    taken_from_batch: str
    batch_expiry: date
    remaining_on_row: int
    needs_restock: bool


class TaskRead(BaseModel):
    id: int
    status: str
    shelf_code: str
    shelf_name: str
    row_position: int
    product_name: str
    sku: str
    on_shelf: int
    capacity: int
    units_needed: int
    assigned_to: Optional[str] = None
    assigned_to_id: Optional[int] = None


class AssignRequest(BaseModel):
    assignee_id: int


class StaffRead(BaseModel):
    id: int
    name: str
    role: UserRole


def _batch_read(batch: Batch) -> BatchRead:
    return BatchRead(
        id=batch.id,
        batch_code=batch.batch_code,
        product_id=batch.product_id,
        product_name=batch.product.name,
        sku=batch.product.sku,
        expiry_date=batch.expiry_date,
        quantity_received=batch.quantity_received,
        quantity_remaining=batch.quantity_remaining,
        days_to_expiry=(batch.expiry_date - date.today()).days,
    )


def _task_read(task) -> TaskRead:  # noqa: ANN001 - ORM task
    row = task.row
    allocation = row.allocation
    return TaskRead(
        id=task.id,
        status=task.status.value,
        shelf_code=row.shelf.code,
        shelf_name=row.shelf.name,
        row_position=row.position,
        product_name=allocation.product.name if allocation else "—",
        sku=allocation.product.sku if allocation else "—",
        on_shelf=row.on_shelf,
        capacity=allocation.capacity if allocation else 0,
        units_needed=task.units_needed,
        assigned_to=task.assignee.name if task.assignee else None,
        assigned_to_id=task.assigned_to_id,
    )


# ------------------------------------------------------------------ batches --
@router.get("/batches", response_model=List[BatchRead])
def list_batches(
    only_available: bool = True,
    db: DbSession = Depends(get_db),
    _: User = Depends(require_user),
) -> List[BatchRead]:
    """Stockroom contents, soonest-expiring first — the order they should be used."""
    statement = select(Batch).options(selectinload(Batch.product)).order_by(Batch.expiry_date)
    if only_available:
        statement = statement.where(Batch.quantity_remaining > 0)
    return [_batch_read(batch) for batch in db.execute(statement).scalars().unique()]


@router.post("/batches", response_model=BatchRead, status_code=status.HTTP_201_CREATED)
def receive_batch(
    payload: BatchCreate,
    db: DbSession = Depends(get_db),
    actor: User = Depends(requires("batch:receive")),
) -> BatchRead:
    try:
        batch = stock.receive_batch(
            db,
            product_id=payload.product_id,
            quantity=payload.quantity,
            expiry=payload.expiry_date,
            batch_code=payload.batch_code,
            actor=actor,
        )
    except stock.StockError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    db.refresh(batch)
    return _batch_read(batch)


# -------------------------------------------------------------- filling rows --
@router.post("/stock/place")
def place_stock(
    payload: PlaceRequest,
    db: DbSession = Depends(get_db),
    actor: User = Depends(requires("stock:place")),
) -> dict:
    try:
        row = stock.place_on_row(
            db,
            row_id=payload.row_id,
            batch_id=payload.batch_id,
            quantity=payload.quantity,
            actor=actor,
        )
    except stock.StockError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return {
        "row_id": row.id,
        "on_shelf": row.on_shelf,
        "capacity": row.allocation.capacity if row.allocation else 0,
        "needs_restock": row.needs_restock,
        # Front-to-back, so the caller can show that FEFO actually reordered it.
        "front_to_back": [
            {
                "batch_code": placement.batch.batch_code,
                "expiry_date": placement.batch.expiry_date.isoformat(),
                "quantity": placement.quantity,
            }
            for placement in sorted(row.placements, key=lambda p: p.sequence)
        ],
    }


# --------------------------------------------------------------------- sales --
@router.post("/sales/scan", response_model=ScanResponse)
def scan_sale(
    payload: ScanRequest,
    db: DbSession = Depends(get_db),
    actor: User = Depends(requires("sale:scan")),
) -> ScanResponse:
    try:
        result = stock.sell_by_barcode(
            db, barcode=payload.barcode, quantity=payload.quantity, actor=actor
        )
    except stock.StockError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    row = result["row"]
    first = result["batches"][0]
    return ScanResponse(
        product_name=result["product"].name,
        sku=result["product"].sku,
        shelf_code=row.shelf.code,
        row_position=row.position,
        taken_from_batch=first.batch_code,
        batch_expiry=first.expiry_date,
        remaining_on_row=result["remaining_on_row"],
        needs_restock=result["needs_restock"],
    )


# ------------------------------------------------------------------ restock --
@router.get("/restock/tasks", response_model=List[TaskRead])
def list_tasks(
    db: DbSession = Depends(get_db),
    _: User = Depends(require_user),
) -> List[TaskRead]:
    return [_task_read(task) for task in stock.open_tasks(db)]


@router.get("/restock/staff", response_model=List[StaffRead])
def list_staff(
    db: DbSession = Depends(get_db),
    _: User = Depends(requires("restock:assign")),
) -> List[StaffRead]:
    """Who a job can be given to. Any role can refill a shelf."""
    users = db.execute(select(User).where(User.is_active.is_(True))).scalars().all()
    return [StaffRead(id=u.id, name=u.name, role=u.role) for u in users]


@router.post("/restock/tasks/{task_id}/assign", response_model=TaskRead)
def assign_task(
    task_id: int,
    payload: AssignRequest,
    db: DbSession = Depends(get_db),
    actor: User = Depends(requires("restock:assign")),
) -> TaskRead:
    try:
        stock.assign_task(db, task_id=task_id, assignee_id=payload.assignee_id, actor=actor)
    except stock.StockError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _task_read(next(t for t in stock.open_tasks(db) if t.id == task_id))


@router.post("/restock/tasks/{task_id}/complete")
def complete_task(
    task_id: int,
    db: DbSession = Depends(get_db),
    actor: User = Depends(requires("restock:complete")),
) -> dict:
    try:
        task = stock.complete_task(db, task_id=task_id, actor=actor)
    except stock.StockError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return {"id": task.id, "status": task.status.value}


# ---------------------------------------------------------------- row lookup --
@router.get("/rows/{row_id}")
def read_row(
    row_id: int,
    db: DbSession = Depends(get_db),
    _: User = Depends(require_user),
) -> dict:
    """One row with what is on it, for the placement screen."""
    row = db.execute(
        select(ShelfRow)
        .where(ShelfRow.id == row_id)
        .options(
            selectinload(ShelfRow.allocation).selectinload(RowAllocation.product),
            selectinload(ShelfRow.placements),
            selectinload(ShelfRow.shelf),
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That row does not exist.")
    return {
        "id": row.id,
        "position": row.position,
        "shelf_code": row.shelf.code,
        "on_shelf": row.on_shelf,
        "capacity": row.allocation.capacity if row.allocation else 0,
        "product_id": row.allocation.product_id if row.allocation else None,
        "product_name": row.allocation.product.name if row.allocation else None,
    }
