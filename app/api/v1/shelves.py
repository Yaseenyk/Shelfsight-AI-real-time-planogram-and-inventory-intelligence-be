"""Shelf design: create a shelf, lay out its rows, allocate rows to products.

Every write is manager-only, enforced by the shared permission table rather than
a role comparison at the call site, so widening a rule is one edit.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DbSession

from app.api.deps import get_db
from app.api.v1.auth import require_user, requires
from app.models.user import User
from app.services import shelf_manager

router = APIRouter(prefix="/shelves", tags=["shelves"])


# ------------------------------------------------------------------ schemas --
class AllocationRead(BaseModel):
    product_id: int
    sku: str
    product_name: str
    capacity: int
    buffer_threshold: int
    slotting_fee: float

    model_config = {"from_attributes": True}


class RowRead(BaseModel):
    id: int
    position: int
    label: Optional[str] = None
    on_shelf: int
    needs_restock: bool
    allocation: Optional[AllocationRead] = None


class ShelfRead(BaseModel):
    id: int
    code: str
    name: str
    location: Optional[str] = None
    rows: List[RowRead]

    @property
    def row_count(self) -> int:
        return len(self.rows)


class ShelfCreate(BaseModel):
    code: str = Field(..., min_length=1, max_length=32)
    name: str = Field(..., min_length=1, max_length=160)
    row_count: int = Field(..., ge=1, le=shelf_manager.MAX_ROWS)
    location: Optional[str] = Field(default=None, max_length=255)


class AllocationWrite(BaseModel):
    product_id: int
    #: Omitted means "use the product's own per-row hint", so the manager is
    #: never asked to invent a number from nothing.
    capacity: Optional[int] = Field(default=None, ge=1, le=shelf_manager.MAX_CAPACITY)
    buffer_threshold: Optional[int] = Field(default=None, ge=0)
    slotting_fee: float = 0.0


def _to_row(row) -> RowRead:  # noqa: ANN001 - ORM row
    allocation = None
    if row.allocation is not None:
        allocation = AllocationRead(
            product_id=row.allocation.product_id,
            sku=row.allocation.product.sku,
            product_name=row.allocation.product.name,
            capacity=row.allocation.capacity,
            buffer_threshold=row.allocation.buffer_threshold,
            slotting_fee=row.allocation.slotting_fee,
        )
    return RowRead(
        id=row.id,
        position=row.position,
        label=row.label,
        on_shelf=row.on_shelf,
        needs_restock=row.needs_restock,
        allocation=allocation,
    )


def _to_shelf(shelf) -> ShelfRead:  # noqa: ANN001 - ORM shelf
    return ShelfRead(
        id=shelf.id,
        code=shelf.code,
        name=shelf.name,
        location=shelf.location,
        rows=[_to_row(row) for row in shelf.rows],
    )


# ------------------------------------------------------------------- routes --
@router.get("", response_model=List[ShelfRead])
def list_shelves(
    db: DbSession = Depends(get_db),
    _: User = Depends(require_user),
) -> List[ShelfRead]:
    return [_to_shelf(shelf) for shelf in shelf_manager.list_shelves(db)]


@router.get("/{shelf_id}", response_model=ShelfRead)
def read_shelf(
    shelf_id: int,
    db: DbSession = Depends(get_db),
    _: User = Depends(require_user),
) -> ShelfRead:
    try:
        return _to_shelf(shelf_manager.get_shelf(db, shelf_id))
    except shelf_manager.ShelfError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("", response_model=ShelfRead, status_code=status.HTTP_201_CREATED)
def create_shelf(
    payload: ShelfCreate,
    db: DbSession = Depends(get_db),
    actor: User = Depends(requires("shelf:create")),
) -> ShelfRead:
    try:
        shelf = shelf_manager.create_shelf(
            db,
            code=payload.code,
            name=payload.name,
            row_count=payload.row_count,
            location=payload.location,
            actor=actor,
        )
    except shelf_manager.ShelfError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _to_shelf(shelf)


@router.put("/rows/{row_id}/allocation", response_model=ShelfRead)
def allocate_row(
    row_id: int,
    payload: AllocationWrite,
    db: DbSession = Depends(get_db),
    actor: User = Depends(requires("shelf:allocate")),
) -> ShelfRead:
    try:
        allocation = shelf_manager.allocate_row(
            db,
            row_id=row_id,
            product_id=payload.product_id,
            capacity=payload.capacity,
            buffer_threshold=payload.buffer_threshold,
            slotting_fee=payload.slotting_fee,
            actor=actor,
        )
        shelf = shelf_manager.get_shelf(db, allocation.row.shelf_id)
    except shelf_manager.ShelfError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _to_shelf(shelf)


@router.delete("/rows/{row_id}/allocation", status_code=status.HTTP_204_NO_CONTENT)
def clear_allocation(
    row_id: int,
    db: DbSession = Depends(get_db),
    _: User = Depends(requires("shelf:allocate")),
) -> None:
    try:
        shelf_manager.clear_allocation(db, row_id)
    except shelf_manager.ShelfError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.delete("/{shelf_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_shelf(
    shelf_id: int,
    db: DbSession = Depends(get_db),
    _: User = Depends(requires("shelf:create")),
) -> None:
    try:
        shelf_manager.delete_shelf(db, shelf_id)
    except shelf_manager.ShelfError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
