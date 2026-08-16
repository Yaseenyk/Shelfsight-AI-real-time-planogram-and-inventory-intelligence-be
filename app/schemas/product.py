from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class ProductBase(BaseModel):
    sku: str = Field(..., max_length=64)
    name: str = Field(..., max_length=255)
    category: Optional[str] = None
    brand: Optional[str] = None
    detection_class_id: Optional[int] = None
    detection_class_name: Optional[str] = None
    unit_price: float = Field(default=0.0, ge=0.0)
    system_stock: int = Field(default=0, ge=0)
    reorder_threshold: int = Field(default=0, ge=0)
    is_perishable: bool = False
    shelf_life_days: Optional[int] = Field(default=None, ge=0)
    #: EAN/UPC, used by the checkout scan to resolve a product.
    barcode: Optional[str] = None
    #: Typical units of this product that fit in one shelf row. Sent to the
    #: dashboard so the allocation form can pre-fill a capacity instead of
    #: asking the manager to invent a number.
    units_per_row: int = Field(default=20, ge=1)


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    detection_class_id: Optional[int] = None
    detection_class_name: Optional[str] = None
    unit_price: Optional[float] = Field(default=None, ge=0.0)
    system_stock: Optional[int] = Field(default=None, ge=0)
    reorder_threshold: Optional[int] = Field(default=None, ge=0)
    is_perishable: Optional[bool] = None
    shelf_life_days: Optional[int] = Field(default=None, ge=0)
    barcode: Optional[str] = None
    units_per_row: Optional[int] = Field(default=None, ge=1)


class ProductRead(ORMModel, ProductBase):
    id: int
    created_at: datetime
    updated_at: datetime
