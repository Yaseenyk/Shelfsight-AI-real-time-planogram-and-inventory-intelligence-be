"""Product master data — the join point between vision output and system stock."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.expiry import ExpiryAudit
    from app.models.freshness import FreshnessAudit
    from app.models.inventory import InventoryLog


class Product(Base, TimestampMixin):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sku: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    brand: Mapped[Optional[str]] = mapped_column(String(128))

    # Class id emitted by the YOLOv8 detector; nullable until a SKU is trained in.
    detection_class_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    detection_class_name: Mapped[Optional[str]] = mapped_column(String(128), index=True)

    #: EAN/UPC printed on the pack. The checkout scan resolves a product by
    #: this, so it is the join between a barcode reader and the catalogue.
    barcode: Mapped[Optional[str]] = mapped_column(String(32), unique=True, index=True)
    #: Typical units of this product that fit in one shelf row. A starting
    #: point the manager can override per allocation, since row depth varies.
    units_per_row: Mapped[int] = mapped_column(Integer, default=20, nullable=False)
    unit_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    system_stock: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reorder_threshold: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    is_perishable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    shelf_life_days: Mapped[Optional[int]] = mapped_column(Integer)

    inventory_logs: Mapped[List["InventoryLog"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    freshness_audits: Mapped[List["FreshnessAudit"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    expiry_audits: Mapped[List["ExpiryAudit"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Product {self.sku} stock={self.system_stock}>"
