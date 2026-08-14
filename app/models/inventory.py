"""Phantom-inventory ledger: detected facings vs. system stock, per capture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import DiscrepancyType, Severity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.product import Product
    from app.models.scan import ScanSession


class InventoryLog(Base, TimestampMixin):
    __tablename__ = "inventory_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("scan_sessions.id", ondelete="CASCADE"), index=True
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True, nullable=False
    )

    detected_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    system_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Stored (not computed) so historical rows stay stable if the rule changes.
    discrepancy: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    discrepancy_type: Mapped[DiscrepancyType] = mapped_column(
        SAEnum(DiscrepancyType, native_enum=False, length=16),
        default=DiscrepancyType.MATCH,
        nullable=False,
        index=True,
    )
    severity: Mapped[Severity] = mapped_column(
        SAEnum(Severity, native_enum=False, length=16),
        default=Severity.INFO,
        nullable=False,
        index=True,
    )

    mean_confidence: Mapped[Optional[float]] = mapped_column(Float)
    shelf_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    notes: Mapped[Optional[str]] = mapped_column(String(512))

    product: Mapped["Product"] = relationship(back_populates="inventory_logs")
    session: Mapped[Optional["ScanSession"]] = relationship(back_populates="inventory_logs")

    @staticmethod
    def classify(detected: int, system: int) -> tuple[int, DiscrepancyType, Severity]:
        """Single source of truth for the discrepancy rule (mirrored in the docs)."""
        delta = detected - system
        if delta == 0:
            return delta, DiscrepancyType.MATCH, Severity.INFO
        if detected == 0 and system > 0:
            return delta, DiscrepancyType.PHANTOM, Severity.CRITICAL
        if delta < 0:
            severity = Severity.WARNING if abs(delta) < max(1, system // 2) else Severity.CRITICAL
            return delta, DiscrepancyType.UNDERCOUNT, severity
        return delta, DiscrepancyType.OVERCOUNT, Severity.WARNING
