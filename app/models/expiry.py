"""OCR expiry-date extraction results and derived validity status."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Date, Float, ForeignKey, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import ExpiryStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.product import Product
    from app.models.scan import ScanSession


class ExpiryAudit(Base, TimestampMixin):
    __tablename__ = "expiry_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("scan_sessions.id", ondelete="CASCADE"), index=True
    )
    product_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True
    )

    raw_text: Mapped[Optional[str]] = mapped_column(String(512))
    normalized_text: Mapped[Optional[str]] = mapped_column(String(256))
    matched_pattern: Mapped[Optional[str]] = mapped_column(String(64))

    parsed_date: Mapped[Optional[date]] = mapped_column(Date, index=True)
    days_remaining: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    status: Mapped[ExpiryStatus] = mapped_column(
        SAEnum(ExpiryStatus, native_enum=False, length=16),
        default=ExpiryStatus.UNREADABLE,
        nullable=False,
        index=True,
    )

    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float)
    bbox: Mapped[Optional[list]] = mapped_column(JSON)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)

    # Ground truth for CER/WER + date-parsing precision.
    ground_truth_text: Mapped[Optional[str]] = mapped_column(String(512))
    ground_truth_date: Mapped[Optional[date]] = mapped_column(Date)

    product: Mapped[Optional["Product"]] = relationship(back_populates="expiry_audits")
    session: Mapped[Optional["ScanSession"]] = relationship(back_populates="expiry_audits")
