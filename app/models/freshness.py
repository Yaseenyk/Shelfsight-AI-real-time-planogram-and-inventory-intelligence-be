"""Per-crop perishable freshness classification results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Float, ForeignKey, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import FreshnessLabel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.product import Product
    from app.models.scan import ScanSession


class FreshnessAudit(Base, TimestampMixin):
    __tablename__ = "freshness_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("scan_sessions.id", ondelete="CASCADE"), index=True
    )
    product_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True
    )

    label: Mapped[FreshnessLabel] = mapped_column(
        SAEnum(FreshnessLabel, native_enum=False, length=16), nullable=False, index=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    class_probabilities: Mapped[Optional[dict]] = mapped_column(JSON)

    # Normalised xyxy of the crop this prediction came from.
    bbox: Mapped[Optional[list]] = mapped_column(JSON)
    crop_path: Mapped[Optional[str]] = mapped_column(String(512))

    backbone: Mapped[Optional[str]] = mapped_column(String(64))
    model_version: Mapped[Optional[str]] = mapped_column(String(128))
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)

    # Populated only for labelled evaluation frames; drives the confusion matrix.
    ground_truth_label: Mapped[Optional[FreshnessLabel]] = mapped_column(
        SAEnum(FreshnessLabel, native_enum=False, length=16)
    )

    product: Mapped[Optional["Product"]] = relationship(back_populates="freshness_audits")
    session: Mapped[Optional["ScanSession"]] = relationship(back_populates="freshness_audits")
