"""A ScanSession is one camera frame / uploaded image pushed through the pipeline.

Every audit row (inventory, compliance, freshness, expiry) hangs off a session so
that a single capture can be reproduced end-to-end during evaluation.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import ScanStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.compliance import ComplianceAudit
    from app.models.expiry import ExpiryAudit
    from app.models.freshness import FreshnessAudit
    from app.models.inventory import InventoryLog


class ScanSession(Base, TimestampMixin):
    __tablename__ = "scan_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_uid: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)

    shelf_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    store_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    image_path: Mapped[Optional[str]] = mapped_column(String(512))
    image_width: Mapped[Optional[int]] = mapped_column(Integer)
    image_height: Mapped[Optional[int]] = mapped_column(Integer)

    status: Mapped[ScanStatus] = mapped_column(
        SAEnum(ScanStatus, native_enum=False, length=16),
        default=ScanStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(String(1024))

    # Reproducibility + latency instrumentation for the paper.
    detector_version: Mapped[Optional[str]] = mapped_column(String(128))
    detection_latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    total_latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    detections: Mapped[Optional[dict]] = mapped_column(JSON)

    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    inventory_logs: Mapped[List["InventoryLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    compliance_audits: Mapped[List["ComplianceAudit"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    freshness_audits: Mapped[List["FreshnessAudit"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    expiry_audits: Mapped[List["ExpiryAudit"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
