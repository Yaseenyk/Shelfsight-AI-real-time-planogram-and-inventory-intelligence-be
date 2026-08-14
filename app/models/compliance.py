"""Planogram compliance audit results (shelf-level roll-up + per-slot detail)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.planogram import PlanogramLayout
    from app.models.scan import ScanSession


class ComplianceAudit(Base, TimestampMixin):
    __tablename__ = "compliance_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("scan_sessions.id", ondelete="CASCADE"), index=True
    )
    planogram_id: Mapped[int] = mapped_column(
        ForeignKey("planogram_layouts.id", ondelete="CASCADE"), index=True, nullable=False
    )
    shelf_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)

    total_slots: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    compliant_slots: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    misplaced_slots: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    missing_slots: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    extra_detections: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Publication metrics (see evaluation/metrics/compliance.py).
    compliance_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    spatial_alignment_accuracy: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mean_iou: Mapped[Optional[float]] = mapped_column(Float)
    mean_center_distance: Mapped[Optional[float]] = mapped_column(Float)
    false_positive_rate: Mapped[Optional[float]] = mapped_column(Float)

    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    slot_results: Mapped[Optional[list]] = mapped_column(JSON)

    planogram: Mapped["PlanogramLayout"] = relationship(back_populates="audits")
    session: Mapped[Optional["ScanSession"]] = relationship(back_populates="compliance_audits")
