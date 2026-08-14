"""Versioned planogram layouts.

`layout_json` holds a document validated against
`data/schemas/planogram.schema.json`. Keeping it as a JSON blob (rather than
normalising slots into rows) preserves the exact artefact used for a given
experiment run, which is what the paper needs to cite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.compliance import ComplianceAudit


class PlanogramLayout(Base, TimestampMixin):
    __tablename__ = "planogram_layouts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    planogram_uid: Mapped[str] = mapped_column(
        String(128), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(32), default="1.0.0", nullable=False)

    store_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    aisle: Mapped[Optional[str]] = mapped_column(String(32))
    bay: Mapped[Optional[str]] = mapped_column(String(32))

    shelf_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    slot_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Per-layout tolerance overrides; fall back to settings when null.
    iou_threshold: Mapped[Optional[float]] = mapped_column(Float)
    center_distance_threshold: Mapped[Optional[float]] = mapped_column(Float)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    checksum: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    layout_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    audits: Mapped[List["ComplianceAudit"]] = relationship(
        back_populates="planogram", cascade="all, delete-orphan"
    )
