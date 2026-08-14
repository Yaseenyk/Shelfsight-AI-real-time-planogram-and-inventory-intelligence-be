"""Pydantic mirror of `data/schemas/planogram.schema.json`.

The JSON Schema file is the contract published with the paper; these models are
the runtime enforcement of that contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import ComplianceStatus
from app.schemas.common import BoundingBox, Detection, ORMModel


class PlanogramTolerances(BaseModel):
    iou_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    center_distance_threshold: float = Field(default=0.08, ge=0.0, le=1.5)
    row_band_tolerance: float = Field(default=0.05, ge=0.0, le=0.5)
    min_detection_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class PlanogramSlot(BaseModel):
    slot_id: str
    position: int = Field(..., ge=1, description="1-indexed left-to-right position in the row")
    sku: str
    expected_facings: int = Field(default=1, ge=1)
    bbox: BoundingBox
    orientation: Literal["front", "left", "right", "top"] = "front"
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_mandatory: bool = True


class PlanogramRow(BaseModel):
    row_id: str
    index: int = Field(..., ge=1, description="1-indexed top-to-bottom row within the shelf")
    slots: List[PlanogramSlot] = Field(..., min_length=1)


class PlanogramShelf(BaseModel):
    shelf_id: str
    level: int = Field(..., ge=1, description="1 = top shelf")
    y_range: List[float] = Field(..., min_length=2, max_length=2)
    rows: List[PlanogramRow] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _ordered_range(self) -> "PlanogramShelf":
        low, high = self.y_range
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("y_range must satisfy 0 <= low < high <= 1")
        return self


class PlanogramDocument(BaseModel):
    """The canonical multi-shelf / multi-row planogram matrix."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.0.0"
    planogram_id: str
    name: str
    version: str = "1.0.0"
    store_id: Optional[str] = None
    aisle: Optional[str] = None
    bay: Optional[str] = None
    units: Literal["normalized"] = "normalized"
    tolerances: PlanogramTolerances = Field(default_factory=PlanogramTolerances)
    shelves: List[PlanogramShelf] = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def slot_count(self) -> int:
        return sum(len(row.slots) for shelf in self.shelves for row in shelf.rows)

    def iter_slots(self):
        for shelf in self.shelves:
            for row in shelf.rows:
                for slot in row.slots:
                    yield shelf, row, slot

    @model_validator(mode="after")
    def _unique_slot_ids(self) -> "PlanogramDocument":
        seen: set[str] = set()
        for _, _, slot in self.iter_slots():
            if slot.slot_id in seen:
                raise ValueError(f"duplicate slot_id: {slot.slot_id}")
            seen.add(slot.slot_id)
        return self


class PlanogramCreate(BaseModel):
    document: PlanogramDocument
    is_active: bool = True


class PlanogramRead(ORMModel):
    id: int
    planogram_uid: str
    name: str
    version: str
    store_id: Optional[str]
    aisle: Optional[str]
    bay: Optional[str]
    shelf_count: int
    slot_count: int
    is_active: bool
    checksum: Optional[str]
    created_at: datetime
    updated_at: datetime


class PlanogramDetail(PlanogramRead):
    layout_json: Dict[str, Any]


class SlotResult(BaseModel):
    """Per-slot verdict produced by the spatial compliance engine."""

    slot_id: str
    shelf_id: str
    row_id: str
    expected_sku: str
    observed_sku: Optional[str] = None
    status: ComplianceStatus
    iou: float = 0.0
    center_distance: float = 1.0
    confidence: Optional[float] = None
    expected_bbox: Optional[BoundingBox] = None
    observed_bbox: Optional[BoundingBox] = None
    expected_facings: int = 1
    observed_facings: int = 0


class ComplianceCheckRequest(BaseModel):
    planogram_uid: Optional[str] = Field(
        default=None, description="Defaults to the active layout for the shelf"
    )
    shelf_id: Optional[str] = None
    detections: List[Detection] = Field(default_factory=list)
    tolerances: Optional[PlanogramTolerances] = None
    persist: bool = True


class ComplianceAuditRead(ORMModel):
    id: int
    session_id: Optional[int]
    planogram_id: int
    shelf_id: Optional[str]
    total_slots: int
    compliant_slots: int
    misplaced_slots: int
    missing_slots: int
    extra_detections: int
    compliance_score: float
    spatial_alignment_accuracy: float
    mean_iou: Optional[float]
    mean_center_distance: Optional[float]
    false_positive_rate: Optional[float]
    latency_ms: Optional[float]
    created_at: datetime


class ComplianceCheckResponse(ComplianceAuditRead):
    slot_results: List[SlotResult] = Field(default_factory=list)

    # A `persist=false` check is a preview: the audit is never written, so it has
    # no database id. Inheriting the strict `int` from ComplianceAuditRead made
    # that documented option fail with a 422 — the row it validated against did
    # not exist. The list endpoints keep the strict type.
    id: Optional[int] = Field(
        default=None, description="Null when the check was run with persist=false"
    )
    created_at: Optional[datetime] = None


class DetectionSummary(BaseModel):
    """Instrumentation for one detector pass, surfaced with every verification."""

    count: int = 0
    resolved_skus: int = Field(
        default=0, description="Detections mapped to a catalogue SKU"
    )
    unresolved: int = Field(
        default=0, description="Detected objects with no SKU — reported as EXTRA"
    )
    suppressed: int = Field(default=0, description="Boxes dropped by NMS / area filters")
    mean_confidence: Optional[float] = None
    class_counts: Dict[str, int] = Field(default_factory=dict)
    model_version: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    inference_ms: Optional[float] = None
    postprocess_ms: Optional[float] = None


class PlanogramVerifyResponse(ComplianceCheckResponse):
    """Response of `POST /planogram/verify` — the full Phase 1 pipeline output."""

    session_uid: str
    planogram_uid: str
    detections: List[Detection] = Field(default_factory=list)
    detection: DetectionSummary = Field(default_factory=DetectionSummary)
    detection_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
