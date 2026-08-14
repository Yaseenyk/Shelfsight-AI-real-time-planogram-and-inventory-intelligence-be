from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.enums import DiscrepancyType, Severity
from app.schemas.common import Detection, ORMModel


class InventoryScanRequest(BaseModel):
    """Body for a detection-driven stock reconciliation.

    `detections` may be supplied directly (offline replay / evaluation) or left
    empty when an image is uploaded to the multipart endpoint.
    """

    shelf_id: Optional[str] = None
    store_id: Optional[str] = None
    detections: List[Detection] = Field(default_factory=list)
    persist: bool = True


class InventoryLogRead(ORMModel):
    id: int
    session_id: Optional[int]
    product_id: int
    detected_count: int
    system_count: int
    discrepancy: int
    discrepancy_type: DiscrepancyType
    severity: Severity
    mean_confidence: Optional[float]
    shelf_id: Optional[str]
    notes: Optional[str]
    created_at: datetime


class DiscrepancyItem(BaseModel):
    sku: str
    product_name: str
    detected_count: int
    system_count: int
    discrepancy: int
    discrepancy_type: DiscrepancyType
    severity: Severity
    estimated_value_impact: float = 0.0


class InventoryScanResponse(BaseModel):
    session_uid: str
    shelf_id: Optional[str] = None
    #: Products the detector localised in the frame, before SKU resolution.
    #: With a single-class detector this is large while total_detected is 0:
    #: the shelf is full, but nothing can be attributed to a catalogue entry.
    #: Reporting only the resolved figure describes a full shelf as empty.
    objects_detected: int = 0
    unresolved_detections: int = 0
    total_detected: int
    total_system: int
    matched_skus: int
    discrepancies: List[DiscrepancyItem]
    phantom_count: int
    latency_ms: Optional[float] = None
    created_at: datetime


class InventorySummary(BaseModel):
    """Powers the dashboard overview tiles."""

    total_products: int
    total_system_stock: int
    total_detected_stock: int
    phantom_skus: int
    undercount_skus: int
    overcount_skus: int
    accuracy_rate: float = Field(..., ge=0.0, le=1.0)
    value_at_risk: float = 0.0
    last_scan_at: Optional[datetime] = None
