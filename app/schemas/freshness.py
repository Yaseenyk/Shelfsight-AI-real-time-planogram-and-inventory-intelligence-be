from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.enums import FreshnessLabel
from app.schemas.common import BoundingBox, ORMModel


class FreshnessPrediction(BaseModel):
    label: FreshnessLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    bbox: Optional[BoundingBox] = None
    backbone: Optional[str] = None
    latency_ms: Optional[float] = None


class FreshnessClassifyRequest(BaseModel):
    """Used when crops are already on disk (batch/eval path)."""

    image_paths: List[str] = Field(default_factory=list)
    product_sku: Optional[str] = None
    session_uid: Optional[str] = None
    persist: bool = True


class FreshnessClassifyResponse(BaseModel):
    session_uid: Optional[str] = None
    predictions: List[FreshnessPrediction]
    spoiled_count: int = 0
    ripening_count: int = 0
    latency_ms: Optional[float] = None


class FreshnessAuditRead(ORMModel):
    id: int
    session_id: Optional[int]
    product_id: Optional[int]
    label: FreshnessLabel
    confidence: float
    class_probabilities: Optional[Dict[str, float]]
    bbox: Optional[List[float]]
    backbone: Optional[str]
    model_version: Optional[str]
    latency_ms: Optional[float]
    ground_truth_label: Optional[FreshnessLabel]
    created_at: datetime


class FreshnessSummary(BaseModel):
    total_assessed: int
    fresh: int
    ripening: int
    spoiled: int
    spoilage_rate: float = Field(..., ge=0.0, le=1.0)
    mean_confidence: Optional[float] = None
