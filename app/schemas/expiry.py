from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.enums import ExpiryStatus
from app.schemas.common import BoundingBox, ORMModel


class ExpiryExtraction(BaseModel):
    raw_text: Optional[str] = None
    normalized_text: Optional[str] = None
    matched_pattern: Optional[str] = None
    parsed_date: Optional[date] = None
    days_remaining: Optional[int] = None
    status: ExpiryStatus = ExpiryStatus.UNREADABLE
    ocr_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = None
    latency_ms: Optional[float] = None


class ExpiryExtractRequest(BaseModel):
    """Text-only path: feed OCR output (or ground-truth strings) to the parser."""

    texts: List[str] = Field(default_factory=list)
    product_sku: Optional[str] = None
    session_uid: Optional[str] = None
    reference_date: Optional[date] = None
    persist: bool = False


class ExpiryExtractResponse(BaseModel):
    session_uid: Optional[str] = None
    extractions: List[ExpiryExtraction]
    expired_count: int = 0
    near_expiry_count: int = 0
    unreadable_count: int = 0
    latency_ms: Optional[float] = None

    #: The single read a store manager should act on — the most decisive dated
    #: candidate (expired > near-expiry > valid), not merely the most confident.
    best: Optional[ExpiryExtraction] = None
    raw_text: Optional[str] = Field(
        default=None, description="Everything OCR read, newline-joined, for audit"
    )
    variant_used: Optional[str] = Field(
        default=None, description="Preprocessing variant that produced the winning date"
    )
    variants_tried: List[str] = Field(default_factory=list)
    ocr_ms: Optional[float] = None


class ExpiryAuditRead(ORMModel):
    id: int
    session_id: Optional[int]
    product_id: Optional[int]
    raw_text: Optional[str]
    normalized_text: Optional[str]
    matched_pattern: Optional[str]
    parsed_date: Optional[date]
    days_remaining: Optional[int]
    status: ExpiryStatus
    ocr_confidence: Optional[float]
    latency_ms: Optional[float]
    created_at: datetime


class ExpirySummary(BaseModel):
    total_scanned: int
    valid: int
    near_expiry: int
    expired: int
    unreadable: int
    read_rate: float = Field(..., ge=0.0, le=1.0)
