"""Primitives reused across every router."""

from __future__ import annotations

from datetime import datetime
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base for response models read straight off SQLAlchemy instances."""

    model_config = ConfigDict(from_attributes=True)


class BoundingBox(BaseModel):
    """Axis-aligned box in **normalised** image coordinates (0..1, xyxy).

    Normalised units keep planogram definitions resolution-independent, which is
    what makes the same JSON matrix reusable across cameras.
    """

    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "BoundingBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox requires x2 > x1 and y2 > y1")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_xyxy(cls, values: "list[float] | tuple[float, float, float, float]") -> "BoundingBox":
        x1, y1, x2, y2 = values
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)


class Detection(BaseModel):
    """One YOLOv8 detection, normalised and enriched with catalogue metadata."""

    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    sku: Optional[str] = None
    track_id: Optional[int] = None


class PageMeta(BaseModel):
    total: int
    limit: int
    offset: int


class Page(BaseModel, Generic[T]):
    items: List[T]
    meta: PageMeta


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    timestamp: datetime
    database: str
    detector_loaded: bool
    detector_model: Optional[str] = None
    detector_classes: Optional[int] = None
    detector_error: Optional[str] = None
    ollama_reachable: Optional[bool] = None


class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None


class LatencyBreakdown(BaseModel):
    """Per-stage timings, persisted so the paper can report inference latency."""

    detection_ms: Optional[float] = None
    compliance_ms: Optional[float] = None
    freshness_ms: Optional[float] = None
    ocr_ms: Optional[float] = None
    total_ms: Optional[float] = None
