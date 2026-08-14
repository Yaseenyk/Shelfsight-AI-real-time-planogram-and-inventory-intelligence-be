"""Insight contracts.

Three layers, deliberately separate:

1. `InsightContext` — the telemetry we compile *into* the prompt. Pydantic so the
   exact payload sent to the model is typed, serialisable and auditable.
2. `LLMInsightPayload` — the raw JSON we expect *back*. Strictly validated: a
   local 3B model will happily return prose, wrong types or half the fields, and
   an unvalidated dict reaching the dashboard is how a briefing quietly acquires
   invented numbers.
3. `InsightResponse` — what the API returns, carrying provenance (which model
   actually ran, whether it was substituted, whether we fell back).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import Severity

Audience = Literal["store_manager", "regional_director", "analyst"]

#: The store manager gets three things to do. More is a to-do list nobody reads.
MAX_ACTIONS: int = 3


# ------------------------------------------------------------------ context --
class InventoryMetrics(BaseModel):
    total_skus: int = 0
    phantom_skus: int = 0
    undercount_skus: int = 0
    overcount_skus: int = 0
    total_detected: int = 0
    total_system: int = 0
    accuracy_rate: float = 0.0
    value_at_risk: float = 0.0


class ComplianceMetrics(BaseModel):
    total_slots: int = 0
    compliant_slots: int = 0
    misplaced_slots: int = 0
    missing_slots: int = 0
    extra_detections: int = 0
    compliance_score: float = 0.0
    spatial_alignment_accuracy: float = 0.0


class FreshnessMetrics(BaseModel):
    assessed: int = 0
    fresh: int = 0
    ripening: int = 0
    spoiled: int = 0
    spoilage_rate: float = 0.0


class ExpiryMetrics(BaseModel):
    scanned: int = 0
    valid: int = 0
    near_expiry: int = 0
    expired: int = 0
    unreadable: int = 0


class DiscrepancyRow(BaseModel):
    sku: str
    name: str
    detected: int
    system: int
    discrepancy: int
    type: str
    value_impact: float = 0.0


class InsightContext(BaseModel):
    """Exactly what the model is shown — nothing hidden, nothing implied."""

    generated_at: datetime
    scope: Literal["session", "window"] = "window"
    session_uid: Optional[str] = None
    shelf_id: Optional[str] = None
    window_hours: Optional[int] = None

    inventory: InventoryMetrics = Field(default_factory=InventoryMetrics)
    compliance: ComplianceMetrics = Field(default_factory=ComplianceMetrics)
    freshness: FreshnessMetrics = Field(default_factory=FreshnessMetrics)
    expiry: ExpiryMetrics = Field(default_factory=ExpiryMetrics)
    top_discrepancies: List[DiscrepancyRow] = Field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        """False when the shelf is clean — the prompt says so explicitly then."""
        return bool(
            self.inventory.phantom_skus
            or self.inventory.undercount_skus
            or self.inventory.overcount_skus
            or self.compliance.misplaced_slots
            or self.compliance.missing_slots
            or self.freshness.spoiled
            or self.freshness.ripening
            or self.expiry.expired
            or self.expiry.near_expiry
        )


class PromptBundle(BaseModel):
    """The compiled prompt, exposed so a paper can quote it verbatim."""

    system: str
    user: str
    audience: Audience = "store_manager"

    @property
    def approx_tokens(self) -> int:
        """Rough size check (~4 chars/token) to catch runaway contexts."""
        return (len(self.system) + len(self.user)) // 4


# -------------------------------------------------------------- LLM payload --
class LLMAction(BaseModel):
    """One action item as the model returned it, before we rank it."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=160)
    rationale: str = Field(..., min_length=1, max_length=600)
    severity: Severity = Severity.INFO

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, value: Any) -> Any:
        """Models write "high"/"urgent"/"med" — map them rather than 422 the lot."""
        if isinstance(value, str):
            key = value.strip().lower()
            aliases = {
                "critical": "critical", "urgent": "critical", "high": "critical",
                "severe": "critical", "blocker": "critical",
                "warning": "warning", "medium": "warning", "med": "warning",
                "moderate": "warning", "attention": "warning",
                "info": "info", "low": "info", "minor": "info", "normal": "info",
            }
            return aliases.get(key, "info")
        return value


class LLMInsightPayload(BaseModel):
    """Strict shape of the model's JSON reply.

    `extra="ignore"` on purpose: small models like to append commentary keys.
    Rejecting a well-formed briefing over a stray field would trade a good
    answer for a fallback, which is the wrong way round. Types and required
    fields are still enforced.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    headline: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(..., min_length=1, max_length=2000)
    actions: List[LLMAction] = Field(default_factory=list, max_length=10)

    @field_validator("actions", mode="before")
    @classmethod
    def _drop_malformed_actions(cls, value: Any) -> Any:
        """Keep the well-formed actions instead of failing the whole payload."""
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict) and item.get("title")]


# ----------------------------------------------------------------- response --
class InsightRequest(BaseModel):
    shelf_id: Optional[str] = None
    session_uid: Optional[str] = Field(
        default=None, description="Scope the briefing to one capture instead of a window"
    )
    window_hours: int = Field(default=24, ge=1, le=24 * 30)
    audience: Audience = "store_manager"
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Override the compiled telemetry (evaluation/replay)"
    )
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class InsightAction(BaseModel):
    priority: int = Field(..., ge=1, description="1 = do this first")
    title: str
    rationale: str
    severity: Severity = Severity.INFO


class InsightResponse(BaseModel):
    summary: str
    headline: Optional[str] = None
    actions: List[InsightAction] = Field(default_factory=list)

    model: str = Field(..., description="Model that actually produced this text")
    model_requested: Optional[str] = None
    model_substituted: bool = Field(
        default=False,
        description="True when the configured model was absent and another was used",
    )

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    generated_at: datetime

    degraded: bool = Field(
        default=False, description="True when a rule-based fallback produced this"
    )
    degraded_reason: Optional[str] = Field(
        default=None, description="Why the LLM path did not produce the briefing"
    )
    scope: Literal["session", "window"] = "window"
    session_uid: Optional[str] = None


class OllamaStatus(BaseModel):
    reachable: bool
    base_url: str
    default_model: str
    model_available: bool = Field(
        default=False, description="Whether the configured model is installed"
    )
    available_models: List[str] = Field(default_factory=list)
    version: Optional[str] = None
    error: Optional[str] = None
    hint: Optional[str] = Field(
        default=None, description="Actionable next step when something is missing"
    )
