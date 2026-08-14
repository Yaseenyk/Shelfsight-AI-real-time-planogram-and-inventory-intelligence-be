from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.models.enums import Severity


class InsightRequest(BaseModel):
    """Ask the local LLM to narrate the current shelf state.

    `context` is optional: when omitted the service assembles it from the most
    recent audits so the frontend can fire a one-click "explain this shelf".
    """

    shelf_id: Optional[str] = None
    session_uid: Optional[str] = None
    window_hours: int = Field(default=24, ge=1, le=24 * 30)
    audience: Literal["store_manager", "regional_director", "analyst"] = "store_manager"
    context: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class InsightAction(BaseModel):
    title: str
    rationale: str
    severity: Severity = Severity.INFO


class InsightResponse(BaseModel):
    summary: str
    headline: Optional[str] = None
    actions: List[InsightAction] = Field(default_factory=list)
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    generated_at: datetime
    degraded: bool = Field(
        default=False, description="True when Ollama was unreachable and a rule-based fallback ran"
    )


class OllamaStatus(BaseModel):
    reachable: bool
    base_url: str
    default_model: str
    available_models: List[str] = Field(default_factory=list)
    error: Optional[str] = None
