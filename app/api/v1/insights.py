"""`/api/v1/insights` — executive summaries generated locally via Ollama."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.insights import (
    InsightContext,
    InsightRequest,
    InsightResponse,
    OllamaStatus,
    PromptBundle,
)
from app.services.insight_context import (
    build_session_context,
    build_window_context,
    compile_prompt,
)
from app.services.llm_client import get_ollama_client

router = APIRouter()


@router.get("/status", response_model=OllamaStatus)
async def ollama_status() -> OllamaStatus:
    """Reachability *and* whether the configured model is actually installed."""
    return await get_ollama_client().status()


@router.post("/generate", response_model=InsightResponse)
async def generate_insight(
    payload: InsightRequest, db: Session = Depends(get_db)
) -> InsightResponse:
    """Produce a briefing: headline, summary and up to 3 prioritised actions.

    Scope is a single capture when `session_uid` is given, otherwise a rolling
    window. This endpoint does not fail on LLM problems — it returns a
    rule-based briefing with `degraded: true` and a `degraded_reason`, so the
    dashboard always renders something truthful.
    """
    context = _resolve_context(db, payload)
    return await get_ollama_client().generate(
        context=context,
        audience=payload.audience,
        model=payload.model,
        temperature=payload.temperature,
    )


@router.get("/context", response_model=InsightContext)
def insight_context(
    db: Session = Depends(get_db),
    shelf_id: Optional[str] = Query(default=None),
    session_uid: Optional[str] = Query(default=None),
    window_hours: int = Query(default=24, ge=1, le=24 * 30),
) -> InsightContext:
    """The exact telemetry the model is shown — required for reproducibility."""
    if session_uid:
        context = build_session_context(db, session_uid)
        if context is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {session_uid}")
        return context
    return build_window_context(db, shelf_id=shelf_id, window_hours=window_hours)


@router.get("/prompt", response_model=PromptBundle)
def insight_prompt(
    db: Session = Depends(get_db),
    shelf_id: Optional[str] = Query(default=None),
    session_uid: Optional[str] = Query(default=None),
    window_hours: int = Query(default=24, ge=1, le=24 * 30),
    audience: str = Query(default="store_manager"),
) -> PromptBundle:
    """The compiled system/user prompt, verbatim.

    Exposed so the paper can quote the prompt that produced a given figure
    without re-deriving it from source.
    """
    request = InsightRequest(
        shelf_id=shelf_id,
        session_uid=session_uid,
        window_hours=window_hours,
        audience=audience,  # type: ignore[arg-type]
    )
    return compile_prompt(_resolve_context(db, request), audience=request.audience)


def _resolve_context(db: Session, payload: InsightRequest) -> InsightContext:
    """Caller-supplied context wins; then session scope; then the window."""
    if payload.context is not None:
        return InsightContext.model_validate(payload.context)

    if payload.session_uid:
        context = build_session_context(db, payload.session_uid)
        if context is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Unknown session {payload.session_uid}"
            )
        return context

    return build_window_context(
        db, shelf_id=payload.shelf_id, window_hours=payload.window_hours
    )
