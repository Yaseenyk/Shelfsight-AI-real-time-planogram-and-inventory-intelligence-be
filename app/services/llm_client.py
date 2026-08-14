"""Ollama HTTP client for executive summaries (fully local, zero-cost).

Uses `POST /api/generate` with `stream=false`. When Ollama is unreachable the
service returns a deterministic rule-based summary flagged `degraded=True`, so
the dashboard never shows an empty insight panel during a demo.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import Severity
from app.schemas.insights import InsightAction, InsightResponse, OllamaStatus

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are ShelfSight AI, a retail operations analyst.
You receive structured shelf-audit telemetry from a computer-vision pipeline and
write concise, decision-ready briefings for a {audience}.

Rules:
- Lead with the single most costly problem.
- Quantify: cite SKU counts, discrepancy magnitude and value at risk.
- Never invent data that is not in the telemetry.
- Return STRICT JSON only, matching this shape:
  {{"headline": str, "summary": str,
    "actions": [{{"title": str, "rationale": str, "severity": "info|warning|critical"}}]}}
"""

USER_PROMPT = """Shelf telemetry (JSON):
{context}

Write the briefing now. JSON only, no markdown fences."""


class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_S

    async def status(self) -> OllamaStatus:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = [m.get("name", "") for m in response.json().get("models", [])]
            return OllamaStatus(
                reachable=True,
                base_url=self.base_url,
                default_model=self.model,
                available_models=models,
            )
        except Exception as exc:  # noqa: BLE001 - status must never raise
            return OllamaStatus(
                reachable=False,
                base_url=self.base_url,
                default_model=self.model,
                error=str(exc),
            )

    async def generate(
        self,
        context: Dict[str, Any],
        audience: str = "store_manager",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> InsightResponse:
        model_name = model or self.model
        payload = {
            "model": model_name,
            "system": SYSTEM_PROMPT.format(audience=audience.replace("_", " ")),
            "prompt": USER_PROMPT.format(context=json.dumps(context, indent=2, default=str)),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": (
                    temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
                ),
                "num_predict": settings.OLLAMA_NUM_PREDICT,
            },
        }

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception as exc:  # noqa: BLE001 - fall back, do not 500 the dashboard
            logger.warning("Ollama unreachable (%s) — using rule-based fallback", exc)
            return fallback_summary(context, model_name)

        latency_ms = (time.perf_counter() - started) * 1000.0
        parsed = _parse_json_response(body.get("response", ""))
        if parsed is None:
            return fallback_summary(context, model_name, latency_ms=latency_ms)

        actions = [
            InsightAction(
                title=str(item.get("title", "")).strip(),
                rationale=str(item.get("rationale", "")).strip(),
                severity=_coerce_severity(item.get("severity")),
            )
            for item in parsed.get("actions", [])
            if isinstance(item, dict) and item.get("title")
        ]
        return InsightResponse(
            summary=str(parsed.get("summary", "")).strip(),
            headline=str(parsed.get("headline", "")).strip() or None,
            actions=actions,
            model=model_name,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            latency_ms=latency_ms,
            generated_at=datetime.now(timezone.utc),
            degraded=False,
        )


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Tolerate models that wrap JSON in prose or fences."""
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _coerce_severity(value: Any) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.INFO


def fallback_summary(
    context: Dict[str, Any], model_name: str, latency_ms: Optional[float] = None
) -> InsightResponse:
    """Deterministic template used when the LLM is unavailable."""
    inventory = context.get("inventory", {}) or {}
    compliance = context.get("compliance", {}) or {}
    freshness = context.get("freshness", {}) or {}
    expiry = context.get("expiry", {}) or {}

    phantom = int(inventory.get("phantom_skus", 0) or 0)
    score = float(compliance.get("compliance_score", 0.0) or 0.0)
    spoiled = int(freshness.get("spoiled", 0) or 0)
    expired = int(expiry.get("expired", 0) or 0)

    actions: List[InsightAction] = []
    if phantom:
        actions.append(
            InsightAction(
                title=f"Audit {phantom} phantom SKU(s)",
                rationale="System stock is non-zero but nothing was detected on shelf.",
                severity=Severity.CRITICAL,
            )
        )
    if score < 0.85:
        actions.append(
            InsightAction(
                title="Reset shelf to planogram",
                rationale=f"Compliance is {score:.0%}, below the 85% operating floor.",
                severity=Severity.WARNING,
            )
        )
    if spoiled:
        actions.append(
            InsightAction(
                title=f"Remove {spoiled} spoiled unit(s)",
                rationale="Spoiled perishables detected by the freshness classifier.",
                severity=Severity.CRITICAL,
            )
        )
    if expired:
        actions.append(
            InsightAction(
                title=f"Pull {expired} expired pack(s)",
                rationale="OCR read an expiry date in the past.",
                severity=Severity.CRITICAL,
            )
        )

    summary = (
        f"Shelf audit: {phantom} phantom SKU(s), planogram compliance {score:.0%}, "
        f"{spoiled} spoiled and {expired} expired unit(s). "
        "Generated without the local LLM — start Ollama for narrative insight."
    )
    return InsightResponse(
        summary=summary,
        headline="Rule-based shelf briefing (LLM offline)",
        actions=actions,
        model=model_name,
        latency_ms=latency_ms,
        generated_at=datetime.now(timezone.utc),
        degraded=True,
    )


_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
