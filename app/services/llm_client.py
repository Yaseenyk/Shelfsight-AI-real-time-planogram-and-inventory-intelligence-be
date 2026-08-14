"""Ollama HTTP client for executive summaries (fully local, zero-cost).

Contract: `generate()` **always returns an `InsightResponse`**. It never raises
and never returns None — a dashboard panel that shows an error box on a slow
model is worse than one showing a deterministic rule-based briefing marked
`degraded: true`. Every degradation carries a `degraded_reason` so the cause is
visible rather than guessed at.

Failure modes handled distinctly, because they need different fixes:

| Situation | `degraded_reason` |
| --- | --- |
| Ollama not running | `ollama_unreachable` |
| Configured model not installed | `model_not_found` (with the `ollama pull` hint) |
| Generation exceeded the timeout | `timeout` |
| Reply was not JSON / failed validation | `invalid_json` / `schema_validation_failed` |

The earlier version collapsed the middle case into "unreachable", which sent you
looking at the server when the actual fix was one `ollama pull`.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import Severity
from app.schemas.insights import (
    MAX_ACTIONS,
    Audience,
    InsightAction,
    InsightContext,
    InsightResponse,
    LLMInsightPayload,
    OllamaStatus,
    PromptBundle,
)
from app.services.insight_context import compile_prompt

logger = get_logger(__name__)

GENERATE_PATH = "/api/generate"
TAGS_PATH = "/api/tags"
VERSION_PATH = "/api/version"


class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        client_factory: Optional[Callable[..., httpx.AsyncClient]] = None,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_S
        # Injectable so tests drive a MockTransport instead of a live server.
        self._client_factory: Callable[..., httpx.AsyncClient] = (
            client_factory or httpx.AsyncClient
        )

    # -------------------------------------------------------------- status --
    async def status(self) -> OllamaStatus:
        """Reachability *and* model availability — status must never raise."""
        try:
            async with self._client_factory(
                timeout=settings.OLLAMA_CONNECT_TIMEOUT_S
            ) as client:
                response = await client.get(f"{self.base_url}{TAGS_PATH}")
                response.raise_for_status()
                models = [
                    str(entry.get("name", ""))
                    for entry in response.json().get("models", [])
                    if entry.get("name")
                ]
                version = await self._version(client)
        except Exception as exc:  # noqa: BLE001 - status reports, never raises
            return OllamaStatus(
                reachable=False,
                base_url=self.base_url,
                default_model=self.model,
                error=str(exc),
                hint=f"Start Ollama, or set OLLAMA_BASE_URL (currently {self.base_url})",
            )

        available = _model_matches(self.model, models)
        return OllamaStatus(
            reachable=True,
            base_url=self.base_url,
            default_model=self.model,
            model_available=available is not None,
            available_models=models,
            version=version,
            hint=None if available else f"Model not installed — run: ollama pull {self.model}",
        )

    async def _version(self, client: httpx.AsyncClient) -> Optional[str]:
        try:
            response = await client.get(f"{self.base_url}{VERSION_PATH}")
            response.raise_for_status()
            return str(response.json().get("version"))
        except Exception:  # noqa: BLE001 - version is decoration, not a dependency
            return None

    async def resolve_model(self, requested: Optional[str] = None) -> Tuple[str, bool]:
        """Return `(model_to_use, substituted)`.

        If the requested model is absent but another completion-capable one is
        installed, use it and flag the substitution. The dashboard then works on
        a fresh machine, while the response still records that the configured
        model was not what ran — silent substitution would poison a paper's
        reproducibility claims.
        """
        requested = requested or self.model
        if not settings.OLLAMA_AUTO_SELECT_MODEL:
            return requested, False

        state = await self.status()
        if not state.reachable or not state.available_models:
            return requested, False

        exact = _model_matches(requested, state.available_models)
        if exact:
            # Resolving "llama3.2" to "llama3.2:latest" is tag expansion, not a
            # substitution — reporting it as one would cry wolf on every run.
            return exact, False

        # Prefer a generative model; embedding-only models cannot write prose.
        candidates = [m for m in state.available_models if not _is_embedding_model(m)]
        if not candidates:
            return requested, False

        chosen = candidates[0]
        logger.warning(
            "Configured model %r is not installed; using %r instead (ollama pull %s)",
            requested,
            chosen,
            requested,
        )
        return chosen, True

    # ------------------------------------------------------------ generate --
    async def generate(
        self,
        context: InsightContext,
        audience: Audience = "store_manager",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> InsightResponse:
        """Compile the prompt, call Ollama, validate the reply, or degrade."""
        prompt = compile_prompt(context, audience=audience, max_actions=MAX_ACTIONS)
        requested = model or self.model
        resolved, substituted = await self.resolve_model(requested)

        logger.debug("Insight prompt ~%d tokens for %s", prompt.approx_tokens, resolved)
        started = time.perf_counter()
        try:
            body = await self._post_generate(prompt, resolved, temperature)
        except _GenerationFailure as failure:
            return fallback_summary(
                context,
                model=resolved,
                requested=requested,
                substituted=substituted,
                reason=failure.reason,
                detail=failure.detail,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_text = str(body.get("response", ""))

        payload, parse_error = parse_llm_payload(raw_text)
        if payload is None:
            logger.warning("Model %s returned unusable JSON: %s", resolved, parse_error)
            return fallback_summary(
                context,
                model=resolved,
                requested=requested,
                substituted=substituted,
                reason="invalid_json" if "JSON" in (parse_error or "") else "schema_validation_failed",
                detail=parse_error,
                latency_ms=latency_ms,
            )

        return InsightResponse(
            summary=payload.summary,
            headline=payload.headline,
            actions=_rank_actions(payload.actions),
            model=resolved,
            model_requested=requested,
            model_substituted=substituted,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            latency_ms=round(latency_ms, 2),
            generated_at=datetime.now(timezone.utc),
            degraded=False,
            scope=context.scope,
            session_uid=context.session_uid,
        )

    async def _post_generate(
        self, prompt: PromptBundle, model: str, temperature: Optional[float]
    ) -> Dict[str, Any]:
        """POST /api/generate, mapping transport failures onto typed reasons."""
        payload = {
            "model": model,
            "system": prompt.system,
            "prompt": prompt.user,
            "stream": False,
            "format": "json",  # Ollama constrains sampling to valid JSON
            "options": {
                "temperature": (
                    temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
                ),
                "num_predict": settings.OLLAMA_NUM_PREDICT,
            },
        }
        timeout = httpx.Timeout(
            self.timeout, connect=settings.OLLAMA_CONNECT_TIMEOUT_S
        )
        try:
            async with self._client_factory(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}{GENERATE_PATH}", json=payload)
                if response.status_code == 404:
                    raise _GenerationFailure(
                        "model_not_found",
                        f"Ollama has no model named {model!r} — run: ollama pull {model}",
                    )
                response.raise_for_status()
                return response.json()
        except _GenerationFailure:
            raise
        except httpx.TimeoutException as exc:
            raise _GenerationFailure(
                "timeout", f"Ollama did not respond within {self.timeout:.0f}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise _GenerationFailure(
                "http_error", f"Ollama returned {exc.response.status_code}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - connection refused, DNS, etc.
            raise _GenerationFailure("ollama_unreachable", str(exc)) from exc


class _GenerationFailure(Exception):
    """Internal: a typed transport/generation failure with a stable reason code."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


# ------------------------------------------------------------ JSON handling --
def parse_llm_payload(text: str) -> Tuple[Optional[LLMInsightPayload], Optional[str]]:
    """Validate the model's reply into `LLMInsightPayload`.

    Returns `(payload, None)` or `(None, error)`. Tolerates the two things small
    models reliably do — fencing the JSON, and wrapping it in prose — but the
    *shape* is then validated strictly by Pydantic.
    """
    raw = (text or "").strip()
    if not raw:
        return None, "Empty response from model"

    for candidate in _json_candidates(raw):
        try:
            return LLMInsightPayload.model_validate_json(candidate), None
        except ValidationError as exc:
            last_error = f"Schema validation failed: {exc.errors()[0].get('msg', exc)}"
        except ValueError:
            last_error = "Response was not valid JSON"
    return None, last_error


def _json_candidates(raw: str) -> List[str]:
    """Progressively more forgiving extractions of a JSON object."""
    candidates = [raw]

    if raw.startswith("```"):
        stripped = raw.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        candidates.append(stripped.strip())

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    seen: set[str] = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]


def _rank_actions(actions: List[Any]) -> List[InsightAction]:
    """Order by severity, then cap at `MAX_ACTIONS`, assigning 1-based priority."""
    order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    ranked = sorted(actions, key=lambda action: order.get(action.severity, 3))
    return [
        InsightAction(
            priority=index,
            title=action.title,
            rationale=action.rationale,
            severity=action.severity,
        )
        for index, action in enumerate(ranked[:MAX_ACTIONS], start=1)
    ]


def _model_matches(requested: str, available: List[str]) -> Optional[str]:
    """Match `llama3` against `llama3:latest` and vice versa."""
    if requested in available:
        return requested
    base = requested.split(":", 1)[0]
    for name in available:
        if name.split(":", 1)[0] == base:
            return name
    return None


def _is_embedding_model(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("embed", "bge", "minilm", "e5-"))


# --------------------------------------------------------------- fallback ---
def fallback_summary(
    context: InsightContext,
    model: str,
    requested: Optional[str] = None,
    substituted: bool = False,
    reason: Optional[str] = None,
    detail: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> InsightResponse:
    """Deterministic rule-based briefing used whenever the LLM path fails.

    Written from the same telemetry, so the dashboard degrades in content
    quality — not in correctness. `degraded=True` keeps generated prose
    distinguishable from templated prose in any figure or transcript.
    """
    inventory = context.inventory
    compliance = context.compliance
    freshness = context.freshness
    expiry = context.expiry

    actions: List[InsightAction] = []

    def add(title: str, rationale: str, severity: Severity) -> None:
        if len(actions) < MAX_ACTIONS:
            actions.append(
                InsightAction(
                    priority=len(actions) + 1,
                    title=title,
                    rationale=rationale,
                    severity=severity,
                )
            )

    if expiry.expired:
        add(
            f"Pull {expiry.expired} expired pack(s)",
            "OCR read an expiry date in the past; these cannot be sold.",
            Severity.CRITICAL,
        )
    if freshness.spoiled:
        add(
            f"Remove {freshness.spoiled} spoiled unit(s)",
            f"Spoilage rate is {freshness.spoilage_rate:.0%} of assessed produce.",
            Severity.CRITICAL,
        )
    if inventory.phantom_skus:
        worst = context.top_discrepancies[0] if context.top_discrepancies else None
        detail_text = (
            f"Worst is {worst.sku} ({worst.system} units on system, none on shelf)."
            if worst
            else "System stock is non-zero but nothing was detected on shelf."
        )
        add(
            f"Audit {inventory.phantom_skus} phantom SKU(s)",
            detail_text,
            Severity.CRITICAL,
        )
    if compliance.total_slots and compliance.compliance_score < 0.85:
        add(
            "Reset shelf to planogram",
            f"Compliance is {compliance.compliance_score:.0%} "
            f"({compliance.missing_slots} missing, {compliance.misplaced_slots} misplaced).",
            Severity.WARNING,
        )
    if expiry.near_expiry:
        add(
            f"Mark down {expiry.near_expiry} near-expiry pack(s)",
            "Within the 7-day expiry threshold; discount before they are written off.",
            Severity.WARNING,
        )

    if context.has_findings:
        headline = "Shelf needs attention"
        summary = (
            f"{inventory.phantom_skus} phantom SKU(s), planogram compliance "
            f"{compliance.compliance_score:.0%}, {freshness.spoiled} spoiled and "
            f"{expiry.expired} expired unit(s) across {inventory.total_skus} tracked SKU(s)."
        )
    else:
        headline = "Shelf is clean"
        summary = "No discrepancies detected in this scope."

    summary += f" (Rule-based briefing — the local LLM was unavailable: {reason or 'unknown'}.)"

    return InsightResponse(
        summary=summary,
        headline=headline,
        actions=actions,
        model=model,
        model_requested=requested,
        model_substituted=substituted,
        latency_ms=latency_ms,
        generated_at=datetime.now(timezone.utc),
        degraded=True,
        degraded_reason=f"{reason}: {detail}" if detail else reason,
        scope=context.scope,
        session_uid=context.session_uid,
    )


_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
