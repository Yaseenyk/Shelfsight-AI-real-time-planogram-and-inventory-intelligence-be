"""LLM client: strict JSON validation, typed degradation, action ranking.

Ollama is stubbed. What matters here is that the client never raises, never lets
unvalidated model output reach the dashboard, and reports *why* it degraded.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest

from app.models.enums import Severity
from app.schemas.insights import (
    ComplianceMetrics,
    ExpiryMetrics,
    FreshnessMetrics,
    InsightContext,
    InventoryMetrics,
)
from app.services.llm_client import (
    OllamaClient,
    _model_matches,
    _rank_actions,
    fallback_summary,
    parse_llm_payload,
)


@pytest.fixture()
def context() -> InsightContext:
    return InsightContext(
        generated_at=datetime.now(timezone.utc),
        scope="session",
        session_uid="abc-123",
        shelf_id="S1",
        inventory=InventoryMetrics(
            total_skus=5, phantom_skus=2, undercount_skus=1, total_system=39, value_at_risk=41.4
        ),
        compliance=ComplianceMetrics(total_slots=8, compliant_slots=3, missing_slots=4,
                                     misplaced_slots=1, compliance_score=0.375),
        freshness=FreshnessMetrics(assessed=6, spoiled=2, spoilage_rate=0.333),
        expiry=ExpiryMetrics(scanned=4, expired=1, near_expiry=1),
    )


@pytest.fixture()
def clean_context() -> InsightContext:
    return InsightContext(generated_at=datetime.now(timezone.utc), scope="window")


VALID_REPLY = {
    "headline": "Two phantom SKUs driving 41 units of exposure",
    "summary": "Shelf S1 shows two phantom SKUs and 4 empty planogram slots.",
    "actions": [
        {"title": "Audit phantom SKUs", "rationale": "39 units on system, none on shelf.",
         "severity": "critical"},
        {"title": "Reset shelf", "rationale": "Compliance is 38%.", "severity": "warning"},
        {"title": "Review counts", "rationale": "One SKU undercounted.", "severity": "info"},
    ],
}


def _client(handler) -> OllamaClient:  # noqa: ANN001
    """An OllamaClient whose HTTP calls are served by `handler`."""
    transport = httpx.MockTransport(handler)
    return OllamaClient(
        base_url="http://stub:11434",
        model="llama3",
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )


def _run(coro):  # noqa: ANN001, ANN202
    return asyncio.run(coro)


# ------------------------------------------------------------ JSON parsing --
def test_parses_a_clean_json_reply():
    payload, error = parse_llm_payload(json.dumps(VALID_REPLY))
    assert error is None
    assert payload is not None
    assert payload.headline.startswith("Two phantom")
    assert len(payload.actions) == 3


def test_parses_json_wrapped_in_code_fences():
    fenced = "```json\n" + json.dumps(VALID_REPLY) + "\n```"
    payload, error = parse_llm_payload(fenced)
    assert error is None and payload is not None


def test_parses_json_buried_in_prose():
    noisy = "Sure! Here is the briefing:\n" + json.dumps(VALID_REPLY) + "\nHope that helps."
    payload, error = parse_llm_payload(noisy)
    assert error is None and payload is not None


def test_rejects_empty_and_non_json():
    assert parse_llm_payload("")[0] is None
    assert parse_llm_payload("I cannot help with that.")[0] is None


def test_rejects_payload_missing_required_fields():
    payload, error = parse_llm_payload(json.dumps({"summary": "only a summary"}))
    assert payload is None
    assert "validation" in (error or "").lower()


def test_ignores_unexpected_extra_keys():
    # Small models append commentary keys; that must not cost a good briefing.
    reply = {**VALID_REPLY, "confidence": 0.8, "notes": ["chatty"]}
    payload, error = parse_llm_payload(json.dumps(reply))
    assert error is None and payload is not None


def test_drops_malformed_actions_but_keeps_the_briefing():
    reply = {**VALID_REPLY, "actions": [VALID_REPLY["actions"][0], "not an object", {}]}
    payload, _ = parse_llm_payload(json.dumps(reply))
    assert payload is not None
    assert len(payload.actions) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("critical", Severity.CRITICAL), ("HIGH", Severity.CRITICAL), ("urgent", Severity.CRITICAL),
     ("warning", Severity.WARNING), ("medium", Severity.WARNING),
     ("info", Severity.INFO), ("low", Severity.INFO), ("banana", Severity.INFO)],
)
def test_severity_aliases_are_mapped(raw, expected):  # noqa: ANN001
    reply = {**VALID_REPLY, "actions": [{"title": "t", "rationale": "r", "severity": raw}]}
    payload, _ = parse_llm_payload(json.dumps(reply))
    assert payload is not None
    assert payload.actions[0].severity is expected


def test_overlong_strings_are_rejected():
    reply = {**VALID_REPLY, "headline": "x" * 500}
    assert parse_llm_payload(json.dumps(reply))[0] is None


# --------------------------------------------------------- action ranking --
def test_actions_are_ranked_by_severity_and_capped_at_three():
    payload, _ = parse_llm_payload(
        json.dumps(
            {
                **VALID_REPLY,
                "actions": [
                    {"title": "info one", "rationale": "r", "severity": "info"},
                    {"title": "critical one", "rationale": "r", "severity": "critical"},
                    {"title": "warning one", "rationale": "r", "severity": "warning"},
                    {"title": "extra", "rationale": "r", "severity": "info"},
                ],
            }
        )
    )
    ranked = _rank_actions(payload.actions)
    assert [a.priority for a in ranked] == [1, 2, 3]
    assert ranked[0].title == "critical one"
    assert ranked[-1].title == "info one"


# ------------------------------------------------------------- generation --
def test_generate_returns_validated_briefing(context):  # noqa: ANN001
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.32.9"})
        return httpx.Response(
            200,
            json={"response": json.dumps(VALID_REPLY), "prompt_eval_count": 300,
                  "eval_count": 120},
        )

    result = _run(_client(handler).generate(context))
    assert result.degraded is False
    assert result.headline.startswith("Two phantom")
    assert len(result.actions) == 3
    assert result.actions[0].priority == 1
    assert result.prompt_tokens == 300
    assert result.scope == "session" and result.session_uid == "abc-123"


def test_missing_model_is_reported_as_such_not_as_unreachable(context):  # noqa: ANN001
    """A 404 means 'ollama pull', not 'start the server' — the fix differs."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404, json={"error": "model 'llama3' not found"})

    result = _run(_client(handler).generate(context))
    assert result.degraded is True
    assert result.degraded_reason.startswith("model_not_found")
    assert "ollama pull" in result.degraded_reason


def test_unreachable_server_degrades_with_reason(context):  # noqa: ANN001
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = _run(_client(handler).generate(context))
    assert result.degraded is True
    assert result.degraded_reason.startswith("ollama_unreachable")


def test_timeout_degrades_with_reason(context):  # noqa: ANN001
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    result = _run(_client(handler).generate(context))
    assert result.degraded is True
    assert result.degraded_reason.startswith("timeout")


def test_unparseable_reply_degrades_rather_than_surfacing_junk(context):  # noqa: ANN001
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})
        return httpx.Response(200, json={"response": "I'm just a language model!"})

    result = _run(_client(handler).generate(context))
    assert result.degraded is True
    assert "invalid_json" in result.degraded_reason or "schema" in result.degraded_reason


def test_tag_expansion_is_not_reported_as_substitution(context):  # noqa: ANN001
    """`llama3` -> `llama3:latest` is the same model; flagging it cries wolf."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.32.9"})
        return httpx.Response(200, json={"response": json.dumps(VALID_REPLY)})

    result = _run(_client(handler).generate(context))
    assert result.model == "llama3:latest"
    assert result.model_substituted is False


def test_model_substitution_is_flagged_not_silent(context):  # noqa: ANN001
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "mistral:latest"}, {"name": "nomic-embed-text"}]}
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.32.9"})
        return httpx.Response(200, json={"response": json.dumps(VALID_REPLY)})

    result = _run(_client(handler).generate(context))
    assert result.model == "mistral:latest"  # embedding model correctly skipped
    assert result.model_requested == "llama3"
    assert result.model_substituted is True


# ----------------------------------------------------------------- status --
def test_status_reports_missing_model_with_a_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "mistral:latest"}]})
        return httpx.Response(200, json={"version": "0.32.9"})

    state = _run(_client(handler).status())
    assert state.reachable is True
    assert state.model_available is False
    assert "ollama pull" in state.hint


def test_status_never_raises_when_server_is_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    state = _run(_client(handler).status())
    assert state.reachable is False and state.error


def test_model_name_matching_tolerates_the_latest_tag():
    assert _model_matches("llama3", ["llama3:latest"]) == "llama3:latest"
    assert _model_matches("llama3:latest", ["llama3:latest"]) == "llama3:latest"
    assert _model_matches("mistral", ["llama3:latest"]) is None


# --------------------------------------------------------------- fallback --
def test_fallback_uses_only_telemetry_numbers(context):  # noqa: ANN001
    result = fallback_summary(context, model="llama3", reason="timeout")
    assert result.degraded is True
    assert "2 phantom SKU(s)" in result.summary or "phantom" in result.summary
    assert len(result.actions) <= 3
    assert result.actions[0].severity is Severity.CRITICAL  # expired stock first
    assert [a.priority for a in result.actions] == list(range(1, len(result.actions) + 1))


def test_fallback_on_a_clean_shelf_invents_nothing(clean_context):  # noqa: ANN001
    result = fallback_summary(clean_context, model="llama3", reason="ollama_unreachable")
    assert result.actions == []
    assert "clean" in result.headline.lower()
