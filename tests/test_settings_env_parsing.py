"""List-valued settings must load from a plain comma-separated env var.

This is the bug that broke the container on first start and would have broken
every fresh client install:

    pydantic_settings.exceptions.SettingsError:
        error parsing value for field "CORS_ORIGINS" from source "EnvSettingsSource"

pydantic-settings decodes complex field types from the environment with
json.loads *inside the settings source*, before any field_validator runs. A
`mode="before"` validator that splits on commas therefore never executes, and
the process dies during import.

It went unnoticed because the developer .env predated the variable, so the
default was used and nothing parsed. `.env.example` does set it, and START.bat
copies that file on a fresh install -- so the failure was reserved for people
who had never run the project before.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings

LIST_FIELDS = ("CORS_ORIGINS", "FRESHNESS_CLASSES", "OCR_LANGUAGES", "OCR_VARIANTS")


@pytest.mark.parametrize("field", LIST_FIELDS)
def test_csv_env_value_loads(monkeypatch: pytest.MonkeyPatch, field: str):
    """The exact form .env.example and docker-compose.yml use."""
    monkeypatch.setenv(field, "alpha,beta,gamma")
    assert getattr(Settings(), field) == ["alpha", "beta", "gamma"]


@pytest.mark.parametrize("field", LIST_FIELDS)
def test_json_env_value_still_loads(monkeypatch: pytest.MonkeyPatch, field: str):
    """NoDecode suppresses the source's decode, so the validator must do it."""
    monkeypatch.setenv(field, '["alpha","beta"]')
    assert getattr(Settings(), field) == ["alpha", "beta"]


def test_single_value_needs_no_comma(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OCR_LANGUAGES", "en")
    assert Settings().OCR_LANGUAGES == ["en"]


def test_surrounding_whitespace_is_trimmed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CORS_ORIGINS", " http://a.test , http://b.test ")
    assert Settings().CORS_ORIGINS == ["http://a.test", "http://b.test"]


def test_empty_entries_are_dropped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test,,http://b.test,")
    assert Settings().CORS_ORIGINS == ["http://a.test", "http://b.test"]


def test_defaults_apply_when_unset(monkeypatch: pytest.MonkeyPatch):
    for field in LIST_FIELDS:
        monkeypatch.delenv(field, raising=False)
    settings = Settings()
    assert settings.CORS_ORIGINS == ["http://localhost:3000", "http://127.0.0.1:3000"]
    assert settings.FRESHNESS_CLASSES == ["fresh", "ripening", "spoiled"]


def test_env_example_values_load(monkeypatch: pytest.MonkeyPatch):
    """Every list field in .env.example, loaded exactly as written.

    START.bat copies .env.example to .env on a fresh install, so anything in
    that file which cannot be parsed is a guaranteed first-run crash.
    """
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / ".env.example"
    applied = 0
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() in LIST_FIELDS:
            monkeypatch.setenv(key.strip(), value.strip())
            applied += 1

    assert applied, ".env.example should exercise at least one list field"
    settings = Settings()  # must not raise
    for field in LIST_FIELDS:
        assert isinstance(getattr(settings, field), list)
