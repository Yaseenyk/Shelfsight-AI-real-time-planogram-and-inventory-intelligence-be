from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.planogram import PlanogramDocument

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def default_planogram() -> PlanogramDocument:
    payload = json.loads(
        (ROOT / "data" / "planograms" / "default_planogram.json").read_text("utf-8")
    )
    return PlanogramDocument.model_validate(payload)
