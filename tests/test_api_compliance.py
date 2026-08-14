"""API-level compliance checks against a seeded, in-memory system.

Covers the preview path specifically: `persist=false` is documented as
"score this shelf without writing an audit row", and it returned 422 from Phase
0 until Phase 4 — the response model demanded a database id for a row that was
deliberately never inserted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.seed import seed_if_empty
from app.db.base import Base
from app.main import app

DETECTION = {
    "class_id": 0,
    "class_name": "bottle",
    "confidence": 0.9,
    "bbox": {"x1": 0.29, "y1": 0.09, "x2": 0.54, "y2": 0.28},
}


@pytest.fixture()
def client():  # noqa: ANN201
    """A seeded API backed by one shared in-memory database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # every session sees the same in-memory DB
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with TestingSession() as db:
        seed_if_empty(db)

    def override_get_db():  # noqa: ANN202
        db = TestingSession()
        try:
            yield db
            db.commit()
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    # No `with TestClient(...)` — the lifespan would hit the real database.
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_preview_check_returns_scores_without_persisting(client):  # noqa: ANN001
    response = client.post(
        "/api/v1/planogram/compliance",
        json={"detections": [DETECTION], "persist": False},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["id"] is None  # explicitly: nothing was written
    assert body["created_at"] is not None  # but the response is still timestamped
    assert body["total_slots"] == 8
    assert len(body["slot_results"]) == 8

    # And the audit list is still empty.
    assert client.get("/api/v1/planogram/audits").json() == []


def test_persisted_check_is_written_and_listed(client):  # noqa: ANN001
    response = client.post(
        "/api/v1/planogram/compliance",
        json={"detections": [DETECTION], "persist": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["id"], int)

    audits = client.get("/api/v1/planogram/audits").json()
    assert len(audits) == 1
    assert audits[0]["id"] == body["id"]


def test_seeded_system_scores_a_known_slot_as_compliant(client):  # noqa: ANN001
    """The seeded catalogue, class map and planogram must agree end to end."""
    response = client.post(
        "/api/v1/planogram/compliance",
        json={"detections": [DETECTION], "persist": False},
    )
    verdicts = {s["slot_id"]: s["status"] for s in response.json()["slot_results"]}
    # S1-R1-P2 expects SKU-WATER-500, which `bottle` maps to via data/class_map.json.
    assert verdicts["S1-R1-P2"] == "compliant"


def test_empty_shelf_reports_every_slot_missing(client):  # noqa: ANN001
    response = client.post(
        "/api/v1/planogram/compliance", json={"detections": [], "persist": False}
    )
    body = response.json()
    assert body["missing_slots"] == body["total_slots"]
    assert body["compliance_score"] == 0.0
