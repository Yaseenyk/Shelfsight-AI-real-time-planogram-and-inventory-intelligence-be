"""Telemetry compilation and prompt building.

The prompt is the interface to the model, so these tests pin the things that
silently corrupt a briefing: double-counted SKUs, invented metrics, and a clean
shelf being handed a prompt that begs for problems.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.compliance import ComplianceAudit
from app.models.enums import DiscrepancyType, ExpiryStatus, FreshnessLabel, ScanStatus
from app.models.expiry import ExpiryAudit
from app.models.freshness import FreshnessAudit
from app.models.inventory import InventoryLog
from app.models.planogram import PlanogramLayout
from app.models.product import Product
from app.models.scan import ScanSession
from app.schemas.insights import InsightContext
from app.services.insight_context import (
    build_session_context,
    build_window_context,
    compile_prompt,
)


@pytest.fixture()
def db():  # noqa: ANN201
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    products = [
        Product(sku="SKU-COLA-330", name="Cola 330ml", system_stock=12, unit_price=1.20),
        Product(sku="SKU-WATER-500", name="Water 500ml", system_stock=18, unit_price=0.90),
    ]
    session.add_all(products)
    session.flush()

    scan = ScanSession(session_uid="sess-001", shelf_id="S1", status=ScanStatus.COMPLETED)
    layout = PlanogramLayout(planogram_uid="PLN-1", name="Bay", layout_json={"shelves": []})
    session.add_all([scan, layout])
    session.flush()

    session.add_all(
        [
            InventoryLog(session_id=scan.id, product_id=products[0].id, detected_count=0,
                         system_count=12, discrepancy=-12,
                         discrepancy_type=DiscrepancyType.PHANTOM, shelf_id="S1"),
            InventoryLog(session_id=scan.id, product_id=products[1].id, detected_count=15,
                         system_count=18, discrepancy=-3,
                         discrepancy_type=DiscrepancyType.UNDERCOUNT, shelf_id="S1"),
            ComplianceAudit(session_id=scan.id, planogram_id=layout.id, shelf_id="S1",
                            total_slots=8, compliant_slots=3, misplaced_slots=1,
                            missing_slots=4, extra_detections=2, compliance_score=0.375,
                            spatial_alignment_accuracy=0.75),
            FreshnessAudit(session_id=scan.id, label=FreshnessLabel.SPOILED, confidence=0.9),
            FreshnessAudit(session_id=scan.id, label=FreshnessLabel.FRESH, confidence=0.8),
            ExpiryAudit(session_id=scan.id, status=ExpiryStatus.EXPIRED, raw_text="EXP 01/01/2020"),
            ExpiryAudit(session_id=scan.id, status=ExpiryStatus.VALID, raw_text="EXP 01/01/2030"),
        ]
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()


# ------------------------------------------------------------ session scope --
def test_session_context_collects_every_pipeline(db):  # noqa: ANN001
    context = build_session_context(db, "sess-001")
    assert context is not None
    assert context.scope == "session" and context.shelf_id == "S1"

    assert context.inventory.phantom_skus == 1
    assert context.inventory.undercount_skus == 1
    assert context.inventory.total_detected == 15
    assert context.compliance.missing_slots == 4
    assert context.freshness.spoiled == 1
    assert context.freshness.spoilage_rate == pytest.approx(0.5)
    assert context.expiry.expired == 1


def test_unknown_session_returns_none(db):  # noqa: ANN001
    assert build_session_context(db, "does-not-exist") is None


def test_top_discrepancies_are_ranked_by_value(db):  # noqa: ANN001
    context = build_session_context(db, "sess-001")
    rows = context.top_discrepancies
    assert [r.sku for r in rows] == ["SKU-COLA-330", "SKU-WATER-500"]  # 14.40 > 2.70
    assert rows[0].value_impact == pytest.approx(14.4)


def test_rescanned_sku_is_counted_once(db):  # noqa: ANN001
    """A second scan of the same product must not double the phantom count."""
    product = db.query(Product).filter_by(sku="SKU-COLA-330").one()
    scan = db.query(ScanSession).filter_by(session_uid="sess-001").one()
    db.add(
        InventoryLog(session_id=scan.id, product_id=product.id, detected_count=12,
                     system_count=12, discrepancy=0,
                     discrepancy_type=DiscrepancyType.MATCH, shelf_id="S1")
    )
    db.commit()

    context = build_session_context(db, "sess-001")
    assert context.inventory.total_skus == 2  # not 3
    assert context.inventory.phantom_skus == 0  # the newer MATCH row wins


# ------------------------------------------------------------- window scope --
def test_window_context_filters_by_shelf(db):  # noqa: ANN001
    assert build_window_context(db, shelf_id="S1").inventory.total_skus == 2
    assert build_window_context(db, shelf_id="S9").inventory.total_skus == 0


def test_window_context_is_empty_outside_the_window(db):  # noqa: ANN001
    context = build_window_context(db, window_hours=1)
    assert context.scope == "window" and context.window_hours == 1


# ---------------------------------------------------------- prompt compile --
def test_prompt_contains_the_telemetry_and_the_rules(db):  # noqa: ANN001
    context = build_session_context(db, "sess-001")
    prompt = compile_prompt(context, audience="store_manager")

    assert "ShelfSight" in prompt.system
    assert "STRICT JSON" in prompt.system
    assert "exactly 3 action items" in prompt.system
    assert "SKU-COLA-330" in prompt.user
    assert "Never invent" in prompt.system
    assert prompt.approx_tokens > 0


def test_prompt_defines_the_domain_vocabulary():
    """Without a glossary, a small model reads 'phantom' as 'overstock'."""
    context = InsightContext(generated_at=datetime.now(timezone.utc))
    system = compile_prompt(context).system
    for term in ("phantom", "undercount", "overcount", "misplaced", "value_impact"):
        assert term in system
    assert "shelf saw NONE" in system or "camera saw NONE" in system


def test_prompt_varies_by_audience(db):  # noqa: ANN001
    context = build_session_context(db, "sess-001")
    manager = compile_prompt(context, audience="store_manager").system
    director = compile_prompt(context, audience="regional_director").system
    assert manager != director
    assert "shop floor" in manager
    assert "financial exposure" in director


def test_clean_shelf_prompt_discourages_invented_problems():
    context = InsightContext(generated_at=datetime.now(timezone.utc), scope="window")
    prompt = compile_prompt(context)
    assert "no discrepancies" in prompt.user
    assert "do not" in prompt.user.lower()


def test_empty_metric_blocks_are_omitted_from_the_prompt():
    """A model shown a wall of zeros writes about the zeros."""
    context = InsightContext(generated_at=datetime.now(timezone.utc), scope="window")
    user = compile_prompt(context).user
    assert "compliance" not in user
    assert "freshness" not in user


def test_prompt_is_deterministic_for_the_same_context(db):  # noqa: ANN001
    context = build_session_context(db, "sess-001")
    assert compile_prompt(context).user == compile_prompt(context).user


def test_context_round_trips_through_json(db):  # noqa: ANN001
    # /insights/context returns this; it must survive serialisation intact.
    context = build_session_context(db, "sess-001")
    restored = InsightContext.model_validate_json(context.model_dump_json())
    assert restored.inventory.phantom_skus == context.inventory.phantom_skus


def test_has_findings_flag():
    empty = InsightContext(generated_at=datetime.now(timezone.utc))
    assert empty.has_findings is False
    empty.expiry.expired = 1
    assert empty.has_findings is True
