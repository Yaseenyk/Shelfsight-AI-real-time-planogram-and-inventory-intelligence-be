"""Reconciliation scoping — the rule that decides whether a phantom is findable.

Regression guard for the Phase 1 bug: a shelf-scoped scan considered only the
products it had *detected*, so an empty shelf produced zero rows and phantom
inventory — the system's headline capability — was undetectable exactly where it
matters most.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.enums import DiscrepancyType
from app.models.product import Product
from app.schemas.common import BoundingBox, Detection
from app.services.inventory import reconcile


@pytest.fixture()
def db():  # noqa: ANN201
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all(
        [
            Product(sku="SKU-COLA-330", name="Cola", system_stock=12, unit_price=1.2,
                    detection_class_name="cola_can"),
            Product(sku="SKU-WATER-500", name="Water", system_stock=18, unit_price=0.9,
                    detection_class_name="water_bottle"),
            Product(sku="SKU-OFFSHELF-1", name="Elsewhere", system_stock=5, unit_price=3.0),
        ]
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()


def _detection(sku: str, x1: float = 0.1) -> Detection:
    return Detection(
        class_id=0,
        class_name=sku,
        confidence=0.9,
        bbox=BoundingBox(x1=x1, y1=0.1, x2=x1 + 0.08, y2=0.3),
        sku=sku,
    )


def test_empty_shelf_reports_expected_skus_as_phantom(db):  # noqa: ANN001
    """Nothing detected, but the planogram says two SKUs belong here."""
    _, discrepancies = reconcile(
        db,
        detections=[],
        shelf_id="S1",
        persist=False,
        expected_skus=["SKU-COLA-330", "SKU-WATER-500"],
    )
    by_sku = {d.sku: d for d in discrepancies}
    assert set(by_sku) == {"SKU-COLA-330", "SKU-WATER-500"}
    assert all(d.discrepancy_type is DiscrepancyType.PHANTOM for d in discrepancies)
    # Off-shelf stock must not be blamed on this shelf.
    assert "SKU-OFFSHELF-1" not in by_sku


def test_partial_shelf_mixes_phantom_and_undercount(db):  # noqa: ANN001
    _, discrepancies = reconcile(
        db,
        detections=[_detection("SKU-COLA-330", 0.1), _detection("SKU-COLA-330", 0.3)],
        shelf_id="S1",
        persist=False,
        expected_skus=["SKU-COLA-330", "SKU-WATER-500"],
    )
    by_sku = {d.sku: d for d in discrepancies}
    assert by_sku["SKU-COLA-330"].discrepancy_type is DiscrepancyType.UNDERCOUNT
    assert by_sku["SKU-COLA-330"].detected_count == 2
    assert by_sku["SKU-WATER-500"].discrepancy_type is DiscrepancyType.PHANTOM


def test_unscoped_scan_still_covers_the_whole_catalogue(db):  # noqa: ANN001
    _, discrepancies = reconcile(db, detections=[], persist=False)
    assert len(discrepancies) == 3


def test_shelf_scan_without_a_planogram_falls_back_to_detected_only(db):  # noqa: ANN001
    # No expected_skus available: report on what was seen rather than blaming
    # the shelf for every SKU in the store.
    _, discrepancies = reconcile(
        db, detections=[_detection("SKU-COLA-330")], shelf_id="S1", persist=False
    )
    assert {d.sku for d in discrepancies} == {"SKU-COLA-330"}


def test_unresolved_detections_do_not_create_rows(db):  # noqa: ANN001
    # A detected object with no catalogue match is a compliance EXTRA, not an
    # inventory row — there is no system stock to compare it against.
    logs, discrepancies = reconcile(
        db, detections=[_detection("bus")], shelf_id="S1", persist=False
    )
    assert logs == [] and discrepancies == []
