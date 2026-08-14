"""First-boot seeding.

The rule under test: a fresh database becomes usable, and a populated one is
never touched. A restart that resurrects a deleted product or resets a corrected
stock count would be worse than not seeding at all.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.seed import (
    SEED_PRODUCTS,
    catalogue_is_empty,
    planograms_are_empty,
    seed,
    seed_if_empty,
    seed_products,
)
from app.db.base import Base
from app.models.planogram import PlanogramLayout
from app.models.product import Product


@pytest.fixture()
def db():  # noqa: ANN201
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def test_fresh_database_is_detected_as_empty(db):  # noqa: ANN001
    assert catalogue_is_empty(db) is True
    assert planograms_are_empty(db) is True


def test_seeding_populates_catalogue_and_planogram(db):  # noqa: ANN001
    result = seed_if_empty(db)

    assert result["products"] == len(SEED_PRODUCTS)
    assert result["planograms"] >= 1  # data/planograms/default_planogram.json

    products = db.execute(select(Product)).scalars().all()
    assert {p.sku for p in products} == {p["sku"] for p in SEED_PRODUCTS}

    layout = db.execute(select(PlanogramLayout)).scalars().first()
    assert layout is not None
    assert layout.shelf_count == 3  # the default 3-shelf bay
    assert layout.slot_count > 0
    assert layout.is_active is True


def test_seed_products_map_to_detector_classes(db):  # noqa: ANN001
    """Every seeded SKU must resolve through the shipped class map.

    Otherwise a stock YOLOv8n detects objects that match no product and the
    first scan reports nothing on a system that looks correctly configured.
    """
    import json
    from pathlib import Path

    from app.core.config import settings

    mapping = json.loads(Path(settings.DETECTION_CLASS_MAP).read_text(encoding="utf-8"))
    known = {k.lower() for k in mapping["mapping"]}

    seed_if_empty(db)
    for product in db.execute(select(Product)).scalars().all():
        assert product.detection_class_name, product.sku
        assert product.detection_class_name.lower() in known, product.sku


def test_seeded_skus_match_the_planogram(db):  # noqa: ANN001
    """A planogram slot referencing an unstocked SKU can never be compliant."""
    seed_if_empty(db)
    catalogue = {p.sku for p in db.execute(select(Product)).scalars().all()}
    layout = db.execute(select(PlanogramLayout)).scalars().first()

    slot_skus = {
        slot["sku"]
        for shelf in layout.layout_json["shelves"]
        for row in shelf["rows"]
        for slot in row["slots"]
    }
    assert slot_skus <= catalogue, f"planogram references unknown SKUs: {slot_skus - catalogue}"


def test_second_boot_changes_nothing(db):  # noqa: ANN001
    first = seed_if_empty(db)
    second = seed_if_empty(db)

    assert first["products"] > 0
    assert second == {"products": 0, "planograms": 0}
    assert len(db.execute(select(Product)).scalars().all()) == len(SEED_PRODUCTS)


# Reference the catalogue rather than literal SKUs: renaming a product (as the
# India localisation did) should not break behavioural tests.
FIRST_SKU = SEED_PRODUCTS[0]["sku"]
SECOND_SKU = SEED_PRODUCTS[1]["sku"]


def test_operator_edits_survive_a_restart(db):  # noqa: ANN001
    seed_if_empty(db)

    product = db.execute(select(Product).where(Product.sku == FIRST_SKU)).scalar_one()
    product.system_stock = 999
    db.commit()

    seed_if_empty(db)

    refreshed = db.execute(select(Product).where(Product.sku == FIRST_SKU)).scalar_one()
    assert refreshed.system_stock == 999


def test_deleted_product_is_not_resurrected_on_restart(db):  # noqa: ANN001
    seed_if_empty(db)
    victim = db.execute(select(Product).where(Product.sku == SECOND_SKU)).scalar_one()
    db.delete(victim)
    db.commit()

    seed_if_empty(db)  # catalogue is non-empty, so nothing is re-added

    remaining = {p.sku for p in db.execute(select(Product)).scalars().all()}
    assert SECOND_SKU not in remaining


def test_seed_products_only_adds_missing_rows(db):  # noqa: ANN001
    db.add(Product(sku=FIRST_SKU, name="Pre-existing", system_stock=1))
    db.flush()

    added = seed_products(db)
    assert added == len(SEED_PRODUCTS) - 1

    survivor = db.execute(select(Product).where(Product.sku == FIRST_SKU)).scalar_one()
    assert survivor.name == "Pre-existing"


def test_explicit_seed_reupserts_planograms(db):  # noqa: ANN001
    seed_if_empty(db)
    result = seed(db, force=True)
    assert result["planograms"] >= 1
    assert result["products"] == 0  # products are still never overwritten
