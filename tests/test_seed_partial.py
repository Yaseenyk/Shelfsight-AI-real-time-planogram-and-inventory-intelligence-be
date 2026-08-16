"""Seeding must fill in what is missing, not skip everything when anything exists.

The bug this guards against shipped: `seed_if_empty` asked one question --
"is this database empty?" -- and did nothing when the answer was no. That is
correct only until a new seed category is added. Users were added after products
and planograms, so every already-populated database came up with no accounts,
and the symptom was a login endpoint that worked perfectly and rejected every
correct PIN.

Upgrades are the normal case for a deployed system, so "fresh install works" is
not enough of a test.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

import app.models  # noqa: F401 - registers every table
from app.core.seed import seed_if_empty, seed_products, seed_users
from app.db.base import Base
from app.models.product import Product
from app.models.user import User


@pytest.fixture
def db() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _count(session: Session, model) -> int:  # noqa: ANN001
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def test_fresh_database_gets_everything(db: Session):
    result = seed_if_empty(db)
    assert result["products"] > 0
    assert result["users"] == 3
    assert _count(db, User) == 3


def test_users_are_seeded_into_an_existing_catalogue(db: Session):
    """The upgrade path: products already present, users table brand new."""
    seed_products(db)
    db.commit()
    assert _count(db, Product) > 0
    assert _count(db, User) == 0

    result = seed_if_empty(db)

    assert result["users"] == 3, "users must be seeded even though products existed"
    assert result["products"] == 0, "existing products must not be re-seeded"
    assert _count(db, User) == 3


def test_nothing_is_reseeded_when_everything_exists(db: Session):
    seed_if_empty(db)
    before_users = _count(db, User)
    before_products = _count(db, Product)

    result = seed_if_empty(db)

    assert result == {"products": 0, "planograms": 0, "users": 0}
    assert _count(db, User) == before_users
    assert _count(db, Product) == before_products


def test_seeding_never_overwrites_a_changed_pin(db: Session):
    """An operator who changes the manager PIN must not have it reset on restart."""
    seed_users(db)
    db.commit()
    manager = db.execute(select(User).where(User.username == "manager")).scalar_one()
    manager.set_pin("987654")
    db.commit()

    seed_users(db)
    db.commit()

    manager = db.execute(select(User).where(User.username == "manager")).scalar_one()
    assert manager.verify_pin("987654"), "the operator's PIN must survive a restart"
    assert not manager.verify_pin("1001"), "the published demo PIN must not come back"


def test_every_seeded_role_is_present(db: Session):
    seed_users(db)
    db.commit()
    roles = {user.role.value for user in db.execute(select(User)).scalars()}
    assert roles == {"manager", "coordinator", "staff"}


def test_seeded_users_can_authenticate(db: Session):
    """End to end: the published PINs actually open a session."""
    from app.services import auth

    seed_users(db)
    db.commit()

    for username, pin in (("manager", "1001"), ("coordinator", "2002"), ("staff", "3003")):
        session = auth.authenticate(db, username, pin)
        assert session.token
        assert session.user.username == username
