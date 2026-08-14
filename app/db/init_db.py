"""Schema creation plus the CLI entry point for seeding.

    python -m app.db.init_db            # create tables only
    python -m app.db.init_db --seed     # create tables and seed a fresh database

The seed data itself lives in `app/core/seed.py`, which the API also calls on
startup — one definition of "what a usable system contains", not two that drift.
"""

from __future__ import annotations

import argparse

from app.core.logging import configure_logging, get_logger
from app.core.seed import seed, seed_if_empty
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.models import (  # noqa: F401 - imported for metadata registration
    ComplianceAudit,
    ExpiryAudit,
    FreshnessAudit,
    InventoryLog,
    PlanogramLayout,
    Product,
    ScanSession,
)

logger = get_logger(__name__)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured at %s", engine.url)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Initialise the ShelfSight database")
    parser.add_argument(
        "--seed", action="store_true", help="insert the catalogue + planograms if missing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-upsert planograms even when the database is already populated",
    )
    args = parser.parse_args()

    init_db()
    if args.seed:
        with SessionLocal() as db:
            result = seed(db, force=True) if args.force else seed_if_empty(db)
        print(f"products added: {result['products']}, planograms upserted: {result['planograms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
