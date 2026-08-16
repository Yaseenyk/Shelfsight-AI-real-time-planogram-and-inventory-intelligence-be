"""Schema creation plus the CLI entry point for seeding.

    python -m app.db.init_db            # create tables only
    python -m app.db.init_db --seed     # create tables and seed a fresh database

The seed data itself lives in `app/core/seed.py`, which the API also calls on
startup — one definition of "what a usable system contains", not two that drift.
"""

from __future__ import annotations

import argparse

from sqlalchemy import inspect

import app.models  # noqa: F401 - registers every table with the metadata
from app.core.logging import configure_logging, get_logger
from app.core.seed import seed, seed_if_empty
from app.db.base import Base
from app.db.session import SessionLocal, engine

logger = get_logger(__name__)


def add_missing_columns() -> int:
    """Add columns the models declare but an existing table does not have.

    `create_all` creates missing *tables* and never touches an existing one, so
    adding a field to a model works perfectly on a fresh database and breaks
    every deployment that already has data -- with an error that names a column
    rather than the upgrade that caused it. Adding `barcode` to Product did
    exactly that here.

    Deliberately additive only: it issues ADD COLUMN and nothing else. Dropping,
    renaming or retyping a column changes the meaning of data already stored and
    needs a considered migration, so those still belong in Alembic. This closes
    the common case without pretending to be a migration tool.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added = 0

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # create_all will build it
            present = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                if not column.nullable and column.default is None and column.server_default is None:
                    # A NOT NULL column with no default cannot be added to a
                    # table that already has rows; SQLite would reject it and
                    # the right answer is a real migration, not a guess.
                    logger.warning(
                        "Cannot add %s.%s automatically: NOT NULL with no default. "
                        "Write a migration for this one.",
                        table.name,
                        column.name,
                    )
                    continue
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column.type.compile(engine.dialect)}"
                if column.default is not None and column.default.is_scalar:
                    value = column.default.arg
                    ddl += f" DEFAULT {value!r}" if isinstance(value, str) else f" DEFAULT {value}"
                elif not column.nullable:
                    ddl += " DEFAULT ''" if "CHAR" in str(column.type).upper() else " DEFAULT 0"
                connection.exec_driver_sql(ddl)
                logger.info("Added missing column %s.%s", table.name, column.name)
                added += 1
    return added


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    added = add_missing_columns()
    if added:
        logger.info("Schema upgraded: %d column(s) added", added)
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
