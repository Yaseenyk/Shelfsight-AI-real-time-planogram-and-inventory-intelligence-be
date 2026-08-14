"""Structured-ish logging setup shared by the API and the evaluation harness."""

from __future__ import annotations

import logging
import sys

from app.core.config import settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or settings.LOG_LEVEL).upper(),
        format=_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Third-party chatter that drowns pipeline timings.
    for noisy in ("PIL", "matplotlib", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
