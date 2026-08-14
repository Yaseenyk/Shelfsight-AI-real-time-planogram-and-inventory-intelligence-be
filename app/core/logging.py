"""Structured-ish logging setup shared by the API and the evaluation harness."""

from __future__ import annotations

import contextlib
import logging
import sys

from app.core.config import settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    # Log messages throughout the codebase use en/em dashes, arrows and ellipses.
    # A Windows console defaults to a legacy code page (cp1252 on most Indian and
    # Western installs), where writing any of those raises UnicodeEncodeError
    # inside the logging handler. Logging swallows it and prints a multi-line
    # "--- Logging error ---" traceback in its place, so the run looks broken to
    # anyone watching, and the message itself is lost. Forcing the stream to
    # UTF-8 fixes every such message at once rather than restricting the
    # vocabulary of every log call in the project.
    if hasattr(sys.stdout, "reconfigure"):
        # A redirected or already-closed stream may refuse reconfiguration; that
        # is not worth failing a run over.
        with contextlib.suppress(ValueError, OSError):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
