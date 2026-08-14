"""FastAPI entrypoint for ShelfSight AI.

Run: `uvicorn app.main:app --reload --port 8000`
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.init_db import init_db
from app.schemas.common import ErrorResponse, HealthResponse
from app.services.detection import get_detector
from app.services.llm_client import get_ollama_client

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_directories()
    init_db()
    # Load + warm the detector so the first scan request is not penalised by it.
    detector = get_detector()
    if detector.load():
        logger.info("Detector classes: %d", len(detector.class_names))
    else:
        logger.warning("Detector unavailable: %s", detector.load_failure)
    logger.info("%s v%s ready", settings.APP_NAME, settings.APP_VERSION)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Real-time planogram compliance, phantom-inventory detection, freshness "
        "classification, expiry OCR and local-LLM executive insight."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.ensure_directories()
app.mount("/media", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="media")
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "api": settings.API_V1_PREFIX,
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    from sqlalchemy import text  # noqa: PLC0415

    from app.db.session import engine  # noqa: PLC0415

    database = "ok"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - health must report, not raise
        database = f"error: {exc}"

    ollama = await get_ollama_client().status()
    detector = get_detector()
    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc),
        database=database,
        detector_loaded=detector.is_ready,
        detector_model=detector.version if detector.is_ready else None,
        detector_classes=len(detector.class_names) or None,
        detector_error=detector.load_failure,
        ollama_reachable=ollama.reachable,
    )


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    """Domain validation errors (bad planogram geometry, etc.) → 422, not 500."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(detail=str(exc), code="validation_error").model_dump(),
    )
