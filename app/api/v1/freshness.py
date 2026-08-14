"""`/api/v1/freshness` — perishable spoilage classification."""

from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Pagination, complete_session, create_session, get_db, save_upload
from app.models.enums import FreshnessLabel, ScanStatus
from app.models.freshness import FreshnessAudit
from app.models.product import Product
from app.schemas.freshness import (
    FreshnessAuditRead,
    FreshnessClassifyRequest,
    FreshnessClassifyResponse,
    FreshnessSummary,
)
from app.services.freshness import (
    FreshnessError,
    FreshnessResult,
    FreshnessUnavailableError,
    get_freshness_service,
)
from app.utils.vision import ImageDecodeError, decode_image_bytes, read_image_file

router = APIRouter()


@router.post("/classify", response_model=FreshnessClassifyResponse)
async def classify_image(
    file: UploadFile = File(..., description="Produce crop (JPEG/PNG/WebP/BMP)"),
    product_sku: Optional[str] = Form(default=None),
    persist: bool = Form(default=True),
    db: Session = Depends(get_db),
) -> FreshnessClassifyResponse:
    """Classify a perishable crop as Fresh / Ripening / Spoiled.

    Returns the predicted label, softmax probabilities for **all three** classes
    and latency, and writes a `FreshnessAudit` row. A frame that fails is
    recorded as a `FAILED` scan session rather than disappearing.
    """
    started = time.perf_counter()
    payload = await file.read()
    await file.seek(0)

    try:
        frame = decode_image_bytes(payload)
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    image_path = await save_upload(file)
    session = create_session(db, image_path=image_path)

    service = get_freshness_service()
    try:
        result = service.predict_freshness(frame)
    except FreshnessUnavailableError as exc:
        _fail_session(db, session, str(exc))
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except FreshnessError as exc:
        _fail_session(db, session, str(exc))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    session.image_width = result.image_width
    session.image_height = result.image_height

    if persist:
        _persist(db, session.id, [result], product_sku, service.version)
    latency_ms = (time.perf_counter() - started) * 1000.0
    complete_session(db, session, total_latency_ms=latency_ms)
    return _to_response(session.session_uid, [result], latency_ms)


@router.post("/classify/batch", response_model=FreshnessClassifyResponse)
def classify_paths(
    payload: FreshnessClassifyRequest, db: Session = Depends(get_db)
) -> FreshnessClassifyResponse:
    """Classify crops already on disk — the batch/evaluation path."""
    started = time.perf_counter()
    service = get_freshness_service()

    try:
        images = [read_image_file(path) for path in payload.image_paths]
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    try:
        results = service.predict_batch(images)
    except FreshnessUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except FreshnessError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    session = create_session(db)
    if payload.persist:
        _persist(db, session.id, results, payload.product_sku, service.version)
    latency_ms = (time.perf_counter() - started) * 1000.0
    complete_session(db, session, total_latency_ms=latency_ms)
    return _to_response(session.session_uid, results, latency_ms)


@router.post(
    "/classify/image",
    response_model=FreshnessClassifyResponse,
    deprecated=True,
    summary="Deprecated alias for POST /freshness/classify",
)
async def classify_upload(
    file: UploadFile = File(...),
    product_sku: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
) -> FreshnessClassifyResponse:
    """Phase 0 route name, kept so existing clients do not break."""
    return await classify_image(file=file, product_sku=product_sku, persist=True, db=db)


@router.get("/audits", response_model=List[FreshnessAuditRead])
def list_audits(
    db: Session = Depends(get_db),
    page: Pagination = Depends(),
    label: Optional[FreshnessLabel] = Query(default=None),
):
    stmt = select(FreshnessAudit).order_by(
        FreshnessAudit.created_at.desc(), FreshnessAudit.id.desc()
    )
    if label:
        stmt = stmt.where(FreshnessAudit.label == label)
    return list(db.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all())


@router.get("/summary", response_model=FreshnessSummary)
def summary(db: Session = Depends(get_db)) -> FreshnessSummary:
    audits = db.execute(select(FreshnessAudit)).scalars().all()
    total = len(audits)
    counts = dict.fromkeys(FreshnessLabel, 0)
    for audit in audits:
        counts[audit.label] += 1
    confidences = [a.confidence for a in audits if a.confidence is not None]
    return FreshnessSummary(
        total_assessed=total,
        fresh=counts[FreshnessLabel.FRESH],
        ripening=counts[FreshnessLabel.RIPENING],
        spoiled=counts[FreshnessLabel.SPOILED],
        spoilage_rate=(counts[FreshnessLabel.SPOILED] / total) if total else 0.0,
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
    )


# ------------------------------------------------------------------ helpers --
def _fail_session(db: Session, session, message: str) -> None:  # noqa: ANN001
    session.status = ScanStatus.FAILED
    session.error_message = message[:1024]
    db.flush()


def _persist(
    db: Session,
    session_id: Optional[int],
    results: List[FreshnessResult],
    product_sku: Optional[str],
    model_version: str,
) -> None:
    if not results:
        return

    product_id = None
    if product_sku:
        product = db.execute(
            select(Product).where(Product.sku == product_sku)
        ).scalar_one_or_none()
        product_id = product.id if product else None

    db.add_all(
        FreshnessAudit(
            session_id=session_id,
            product_id=product_id,
            label=result.label,
            confidence=result.confidence,
            class_probabilities=result.probabilities,
            bbox=list(result.bbox.as_tuple()) if result.bbox else None,
            backbone=result.backbone,
            model_version=model_version,
            latency_ms=result.latency_ms,
        )
        for result in results
    )
    db.flush()


def _to_response(
    session_uid: str, results: List[FreshnessResult], latency_ms: float
) -> FreshnessClassifyResponse:
    return FreshnessClassifyResponse(
        session_uid=session_uid,
        predictions=[result.to_prediction() for result in results],
        spoiled_count=sum(1 for r in results if r.label is FreshnessLabel.SPOILED),
        ripening_count=sum(1 for r in results if r.label is FreshnessLabel.RIPENING),
        latency_ms=round(latency_ms, 2),
    )
