"""`/api/v1/freshness` — perishable spoilage classification."""

from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Pagination, complete_session, create_session, get_db, save_upload
from app.models.enums import FreshnessLabel
from app.models.freshness import FreshnessAudit
from app.models.product import Product
from app.schemas.freshness import (
    FreshnessAuditRead,
    FreshnessClassifyRequest,
    FreshnessClassifyResponse,
    FreshnessPrediction,
    FreshnessSummary,
)
from app.services.freshness import get_freshness_service

router = APIRouter()


@router.post("/classify", response_model=FreshnessClassifyResponse)
def classify_paths(
    payload: FreshnessClassifyRequest, db: Session = Depends(get_db)
) -> FreshnessClassifyResponse:
    """Classify crops already on disk (batch / evaluation path)."""
    service = get_freshness_service()
    predictions, latency_ms = service.predict(payload.image_paths)
    if not service.is_ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Freshness weights unavailable — train via models/train_freshness.py",
        )

    session = create_session(db)
    _persist(db, session.id, predictions, payload.product_sku, service.version, payload.persist)
    complete_session(db, session, total_latency_ms=latency_ms)
    return _to_response(session.session_uid, predictions, latency_ms)


@router.post("/classify/image", response_model=FreshnessClassifyResponse)
async def classify_upload(
    file: UploadFile = File(...),
    product_sku: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
) -> FreshnessClassifyResponse:
    started = time.perf_counter()
    image_path = await save_upload(file)

    service = get_freshness_service()
    predictions, _ = service.predict([str(image_path)])
    if not service.is_ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Freshness weights unavailable — train via models/train_freshness.py",
        )

    session = create_session(db, image_path=image_path)
    _persist(db, session.id, predictions, product_sku, service.version, persist=True)
    latency_ms = (time.perf_counter() - started) * 1000.0
    complete_session(db, session, total_latency_ms=latency_ms)
    return _to_response(session.session_uid, predictions, latency_ms)


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


def _persist(
    db: Session,
    session_id: Optional[int],
    predictions: List[FreshnessPrediction],
    product_sku: Optional[str],
    model_version: str,
    persist: bool,
) -> None:
    if not persist or not predictions:
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
            label=prediction.label,
            confidence=prediction.confidence,
            class_probabilities=prediction.class_probabilities,
            bbox=list(prediction.bbox.as_tuple()) if prediction.bbox else None,
            backbone=prediction.backbone,
            model_version=model_version,
            latency_ms=prediction.latency_ms,
        )
        for prediction in predictions
    )
    db.flush()


def _to_response(
    session_uid: str, predictions: List[FreshnessPrediction], latency_ms: float
) -> FreshnessClassifyResponse:
    return FreshnessClassifyResponse(
        session_uid=session_uid,
        predictions=predictions,
        spoiled_count=sum(1 for p in predictions if p.label is FreshnessLabel.SPOILED),
        ripening_count=sum(1 for p in predictions if p.label is FreshnessLabel.RIPENING),
        latency_ms=round(latency_ms, 2),
    )
