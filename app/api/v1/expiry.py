"""`/api/v1/expiry` — packaging OCR plus date normalisation and validity status."""

from __future__ import annotations

import time
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Pagination, complete_session, create_session, get_db, save_upload
from app.models.enums import ExpiryStatus
from app.models.expiry import ExpiryAudit
from app.models.product import Product
from app.schemas.expiry import (
    ExpiryAuditRead,
    ExpiryExtraction,
    ExpiryExtractRequest,
    ExpiryExtractResponse,
    ExpirySummary,
)
from app.services import ocr_expiry

router = APIRouter()


@router.post("/parse", response_model=ExpiryExtractResponse)
def parse_texts(
    payload: ExpiryExtractRequest, db: Session = Depends(get_db)
) -> ExpiryExtractResponse:
    """Regex/normalisation only — no OCR. Used for eval and manual correction."""
    started = time.perf_counter()
    extractions = ocr_expiry.parse_texts(payload.texts, payload.reference_date)

    session_uid = payload.session_uid
    if payload.persist:
        session = create_session(db)
        session_uid = session.session_uid
        _persist(db, session.id, extractions, payload.product_sku)
        complete_session(db, session)

    return _to_response(session_uid, extractions, (time.perf_counter() - started) * 1000.0)


@router.post("/extract/image", response_model=ExpiryExtractResponse)
async def extract_from_image(
    file: UploadFile = File(...),
    product_sku: Optional[str] = Form(default=None),
    reference_date: Optional[date] = Form(default=None),
    db: Session = Depends(get_db),
) -> ExpiryExtractResponse:
    started = time.perf_counter()
    image_path = await save_upload(file)

    service = ocr_expiry.get_ocr_service()
    extractions, _ = service.read_image(str(image_path), reference_date)
    if not service.is_ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "EasyOCR unavailable — install requirements-ml.txt",
        )

    session = create_session(db, image_path=image_path)
    _persist(db, session.id, extractions, product_sku)
    latency_ms = (time.perf_counter() - started) * 1000.0
    complete_session(db, session, total_latency_ms=latency_ms)
    return _to_response(session.session_uid, extractions, latency_ms)


@router.get("/audits", response_model=List[ExpiryAuditRead])
def list_audits(
    db: Session = Depends(get_db),
    page: Pagination = Depends(),
    expiry_status: Optional[ExpiryStatus] = Query(default=None, alias="status"),
):
    stmt = select(ExpiryAudit).order_by(ExpiryAudit.created_at.desc(), ExpiryAudit.id.desc())
    if expiry_status:
        stmt = stmt.where(ExpiryAudit.status == expiry_status)
    return list(db.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all())


@router.get("/summary", response_model=ExpirySummary)
def summary(db: Session = Depends(get_db)) -> ExpirySummary:
    audits = db.execute(select(ExpiryAudit)).scalars().all()
    total = len(audits)
    counts = dict.fromkeys(ExpiryStatus, 0)
    for audit in audits:
        counts[audit.status] += 1
    readable = total - counts[ExpiryStatus.UNREADABLE]
    return ExpirySummary(
        total_scanned=total,
        valid=counts[ExpiryStatus.VALID],
        near_expiry=counts[ExpiryStatus.NEAR_EXPIRY],
        expired=counts[ExpiryStatus.EXPIRED],
        unreadable=counts[ExpiryStatus.UNREADABLE],
        read_rate=(readable / total) if total else 0.0,
    )


def _persist(
    db: Session,
    session_id: Optional[int],
    extractions: List[ExpiryExtraction],
    product_sku: Optional[str],
) -> None:
    if not extractions:
        return
    product_id = None
    if product_sku:
        product = db.execute(
            select(Product).where(Product.sku == product_sku)
        ).scalar_one_or_none()
        product_id = product.id if product else None

    db.add_all(
        ExpiryAudit(
            session_id=session_id,
            product_id=product_id,
            raw_text=item.raw_text,
            normalized_text=item.normalized_text,
            matched_pattern=item.matched_pattern,
            parsed_date=item.parsed_date,
            days_remaining=item.days_remaining,
            status=item.status,
            ocr_confidence=item.ocr_confidence,
            bbox=list(item.bbox.as_tuple()) if item.bbox else None,
            latency_ms=item.latency_ms,
        )
        for item in extractions
    )
    db.flush()


def _to_response(
    session_uid: Optional[str], extractions: List[ExpiryExtraction], latency_ms: float
) -> ExpiryExtractResponse:
    return ExpiryExtractResponse(
        session_uid=session_uid,
        extractions=extractions,
        expired_count=sum(1 for e in extractions if e.status is ExpiryStatus.EXPIRED),
        near_expiry_count=sum(1 for e in extractions if e.status is ExpiryStatus.NEAR_EXPIRY),
        unreadable_count=sum(1 for e in extractions if e.status is ExpiryStatus.UNREADABLE),
        latency_ms=round(latency_ms, 2),
    )
