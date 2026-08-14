"""`/api/v1/expiry` — packaging OCR, date normalisation and validity status."""

from __future__ import annotations

import time
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Pagination, complete_session, create_session, get_db, save_upload
from app.models.enums import ExpiryStatus, ScanStatus
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
from app.services.ocr_expiry import ExpiryReadResult, OCRError, OCRUnavailableError
from app.utils.vision import ImageDecodeError, decode_image_bytes

router = APIRouter()


@router.post("/extract", response_model=ExpiryExtractResponse)
async def extract_expiry(
    file: UploadFile = File(..., description="Packaging crop showing the date panel"),
    product_sku: Optional[str] = Form(default=None),
    reference_date: Optional[date] = Form(
        default=None, description="Evaluate validity against this date instead of today"
    ),
    persist: bool = Form(default=True),
    db: Session = Depends(get_db),
) -> ExpiryExtractResponse:
    """OCR a packaging crop and extract its expiry date.

    Returns the raw OCR text, the normalised text, the matched date pattern, the
    parsed date, days remaining and the validity verdict
    (`valid` / `near_expiry` / `expired` / `unreadable`).

    **An unreadable stamp is a 200, not an error.** "No date found" is a
    legitimate audit outcome that the dashboard must be able to show; only a
    broken OCR engine produces a 5xx.
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

    service = ocr_expiry.get_ocr_service()
    try:
        result = service.extract_expiry(frame, reference_date=reference_date)
    except OCRUnavailableError as exc:
        _fail_session(db, session, str(exc))
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except OCRError as exc:
        _fail_session(db, session, str(exc))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    session.image_width = result.image_width
    session.image_height = result.image_height

    extractions = result.extractions or [_unreadable(result)]
    if persist:
        _persist(db, session.id, extractions, product_sku)

    latency_ms = (time.perf_counter() - started) * 1000.0
    complete_session(db, session, total_latency_ms=latency_ms)
    return _to_response(session.session_uid, extractions, latency_ms, result)


@router.post("/parse", response_model=ExpiryExtractResponse)
def parse_texts(
    payload: ExpiryExtractRequest, db: Session = Depends(get_db)
) -> ExpiryExtractResponse:
    """Regex/normalisation only — no OCR. Used for evaluation and manual correction."""
    started = time.perf_counter()
    extractions = ocr_expiry.parse_texts(payload.texts, payload.reference_date)

    session_uid = payload.session_uid
    if payload.persist:
        session = create_session(db)
        session_uid = session.session_uid
        _persist(db, session.id, extractions, payload.product_sku)
        complete_session(db, session)

    return _to_response(session_uid, extractions, (time.perf_counter() - started) * 1000.0)


@router.post(
    "/extract/image",
    response_model=ExpiryExtractResponse,
    deprecated=True,
    summary="Deprecated alias for POST /expiry/extract",
)
async def extract_from_image(
    file: UploadFile = File(...),
    product_sku: Optional[str] = Form(default=None),
    reference_date: Optional[date] = Form(default=None),
    db: Session = Depends(get_db),
) -> ExpiryExtractResponse:
    """Phase 0 route name, kept so existing clients do not break."""
    return await extract_expiry(
        file=file,
        product_sku=product_sku,
        reference_date=reference_date,
        persist=True,
        db=db,
    )


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


# ------------------------------------------------------------------ helpers --
def _fail_session(db: Session, session, message: str) -> None:  # noqa: ANN001
    session.status = ScanStatus.FAILED
    session.error_message = message[:1024]
    db.flush()


def _unreadable(result: ExpiryReadResult) -> ExpiryExtraction:
    """Placeholder row for a crop where OCR found nothing parseable.

    Recording it keeps the read-rate denominator honest: a package whose stamp
    could not be read is a data point, not a gap in the audit trail.
    """
    return ExpiryExtraction(
        raw_text=result.raw_text or None,
        normalized_text=None,
        matched_pattern=None,
        parsed_date=None,
        days_remaining=None,
        status=ExpiryStatus.UNREADABLE,
        ocr_confidence=None,
        latency_ms=result.latency_ms,
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
    session_uid: Optional[str],
    extractions: List[ExpiryExtraction],
    latency_ms: float,
    result: Optional[ExpiryReadResult] = None,
) -> ExpiryExtractResponse:
    return ExpiryExtractResponse(
        session_uid=session_uid,
        extractions=extractions,
        expired_count=sum(1 for e in extractions if e.status is ExpiryStatus.EXPIRED),
        near_expiry_count=sum(1 for e in extractions if e.status is ExpiryStatus.NEAR_EXPIRY),
        unreadable_count=sum(1 for e in extractions if e.status is ExpiryStatus.UNREADABLE),
        latency_ms=round(latency_ms, 2),
        best=result.best if result else None,
        raw_text=result.raw_text if result else None,
        variant_used=result.variant_used if result else None,
        variants_tried=result.variants_tried if result else [],
        ocr_ms=round(result.ocr_ms, 2) if result else None,
    )
