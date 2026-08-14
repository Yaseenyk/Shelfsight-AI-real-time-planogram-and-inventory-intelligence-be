"""`/api/v1/planogram` — layout CRUD and spatial compliance validation."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Pagination, complete_session, create_session, get_db, save_upload
from app.models.compliance import ComplianceAudit
from app.models.enums import ScanStatus
from app.models.planogram import PlanogramLayout
from app.schemas.common import Detection
from app.schemas.planogram import (
    ComplianceAuditRead,
    ComplianceCheckRequest,
    ComplianceCheckResponse,
    DetectionSummary,
    PlanogramCreate,
    PlanogramDetail,
    PlanogramRead,
    PlanogramVerifyResponse,
)
from app.services import planogram_store
from app.services.class_map import resolve_detections
from app.services.compliance import ComplianceEngine, ComplianceResult
from app.services.detection import (
    DetectionError,
    DetectionResult,
    DetectorUnavailableError,
    get_detector,
)
from app.utils.vision import ImageDecodeError, decode_image_bytes

router = APIRouter()


# --- layouts ---------------------------------------------------------------
@router.get("/layouts", response_model=List[PlanogramRead])
def list_layouts(db: Session = Depends(get_db), page: Pagination = Depends()):
    stmt = select(PlanogramLayout).order_by(PlanogramLayout.updated_at.desc())
    return list(db.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all())


@router.post("/layouts", response_model=PlanogramDetail, status_code=status.HTTP_201_CREATED)
def upsert_layout(payload: PlanogramCreate, db: Session = Depends(get_db)) -> PlanogramLayout:
    """Create or version-bump a planogram. The document is schema-validated."""
    return planogram_store.upsert(db, payload.document, is_active=payload.is_active)


@router.get("/layouts/{planogram_uid}", response_model=PlanogramDetail)
def get_layout(planogram_uid: str, db: Session = Depends(get_db)) -> PlanogramLayout:
    layout = db.execute(
        select(PlanogramLayout).where(PlanogramLayout.planogram_uid == planogram_uid)
    ).scalar_one_or_none()
    if layout is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown planogram {planogram_uid}")
    return layout


@router.delete("/layouts/{planogram_uid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_layout(planogram_uid: str, db: Session = Depends(get_db)) -> None:
    layout = db.execute(
        select(PlanogramLayout).where(PlanogramLayout.planogram_uid == planogram_uid)
    ).scalar_one_or_none()
    if layout is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown planogram {planogram_uid}")
    db.delete(layout)


# --- compliance ------------------------------------------------------------
@router.post("/compliance", response_model=ComplianceCheckResponse)
def check_compliance(
    payload: ComplianceCheckRequest, db: Session = Depends(get_db)
) -> ComplianceCheckResponse:
    layout = _resolve_layout(db, payload.planogram_uid, payload.shelf_id)
    document = planogram_store.to_document(layout)
    if payload.tolerances is not None:
        document.tolerances = payload.tolerances

    session = create_session(db, shelf_id=payload.shelf_id)
    engine = ComplianceEngine()
    # Resolve detector class names to catalogue SKUs here too, not just on the
    # image path. Without it a caller posting raw detector output (`bottle`,
    # `banana`) can never satisfy a slot, because the engine matches on SKU —
    # every slot would read MISSING on a correctly stocked shelf.
    detections = resolve_detections(payload.detections, db=db)
    result = engine.evaluate(document, detections, shelf_id=payload.shelf_id)
    audit = _persist(db, layout, session.id, payload.shelf_id, result, payload.persist)
    complete_session(db, session, total_latency_ms=result.latency_ms)
    return _to_response(audit, result)


@router.post("/verify", response_model=PlanogramVerifyResponse)
async def verify_shelf(
    file: UploadFile = File(..., description="Shelf frame (JPEG/PNG/WebP/BMP)"),
    planogram_id: Optional[str] = Form(
        default=None, description="Target planogram UID; defaults to the active layout"
    ),
    shelf_id: Optional[str] = Form(default=None),
    confidence: Optional[float] = Form(
        default=None, ge=0.0, le=1.0, description="Override the detector threshold"
    ),
    db: Session = Depends(get_db),
) -> PlanogramVerifyResponse:
    """Full Phase 1 pipeline: image → YOLOv8 → SKU resolution → compliance.

    1. Decode the upload with OpenCV (bad bytes → 422, never a silent empty result).
    2. Run the detector, normalising every box into the `BoundingBox` schema.
    3. Map detector classes onto catalogue SKUs.
    4. Score each planogram slot: COMPLIANT / MISPLACED / MISSING, plus EXTRA
       detections that belong to no slot.

    The scan session is marked `FAILED` on any pipeline error, so a failed frame
    is auditable rather than absent.
    """
    started = time.perf_counter()
    payload = await file.read()
    await file.seek(0)

    try:
        frame = decode_image_bytes(payload)
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    image_path = await save_upload(file)
    layout = _resolve_layout(db, planogram_id, shelf_id)
    document = planogram_store.to_document(layout)
    session = create_session(db, shelf_id=shelf_id, image_path=image_path)

    detector = get_detector()
    try:
        detection_result = detector.predict_with_metrics(frame, conf=confidence)
    except DetectorUnavailableError as exc:
        _fail_session(db, session, str(exc))
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except DetectionError as exc:
        _fail_session(db, session, str(exc))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    detections = resolve_detections(detection_result.detections, db=db)

    session.image_width = detection_result.image_width
    session.image_height = detection_result.image_height
    session.detection_latency_ms = detection_result.latency_ms
    session.detections = [d.model_dump(mode="json") for d in detections]

    compliance = ComplianceEngine().evaluate(document, detections, shelf_id=shelf_id)
    audit = _persist(db, layout, session.id, shelf_id, compliance, persist=True)

    total_ms = (time.perf_counter() - started) * 1000.0
    complete_session(db, session, total_latency_ms=total_ms, detector_version=detector.version)

    response = PlanogramVerifyResponse.model_validate(
        {
            **_to_response(audit, compliance).model_dump(),
            "session_uid": session.session_uid,
            "planogram_uid": layout.planogram_uid,
            "detections": detections,
            "detection": _detection_summary(detection_result, detections),
            "detection_latency_ms": round(detection_result.latency_ms, 2),
            "total_latency_ms": round(total_ms, 2),
        }
    )
    return response


@router.post(
    "/compliance/image",
    response_model=PlanogramVerifyResponse,
    deprecated=True,
    summary="Deprecated alias for POST /planogram/verify",
)
async def check_compliance_from_image(
    file: UploadFile = File(...),
    planogram_uid: Optional[str] = Form(default=None),
    shelf_id: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
) -> PlanogramVerifyResponse:
    """Phase 0 route name, kept so existing clients do not break."""
    return await verify_shelf(
        file=file, planogram_id=planogram_uid, shelf_id=shelf_id, confidence=None, db=db
    )


@router.get("/audits", response_model=List[ComplianceAuditRead])
def list_audits(
    db: Session = Depends(get_db),
    page: Pagination = Depends(),
    shelf_id: Optional[str] = Query(default=None),
):
    stmt = select(ComplianceAudit).order_by(
        ComplianceAudit.created_at.desc(), ComplianceAudit.id.desc()
    )
    if shelf_id:
        stmt = stmt.where(ComplianceAudit.shelf_id == shelf_id)
    return list(db.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all())


@router.get("/audits/latest", response_model=Optional[ComplianceCheckResponse])
def latest_audit(db: Session = Depends(get_db), shelf_id: Optional[str] = Query(default=None)):
    stmt = select(ComplianceAudit).order_by(
        ComplianceAudit.created_at.desc(), ComplianceAudit.id.desc()
    )
    if shelf_id:
        stmt = stmt.where(ComplianceAudit.shelf_id == shelf_id)
    audit = db.execute(stmt.limit(1)).scalar_one_or_none()
    if audit is None:
        return None
    response = ComplianceCheckResponse.model_validate(audit)
    response.slot_results = audit.slot_results or []
    return response


# --- helpers ---------------------------------------------------------------
def _resolve_layout(
    db: Session, planogram_uid: Optional[str], shelf_id: Optional[str]
) -> PlanogramLayout:
    if planogram_uid:
        layout = db.execute(
            select(PlanogramLayout).where(PlanogramLayout.planogram_uid == planogram_uid)
        ).scalar_one_or_none()
        if layout is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown planogram {planogram_uid}")
        return layout

    layout = planogram_store.get_active(db, shelf_id)
    if layout is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No active planogram configured — POST one to /api/v1/planogram/layouts",
        )
    return layout


def _persist(
    db: Session,
    layout: PlanogramLayout,
    session_id: Optional[int],
    shelf_id: Optional[str],
    result: ComplianceResult,
    persist: bool,
) -> ComplianceAudit:
    audit = ComplianceAudit(
        session_id=session_id,
        planogram_id=layout.id,
        shelf_id=shelf_id,
        total_slots=result.total_slots,
        compliant_slots=result.compliant_slots,
        misplaced_slots=result.misplaced_slots,
        missing_slots=result.missing_slots,
        extra_detections=result.extra_detections,
        compliance_score=result.compliance_score,
        spatial_alignment_accuracy=result.spatial_alignment_accuracy,
        mean_iou=result.mean_iou,
        mean_center_distance=result.mean_center_distance,
        false_positive_rate=result.false_positive_rate,
        latency_ms=result.latency_ms,
        slot_results=[s.model_dump(mode="json") for s in result.slot_results],
    )
    if persist:
        db.add(audit)
        db.flush()
    else:
        # Transient preview: `created_at` is a server default that only fires on
        # INSERT, so stamp it here rather than returning a null timestamp.
        audit.created_at = datetime.now(timezone.utc)
    return audit


def _to_response(audit: ComplianceAudit, result: ComplianceResult) -> ComplianceCheckResponse:
    payload = ComplianceCheckResponse.model_validate(audit)
    payload.slot_results = result.slot_results
    return payload


def _fail_session(db: Session, session, message: str) -> None:  # noqa: ANN001
    """Record why a frame died so failures are auditable, not just missing."""
    session.status = ScanStatus.FAILED
    session.error_message = message[:1024]
    db.flush()


def _detection_summary(
    result: DetectionResult, resolved: List[Detection]
) -> DetectionSummary:
    confidences = [d.confidence for d in resolved]
    mapped = sum(1 for d in resolved if d.sku)
    return DetectionSummary(
        count=len(resolved),
        resolved_skus=mapped,
        unresolved=len(resolved) - mapped,
        suppressed=result.suppressed,
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
        class_counts=result.class_counts,
        model_version=result.model_version,
        image_width=result.image_width,
        image_height=result.image_height,
        inference_ms=round(result.inference_ms, 2),
        postprocess_ms=round(result.postprocess_ms, 2),
    )
