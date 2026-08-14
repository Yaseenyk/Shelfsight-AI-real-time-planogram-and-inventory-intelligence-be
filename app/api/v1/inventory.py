"""`/api/v1/inventory` — detection-driven stock reconciliation (phantom inventory)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Pagination, complete_session, create_session, get_db, save_upload
from app.models.enums import DiscrepancyType
from app.models.inventory import InventoryLog
from app.models.product import Product
from app.schemas.inventory import (
    InventoryLogRead,
    InventoryScanRequest,
    InventoryScanResponse,
    InventorySummary,
)
from app.schemas.product import ProductCreate, ProductRead, ProductUpdate
from app.services import inventory as inventory_service
from app.services import planogram_store
from app.services.class_map import resolve_detections
from app.services.detection import DetectionError, DetectorUnavailableError, get_detector
from app.utils.vision import ImageDecodeError, decode_image_bytes

router = APIRouter()


# --- catalogue ------------------------------------------------------------
@router.get("/products", response_model=List[ProductRead])
def list_products(
    db: Session = Depends(get_db),
    page: Pagination = Depends(),
    category: Optional[str] = Query(default=None),
) -> List[Product]:
    stmt = select(Product).order_by(Product.sku)
    if category:
        stmt = stmt.where(Product.category == category)
    return list(db.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all())


@router.post("/products", response_model=ProductRead, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)) -> Product:
    exists = db.execute(select(Product).where(Product.sku == payload.sku)).scalar_one_or_none()
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, f"SKU {payload.sku} already exists")
    product = Product(**payload.model_dump())
    db.add(product)
    db.flush()
    return product


@router.patch("/products/{sku}", response_model=ProductRead)
def update_product(sku: str, payload: ProductUpdate, db: Session = Depends(get_db)) -> Product:
    product = db.execute(select(Product).where(Product.sku == sku)).scalar_one_or_none()
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown SKU {sku}")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(product, field, value)
    db.flush()
    return product


# --- reconciliation --------------------------------------------------------
@router.post("/scan", response_model=InventoryScanResponse)
def scan_from_detections(
    payload: InventoryScanRequest, db: Session = Depends(get_db)
) -> InventoryScanResponse:
    """Reconcile a caller-supplied detection set (offline replay / evaluation)."""
    started = time.perf_counter()
    session = create_session(db, shelf_id=payload.shelf_id, store_id=payload.store_id)
    logs, discrepancies = inventory_service.reconcile(
        db,
        payload.detections,
        session_id=session.id,
        shelf_id=payload.shelf_id,
        persist=payload.persist,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    complete_session(db, session, total_latency_ms=latency_ms)
    return _to_response(session.session_uid, payload.shelf_id, logs, discrepancies, latency_ms)


@router.post("/scan/image", response_model=InventoryScanResponse)
async def scan_from_image(
    file: UploadFile = File(...),
    shelf_id: Optional[str] = Form(default=None),
    store_id: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
) -> InventoryScanResponse:
    """Full path: upload a frame, run YOLOv8, then reconcile against SQLite."""
    started = time.perf_counter()
    payload = await file.read()
    await file.seek(0)

    try:
        frame = decode_image_bytes(payload)
    except ImageDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    image_path = await save_upload(file)
    session = create_session(db, shelf_id=shelf_id, store_id=store_id, image_path=image_path)

    detector = get_detector()
    try:
        result = detector.predict_with_metrics(frame)
    except DetectorUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except DetectionError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    detections = resolve_detections(result.detections, db=db)
    session.image_width = result.image_width
    session.image_height = result.image_height
    session.detection_latency_ms = result.latency_ms
    session.detections = [d.model_dump(mode="json") for d in detections]

    # The planogram defines what this shelf *should* hold; without it a scoped
    # scan could only ever report on products it happened to detect.
    logs, discrepancies = inventory_service.reconcile(
        db,
        detections,
        session_id=session.id,
        shelf_id=shelf_id,
        persist=True,
        expected_skus=planogram_store.expected_skus(db, shelf_id) if shelf_id else None,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    complete_session(db, session, total_latency_ms=latency_ms, detector_version=detector.version)
    return _to_response(
        session.session_uid,
        shelf_id,
        logs,
        discrepancies,
        latency_ms,
        objects_detected=len(result.detections),
        unresolved=sum(1 for d in detections if not getattr(d, 'sku', None)),
    )


@router.get("/summary", response_model=InventorySummary)
def summary(db: Session = Depends(get_db)) -> InventorySummary:
    return inventory_service.build_summary(db)


@router.get("/logs", response_model=List[InventoryLogRead])
def list_logs(
    db: Session = Depends(get_db),
    page: Pagination = Depends(),
    discrepancy_type: Optional[DiscrepancyType] = Query(default=None),
    shelf_id: Optional[str] = Query(default=None),
) -> List[InventoryLog]:
    stmt = select(InventoryLog).order_by(InventoryLog.created_at.desc(), InventoryLog.id.desc())
    if discrepancy_type:
        stmt = stmt.where(InventoryLog.discrepancy_type == discrepancy_type)
    if shelf_id:
        stmt = stmt.where(InventoryLog.shelf_id == shelf_id)
    return list(db.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all())


@router.get("/alerts", response_model=List[InventoryLogRead])
def list_alerts(db: Session = Depends(get_db), limit: int = Query(default=20, ge=1, le=100)):
    """Most recent non-matching rows — feeds the real-time alert panel."""
    stmt = (
        select(InventoryLog)
        .where(InventoryLog.discrepancy_type != DiscrepancyType.MATCH)
        .order_by(InventoryLog.created_at.desc(), InventoryLog.id.desc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def _to_response(
    session_uid: str,
    shelf_id: Optional[str],
    logs: List[InventoryLog],
    discrepancies: List,
    latency_ms: float,
    objects_detected: int = 0,
    unresolved: int = 0,
) -> InventoryScanResponse:
    return InventoryScanResponse(
        objects_detected=objects_detected,
        unresolved_detections=unresolved,
        session_uid=session_uid,
        shelf_id=shelf_id,
        total_detected=sum(row.detected_count for row in logs),
        total_system=sum(row.system_count for row in logs),
        matched_skus=sum(1 for row in logs if row.discrepancy_type is DiscrepancyType.MATCH),
        discrepancies=discrepancies,
        phantom_count=sum(
            1 for row in logs if row.discrepancy_type is DiscrepancyType.PHANTOM
        ),
        latency_ms=round(latency_ms, 2),
        created_at=datetime.now(timezone.utc),
    )
