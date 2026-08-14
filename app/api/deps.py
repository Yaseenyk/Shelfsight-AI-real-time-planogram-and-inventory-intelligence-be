"""Shared FastAPI dependencies and request helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.enums import ScanStatus
from app.models.scan import ScanSession

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class Pagination:
    def __init__(
        self,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> None:
        self.limit = limit
        self.offset = offset


async def save_upload(file: UploadFile) -> Path:
    """Persist an uploaded frame under `data/uploads/` and return its path."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {file.content_type}",
        )

    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds the 20 MB limit",
        )

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "frame.jpg").suffix or ".jpg"
    target = settings.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(payload)
    return target


def create_session(
    db: Session,
    shelf_id: Optional[str] = None,
    store_id: Optional[str] = None,
    image_path: Optional[Path] = None,
) -> ScanSession:
    session = ScanSession(
        session_uid=str(uuid.uuid4()),
        shelf_id=shelf_id,
        store_id=store_id,
        image_path=str(image_path) if image_path else None,
        status=ScanStatus.PROCESSING,
    )
    db.add(session)
    db.flush()
    return session


def complete_session(
    db: Session,
    session: ScanSession,
    total_latency_ms: Optional[float] = None,
    detector_version: Optional[str] = None,
) -> ScanSession:
    session.status = ScanStatus.COMPLETED
    session.completed_at = datetime.now(timezone.utc)
    session.total_latency_ms = total_latency_ms
    session.detector_version = detector_version
    db.flush()
    return session


__all__ = ["Pagination", "complete_session", "create_session", "get_db", "save_upload"]
