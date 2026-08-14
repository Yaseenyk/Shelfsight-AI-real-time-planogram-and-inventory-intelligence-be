"""`/api/v1/insights` — executive summaries generated locally via Ollama."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.compliance import ComplianceAudit
from app.models.enums import DiscrepancyType, ExpiryStatus, FreshnessLabel
from app.models.expiry import ExpiryAudit
from app.models.freshness import FreshnessAudit
from app.models.inventory import InventoryLog
from app.models.product import Product
from app.schemas.insights import InsightRequest, InsightResponse, OllamaStatus
from app.services.inventory import build_summary
from app.services.llm_client import get_ollama_client

router = APIRouter()


@router.get("/status", response_model=OllamaStatus)
async def ollama_status() -> OllamaStatus:
    return await get_ollama_client().status()


@router.post("/generate", response_model=InsightResponse)
async def generate_insight(
    payload: InsightRequest, db: Session = Depends(get_db)
) -> InsightResponse:
    context = payload.context or build_context(db, payload.shelf_id, payload.window_hours)
    return await get_ollama_client().generate(
        context=context,
        audience=payload.audience,
        model=payload.model,
        temperature=payload.temperature,
    )


@router.get("/context")
def insight_context(
    db: Session = Depends(get_db),
    shelf_id: Optional[str] = None,
    window_hours: int = 24,
) -> Dict[str, Any]:
    """Expose the exact telemetry sent to the LLM — needed for reproducibility."""
    return build_context(db, shelf_id, window_hours)


def build_context(
    db: Session, shelf_id: Optional[str] = None, window_hours: int = 24
) -> Dict[str, Any]:
    """Assemble a compact, LLM-friendly snapshot of the last `window_hours`."""
    since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    inventory = build_summary(db)

    log_stmt = select(InventoryLog).where(
        InventoryLog.created_at >= since,
        InventoryLog.discrepancy_type != DiscrepancyType.MATCH,
    )
    if shelf_id:
        log_stmt = log_stmt.where(InventoryLog.shelf_id == shelf_id)
    logs = db.execute(log_stmt).scalars().all()

    prices = {
        p.id: (p.unit_price or 0.0, p.sku, p.name)
        for p in db.execute(select(Product)).scalars()
    }
    unknown = (0.0, "?", "?")
    top_offenders = sorted(
        (
            {
                "sku": prices.get(row.product_id, unknown)[1],
                "name": prices.get(row.product_id, unknown)[2],
                "detected": row.detected_count,
                "system": row.system_count,
                "discrepancy": row.discrepancy,
                "type": row.discrepancy_type.value,
                "value_impact": round(
                    abs(row.discrepancy) * prices.get(row.product_id, unknown)[0], 2
                ),
            }
            for row in logs
        ),
        key=lambda row: row["value_impact"],
        reverse=True,
    )[:10]

    audit_stmt = select(ComplianceAudit).where(ComplianceAudit.created_at >= since)
    if shelf_id:
        audit_stmt = audit_stmt.where(ComplianceAudit.shelf_id == shelf_id)
    audits = db.execute(
        audit_stmt.order_by(ComplianceAudit.created_at.desc()).limit(1)
    ).scalars().all()
    latest_audit = audits[0] if audits else None

    freshness_rows = db.execute(
        select(FreshnessAudit).where(FreshnessAudit.created_at >= since)
    ).scalars().all()
    expiry_rows = db.execute(
        select(ExpiryAudit).where(ExpiryAudit.created_at >= since)
    ).scalars().all()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": window_hours,
        "shelf_id": shelf_id,
        "inventory": {
            "total_products": inventory.total_products,
            "total_system_stock": inventory.total_system_stock,
            "total_detected_stock": inventory.total_detected_stock,
            "phantom_skus": inventory.phantom_skus,
            "undercount_skus": inventory.undercount_skus,
            "overcount_skus": inventory.overcount_skus,
            "accuracy_rate": round(inventory.accuracy_rate, 4),
            "value_at_risk": inventory.value_at_risk,
        },
        "top_discrepancies": top_offenders,
        "compliance": (
            {
                "compliance_score": round(latest_audit.compliance_score, 4),
                "spatial_alignment_accuracy": round(latest_audit.spatial_alignment_accuracy, 4),
                "misplaced_slots": latest_audit.misplaced_slots,
                "missing_slots": latest_audit.missing_slots,
                "extra_detections": latest_audit.extra_detections,
            }
            if latest_audit
            else {}
        ),
        "freshness": {
            "assessed": len(freshness_rows),
            "spoiled": sum(1 for r in freshness_rows if r.label is FreshnessLabel.SPOILED),
            "ripening": sum(1 for r in freshness_rows if r.label is FreshnessLabel.RIPENING),
        },
        "expiry": {
            "scanned": len(expiry_rows),
            "expired": sum(1 for r in expiry_rows if r.status is ExpiryStatus.EXPIRED),
            "near_expiry": sum(1 for r in expiry_rows if r.status is ExpiryStatus.NEAR_EXPIRY),
        },
    }
