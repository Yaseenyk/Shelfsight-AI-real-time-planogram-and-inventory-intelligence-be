"""Telemetry → prompt compilation.

Two scopes:

- **session** — one capture (`ScanSession`) and the audit rows that hang off it.
  This is the "explain this scan" button.
- **window** — everything in the last N hours, optionally one shelf. This is the
  "how is the store doing" briefing.

Both compile to the same `InsightContext`, so the prompt builder has one input
shape and the paper has one documented payload. `GET /insights/context` returns
this object verbatim: whatever the model saw is reproducible after the fact,
which is the difference between a citable result and an anecdote.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.compliance import ComplianceAudit
from app.models.enums import DiscrepancyType, ExpiryStatus, FreshnessLabel
from app.models.expiry import ExpiryAudit
from app.models.freshness import FreshnessAudit
from app.models.inventory import InventoryLog
from app.models.product import Product
from app.models.scan import ScanSession
from app.schemas.insights import (
    Audience,
    ComplianceMetrics,
    DiscrepancyRow,
    ExpiryMetrics,
    FreshnessMetrics,
    InsightContext,
    InventoryMetrics,
    PromptBundle,
)

logger = get_logger(__name__)

#: Cap the rows shown to the model. A 3B model handed 200 SKUs writes about the
#: last one it read; the top offenders by value are what a manager acts on.
TOP_DISCREPANCY_LIMIT = 8

AUDIENCE_BRIEF: Dict[str, str] = {
    "store_manager": (
        "a store manager on the shop floor. Be concrete and physical: which aisle, "
        "which SKU, what to pick up or restock in the next hour."
    ),
    "regional_director": (
        "a regional director. Lead with financial exposure and repeat offenders; "
        "they will not walk the aisle themselves."
    ),
    "analyst": (
        "a retail analyst. Be precise about the numbers and note where the "
        "measurement itself is uncertain."
    ),
}

SYSTEM_PROMPT = """You are ShelfSight AI, a retail shelf-audit analyst.

You receive JSON telemetry from a computer-vision pipeline that inspects store
shelves. You are writing for {audience_brief}

What the terms mean (use them exactly this way):
- phantom: the system says stock exists but the camera saw NONE on the shelf.
  The shelf is EMPTY. This is a stockout plus a bad inventory record.
- undercount: fewer units on the shelf than the system claims. Partially empty.
- overcount: MORE units on the shelf than the system claims.
- missing slot: a planogram position with no product in it.
- misplaced slot: a planogram position holding the WRONG product.
- negative "discrepancy" means the shelf has fewer units than the system record.
- value_impact is money at risk in that row, already computed. Never recompute it.

Hard rules:
- Use ONLY numbers present in the telemetry. Never invent a SKU, a count or a price.
- If a metric is zero or absent, do not mention it as if it were a problem.
- Lead with the single most costly issue.
- Give exactly {max_actions} action items, most urgent first.
- severity must be one of: "critical", "warning", "info".

Return STRICT JSON only — no markdown, no code fences, no commentary:
{{"headline": "<= 15 words",
  "summary": "2-4 sentences",
  "actions": [{{"title": "<= 12 words", "rationale": "one sentence citing a number", "severity": "critical|warning|info"}}]}}"""

USER_PROMPT = """Shelf telemetry (JSON):
{context}

{closing}"""

CLEAN_SHELF_NOTE = (
    "This shelf shows no discrepancies. Say so plainly and briefly; do not "
    "manufacture problems. Return an empty actions list if nothing needs doing."
)
FINDINGS_NOTE = "Write the briefing now. JSON only."


# ------------------------------------------------------------ context build --
def build_session_context(db: Session, session_uid: str) -> Optional[InsightContext]:
    """Compile the telemetry for one capture, or None if the session is unknown."""
    session = db.execute(
        select(ScanSession).where(ScanSession.session_uid == session_uid)
    ).scalar_one_or_none()
    if session is None:
        return None

    inventory_logs = db.execute(
        select(InventoryLog).where(InventoryLog.session_id == session.id)
    ).scalars().all()
    audits = db.execute(
        select(ComplianceAudit).where(ComplianceAudit.session_id == session.id)
    ).scalars().all()
    freshness_rows = db.execute(
        select(FreshnessAudit).where(FreshnessAudit.session_id == session.id)
    ).scalars().all()
    expiry_rows = db.execute(
        select(ExpiryAudit).where(ExpiryAudit.session_id == session.id)
    ).scalars().all()

    return InsightContext(
        generated_at=datetime.now(timezone.utc),
        scope="session",
        session_uid=session.session_uid,
        shelf_id=session.shelf_id,
        inventory=_inventory_metrics(inventory_logs),
        compliance=_compliance_metrics(audits),
        freshness=_freshness_metrics(freshness_rows),
        expiry=_expiry_metrics(expiry_rows),
        top_discrepancies=_top_discrepancies(db, inventory_logs),
    )


def build_window_context(
    db: Session, shelf_id: Optional[str] = None, window_hours: int = 24
) -> InsightContext:
    """Compile the telemetry for the last `window_hours` (optionally one shelf)."""
    since = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    log_stmt = select(InventoryLog).where(InventoryLog.created_at >= since)
    if shelf_id:
        log_stmt = log_stmt.where(InventoryLog.shelf_id == shelf_id)
    inventory_logs = db.execute(log_stmt).scalars().all()

    audit_stmt = select(ComplianceAudit).where(ComplianceAudit.created_at >= since)
    if shelf_id:
        audit_stmt = audit_stmt.where(ComplianceAudit.shelf_id == shelf_id)
    audits = db.execute(
        audit_stmt.order_by(ComplianceAudit.created_at.desc()).limit(1)
    ).scalars().all()

    freshness_rows = db.execute(
        select(FreshnessAudit).where(FreshnessAudit.created_at >= since)
    ).scalars().all()
    expiry_rows = db.execute(
        select(ExpiryAudit).where(ExpiryAudit.created_at >= since)
    ).scalars().all()

    return InsightContext(
        generated_at=datetime.now(timezone.utc),
        scope="window",
        shelf_id=shelf_id,
        window_hours=window_hours,
        inventory=_inventory_metrics(inventory_logs),
        compliance=_compliance_metrics(audits),
        freshness=_freshness_metrics(freshness_rows),
        expiry=_expiry_metrics(expiry_rows),
        top_discrepancies=_top_discrepancies(db, inventory_logs),
    )


def _inventory_metrics(logs: Sequence[InventoryLog]) -> InventoryMetrics:
    """Latest row per product, so a re-scanned SKU is not counted twice."""
    latest: Dict[int, InventoryLog] = {}
    for log in sorted(logs, key=lambda row: (row.created_at, row.id), reverse=True):
        latest.setdefault(log.product_id, log)
    rows = list(latest.values())
    if not rows:
        return InventoryMetrics()

    matched = sum(1 for r in rows if r.discrepancy_type is DiscrepancyType.MATCH)
    return InventoryMetrics(
        total_skus=len(rows),
        phantom_skus=sum(1 for r in rows if r.discrepancy_type is DiscrepancyType.PHANTOM),
        undercount_skus=sum(1 for r in rows if r.discrepancy_type is DiscrepancyType.UNDERCOUNT),
        overcount_skus=sum(1 for r in rows if r.discrepancy_type is DiscrepancyType.OVERCOUNT),
        total_detected=sum(r.detected_count for r in rows),
        total_system=sum(r.system_count for r in rows),
        accuracy_rate=round(matched / len(rows), 4),
    )


def _compliance_metrics(audits: Sequence[ComplianceAudit]) -> ComplianceMetrics:
    if not audits:
        return ComplianceMetrics()
    audit = audits[0]  # most recent
    return ComplianceMetrics(
        total_slots=audit.total_slots,
        compliant_slots=audit.compliant_slots,
        misplaced_slots=audit.misplaced_slots,
        missing_slots=audit.missing_slots,
        extra_detections=audit.extra_detections,
        compliance_score=round(audit.compliance_score, 4),
        spatial_alignment_accuracy=round(audit.spatial_alignment_accuracy, 4),
    )


def _freshness_metrics(rows: Sequence[FreshnessAudit]) -> FreshnessMetrics:
    total = len(rows)
    spoiled = sum(1 for r in rows if r.label is FreshnessLabel.SPOILED)
    return FreshnessMetrics(
        assessed=total,
        fresh=sum(1 for r in rows if r.label is FreshnessLabel.FRESH),
        ripening=sum(1 for r in rows if r.label is FreshnessLabel.RIPENING),
        spoiled=spoiled,
        spoilage_rate=round(spoiled / total, 4) if total else 0.0,
    )


def _expiry_metrics(rows: Sequence[ExpiryAudit]) -> ExpiryMetrics:
    return ExpiryMetrics(
        scanned=len(rows),
        valid=sum(1 for r in rows if r.status is ExpiryStatus.VALID),
        near_expiry=sum(1 for r in rows if r.status is ExpiryStatus.NEAR_EXPIRY),
        expired=sum(1 for r in rows if r.status is ExpiryStatus.EXPIRED),
        unreadable=sum(1 for r in rows if r.status is ExpiryStatus.UNREADABLE),
    )


def _top_discrepancies(
    db: Session, logs: Sequence[InventoryLog], limit: int = TOP_DISCREPANCY_LIMIT
) -> List[DiscrepancyRow]:
    """Non-matching rows ranked by monetary exposure."""
    offenders = [r for r in logs if r.discrepancy_type is not DiscrepancyType.MATCH]
    if not offenders:
        return []

    products = {
        p.id: p
        for p in db.execute(
            select(Product).where(Product.id.in_({r.product_id for r in offenders}))
        ).scalars().all()
    }

    rows: List[DiscrepancyRow] = []
    for log in offenders:
        product = products.get(log.product_id)
        if product is None:
            continue
        rows.append(
            DiscrepancyRow(
                sku=product.sku,
                name=product.name,
                detected=log.detected_count,
                system=log.system_count,
                discrepancy=log.discrepancy,
                type=log.discrepancy_type.value,
                value_impact=round(abs(log.discrepancy) * (product.unit_price or 0.0), 2),
            )
        )
    rows.sort(key=lambda row: row.value_impact, reverse=True)
    return rows[:limit]


# ----------------------------------------------------------- prompt compile --
def compile_prompt(
    context: InsightContext, audience: Audience = "store_manager", max_actions: int = 3
) -> PromptBundle:
    """Turn telemetry into the exact system/user pair sent to Ollama.

    Kept as a pure function of `(context, audience)` so a prompt can be
    regenerated for the paper without a database or a running model.
    """
    payload = context.model_dump(mode="json", exclude_none=True)
    # Drop empty metric blocks: a model shown `{"expired": 0, ...}` for a shelf
    # with no perishables tends to write about expiry anyway.
    payload = {key: value for key, value in payload.items() if not _is_empty(value)}

    system = SYSTEM_PROMPT.format(
        audience_brief=AUDIENCE_BRIEF.get(audience, AUDIENCE_BRIEF["store_manager"]),
        max_actions=max_actions,
    )
    user = USER_PROMPT.format(
        context=json.dumps(payload, indent=2, default=str),
        closing=FINDINGS_NOTE if context.has_findings else CLEAN_SHELF_NOTE,
    )
    return PromptBundle(system=system, user=user, audience=audience)


def _is_empty(value: object) -> bool:
    if isinstance(value, dict):
        return all(_is_zeroish(v) for v in value.values())
    if isinstance(value, list):
        return not value
    return False


def _is_zeroish(value: object) -> bool:
    return value in (0, 0.0, None, "", [], {})
