"""Phantom-inventory reconciliation.

Detections are aggregated per SKU (facings counted, confidence averaged) and
compared against `Product.system_stock`. The classification rule lives on
`InventoryLog.classify` so the API, the seeder and the benchmark agree.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.enums import DiscrepancyType
from app.models.inventory import InventoryLog
from app.models.product import Product
from app.schemas.common import Detection
from app.schemas.inventory import DiscrepancyItem, InventorySummary


def aggregate_detections(
    detections: Sequence[Detection],
) -> Dict[str, Tuple[int, float]]:
    """Collapse detections into `{sku_or_class: (facings, mean_confidence)}`."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for det in detections:
        buckets[det.sku or det.class_name].append(det.confidence)
    return {key: (len(vals), sum(vals) / len(vals)) for key, vals in buckets.items()}


def resolve_products(db: Session, keys: Sequence[str]) -> Dict[str, Product]:
    """Map detector output to catalogue rows by SKU first, then class name."""
    if not keys:
        return {}
    stmt = select(Product).where(
        (Product.sku.in_(keys)) | (Product.detection_class_name.in_(keys))
    )
    resolved: Dict[str, Product] = {}
    for product in db.execute(stmt).scalars().all():
        resolved[product.sku] = product
        if product.detection_class_name:
            resolved.setdefault(product.detection_class_name, product)
    return resolved


def reconcile(
    db: Session,
    detections: Sequence[Detection],
    session_id: Optional[int] = None,
    shelf_id: Optional[str] = None,
    persist: bool = True,
    expected_skus: Optional[Sequence[str]] = None,
) -> Tuple[List[InventoryLog], List[DiscrepancyItem]]:
    """Compare detected facings with system stock and build the audit rows.

    Products that exist in the catalogue but were *not* detected at all are
    included with `detected_count=0` — that is precisely the phantom-inventory
    case, so it must not be skipped.

    Which products count as "should be here" depends on the scope:

    - `expected_skus` given (a shelf scan): those SKUs plus anything detected.
      The planogram is the authority on what belongs on that shelf, so an empty
      slot still produces a phantom row.
    - no scope at all (a whole-store reconciliation): the entire catalogue.

    Passing only `shelf_id` without `expected_skus` deliberately narrows to the
    detected set — there is no way to know what else that shelf should hold.
    """
    counts = aggregate_detections(detections)
    products = resolve_products(db, list(counts.keys()))

    considered: Dict[int, Product] = {p.id: p for p in products.values()}
    if expected_skus:
        stmt = select(Product).where(Product.sku.in_(list(expected_skus)))
        for product in db.execute(stmt).scalars().all():
            considered.setdefault(product.id, product)
    elif shelf_id is None:
        for product in db.execute(select(Product)).scalars().all():
            considered.setdefault(product.id, product)

    detected_by_product: Dict[int, Tuple[int, float]] = {}
    for key, (facings, confidence) in counts.items():
        product = products.get(key)
        if product is None:
            continue
        prev_facings, prev_conf = detected_by_product.get(product.id, (0, 0.0))
        total = prev_facings + facings
        blended = (
            (prev_conf * prev_facings + confidence * facings) / total if total else 0.0
        )
        detected_by_product[product.id] = (total, blended)

    logs: List[InventoryLog] = []
    items: List[DiscrepancyItem] = []

    for product_id, product in considered.items():
        detected, confidence = detected_by_product.get(product_id, (0, 0.0))
        delta, kind, severity = InventoryLog.classify(detected, product.system_stock)

        log = InventoryLog(
            session_id=session_id,
            product_id=product.id,
            detected_count=detected,
            system_count=product.system_stock,
            discrepancy=delta,
            discrepancy_type=kind,
            severity=severity,
            mean_confidence=confidence or None,
            shelf_id=shelf_id,
        )
        logs.append(log)

        if kind is not DiscrepancyType.MATCH:
            items.append(
                DiscrepancyItem(
                    sku=product.sku,
                    product_name=product.name,
                    detected_count=detected,
                    system_count=product.system_stock,
                    discrepancy=delta,
                    discrepancy_type=kind,
                    severity=severity,
                    estimated_value_impact=round(abs(delta) * (product.unit_price or 0.0), 2),
                )
            )

    if persist and logs:
        db.add_all(logs)
        db.flush()

    items.sort(key=lambda i: (-i.estimated_value_impact, i.sku))
    return logs, items


def build_summary(db: Session) -> InventorySummary:
    """Latest-state roll-up for the dashboard overview tiles."""
    products = db.execute(select(Product)).scalars().all()
    total_products = len(products)
    total_system = sum(p.system_stock for p in products)

    latest: Dict[int, InventoryLog] = {}
    stmt = select(InventoryLog).order_by(InventoryLog.created_at.desc(), InventoryLog.id.desc())
    for log in db.execute(stmt).scalars().all():
        latest.setdefault(log.product_id, log)

    rows = list(latest.values())
    phantom = sum(1 for row in rows if row.discrepancy_type is DiscrepancyType.PHANTOM)
    under = sum(1 for row in rows if row.discrepancy_type is DiscrepancyType.UNDERCOUNT)
    over = sum(1 for row in rows if row.discrepancy_type is DiscrepancyType.OVERCOUNT)
    matched = sum(1 for row in rows if row.discrepancy_type is DiscrepancyType.MATCH)

    price_by_product = {p.id: (p.unit_price or 0.0) for p in products}
    value_at_risk = sum(
        abs(row.discrepancy) * price_by_product.get(row.product_id, 0.0) for row in rows
    )
    last_scan = max((row.created_at for row in rows), default=None)

    return InventorySummary(
        total_products=total_products,
        total_system_stock=total_system,
        total_detected_stock=sum(row.detected_count for row in rows),
        phantom_skus=phantom,
        undercount_skus=under,
        overcount_skus=over,
        accuracy_rate=(matched / len(rows)) if rows else 0.0,
        value_at_risk=round(value_at_risk, 2),
        last_scan_at=last_scan,
    )
