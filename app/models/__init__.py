"""ORM model registry.

Import every model here so `Base.metadata.create_all()` and Alembic autogenerate
see the full schema regardless of import order.
"""

from app.models.compliance import ComplianceAudit
from app.models.enums import (
    ComplianceStatus,
    DiscrepancyType,
    ExpiryStatus,
    FreshnessLabel,
    ScanStatus,
    Severity,
)
from app.models.expiry import ExpiryAudit
from app.models.freshness import FreshnessAudit
from app.models.inventory import InventoryLog
from app.models.planogram import PlanogramLayout
from app.models.product import Product
from app.models.scan import ScanSession

__all__ = [
    "ComplianceAudit",
    "ComplianceStatus",
    "DiscrepancyType",
    "ExpiryAudit",
    "ExpiryStatus",
    "FreshnessAudit",
    "FreshnessLabel",
    "InventoryLog",
    "PlanogramLayout",
    "Product",
    "ScanSession",
    "ScanStatus",
    "Severity",
]
