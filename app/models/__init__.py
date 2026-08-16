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
    MovementType,
    RestockStatus,
    ScanStatus,
    Severity,
    UserRole,
)
from app.models.expiry import ExpiryAudit
from app.models.freshness import FreshnessAudit
from app.models.inventory import InventoryLog
from app.models.planogram import PlanogramLayout
from app.models.product import Product
from app.models.scan import ScanSession
from app.models.shelf import (
    Batch,
    Placement,
    RestockTask,
    RowAllocation,
    Shelf,
    ShelfRow,
    StockMovement,
)
from app.models.user import Session, User

__all__ = [
    "Batch",
    "ComplianceAudit",
    "ComplianceStatus",
    "DiscrepancyType",
    "ExpiryAudit",
    "ExpiryStatus",
    "FreshnessAudit",
    "FreshnessLabel",
    "InventoryLog",
    "MovementType",
    "Placement",
    "PlanogramLayout",
    "Product",
    "RestockStatus",
    "RestockTask",
    "RowAllocation",
    "ScanSession",
    "ScanStatus",
    "Session",
    "Severity",
    "Shelf",
    "ShelfRow",
    "StockMovement",
    "User",
    "UserRole",
]
