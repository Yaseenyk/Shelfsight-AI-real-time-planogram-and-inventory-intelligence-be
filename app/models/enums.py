"""Canonical enumerations shared by the ORM layer, the API schemas and the
evaluation harness. Values are lowercase strings so they serialise cleanly to
JSON and to the CSV exports consumed by the benchmark scripts.
"""

from __future__ import annotations

from enum import Enum


class ScanStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DiscrepancyType(str, Enum):
    """Outcome of `detected_count` vs `system_count`."""

    MATCH = "match"
    PHANTOM = "phantom"  # system says stocked, shelf is empty -> phantom inventory
    UNDERCOUNT = "undercount"  # detected < system, but non-zero
    OVERCOUNT = "overcount"  # detected > system (misplaced stock / bad count)


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    MISPLACED = "misplaced"
    MISSING = "missing"
    EXTRA = "extra"


class FreshnessLabel(str, Enum):
    FRESH = "fresh"
    RIPENING = "ripening"
    SPOILED = "spoiled"


class ExpiryStatus(str, Enum):
    VALID = "valid"
    NEAR_EXPIRY = "near_expiry"
    EXPIRED = "expired"
    UNREADABLE = "unreadable"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
