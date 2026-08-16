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


class UserRole(str, Enum):
    """Who may do what.

    Deliberately three, matching how the work actually divides on a shop floor:
    the manager designs, the coordinator dispatches, the staff execute.
    """

    MANAGER = "manager"  # designs shelves, allocates rows, sets buffers
    COORDINATOR = "coordinator"  # assigns restock work and confirms completion
    STAFF = "staff"  # executes restocking, scans sales


class RestockStatus(str, Enum):
    OPEN = "open"  # threshold breached, nobody assigned yet
    ASSIGNED = "assigned"  # coordinator gave it to a named person
    DONE = "done"  # staff refilled it and marked it complete
    CANCELLED = "cancelled"  # no longer needed (row re-allocated, or refilled anyway)


class MovementType(str, Enum):
    """Every change to what is physically on a shelf, so stock is auditable.

    Shelf quantity is never edited in place: it is derived from these rows, which
    means a disagreement between the system and the shelf can always be traced
    to a specific event rather than guessed at.
    """

    PLACED = "placed"  # staff put units on the shelf
    SOLD = "sold"  # scanned at checkout
    REMOVED = "removed"  # pulled for damage, expiry or recall
    CORRECTION = "correction"  # manual count adjustment after a physical audit
