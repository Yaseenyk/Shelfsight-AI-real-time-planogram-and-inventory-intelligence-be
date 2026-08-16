"""Staff accounts and their sessions.

A PIN rather than a password, deliberately. This is used standing in an aisle,
often one-handed and sometimes with wet hands; a six-digit numeric entry is
enterable on a phone keypad in a second, where an email and password is not.

The PIN is still treated as a credential: stored only as a salted hash, never
in clear text, and compared in constant time. Six digits is a small keyspace, so
`failed_attempts` and `locked_until` do the work that PIN length cannot -- an
attacker gets a handful of guesses, not a million.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, utcnow
from app.models.enums import UserRole

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.shelf import RestockTask

#: PBKDF2 rounds. High enough that brute-forcing a six-digit PIN offline is slow,
#: low enough that a login on the client's CPU-only machine stays instant.
_PBKDF2_ROUNDS = 200_000
_LOCKOUT_THRESHOLD = 5
_LOCKOUT_MINUTES = 15


def hash_pin(pin: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Return `(salt, hash)` for a PIN. A fresh salt is generated when omitted."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt.encode(), _PBKDF2_ROUNDS)
    return salt, digest.hex()


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    #: Short human identifier shown on the login screen and in restock history.
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, native_enum=False, length=16), nullable=False, index=True
    )

    pin_salt: Mapped[str] = mapped_column(String(64), nullable=False)
    pin_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    sessions: Mapped[List["Session"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    assigned_tasks: Mapped[List["RestockTask"]] = relationship(
        back_populates="assignee", foreign_keys="RestockTask.assigned_to_id"
    )

    # ------------------------------------------------------------- behaviour --
    def set_pin(self, pin: str) -> None:
        self.pin_salt, self.pin_hash = hash_pin(pin)

    @property
    def is_locked(self) -> bool:
        if self.locked_until is None:
            return False
        locked_until = self.locked_until
        if locked_until.tzinfo is None:  # SQLite hands back naive datetimes
            locked_until = locked_until.replace(tzinfo=timezone.utc)
        return locked_until > utcnow()

    def verify_pin(self, pin: str) -> bool:
        """Constant-time PIN check.

        `compare_digest` rather than `==`: string comparison short-circuits on
        the first differing character, which leaks how much of a guess was
        correct to anyone able to time the response.
        """
        _, candidate = hash_pin(pin, self.pin_salt)
        return hmac.compare_digest(candidate, self.pin_hash)

    def register_failure(self) -> None:
        # `or 0` because a column default is applied by the database on INSERT,
        # not by Python on construction: a user object that has not been flushed
        # yet still has None here, and the first failed login would crash.
        self.failed_attempts = (self.failed_attempts or 0) + 1
        if self.failed_attempts >= _LOCKOUT_THRESHOLD:
            self.locked_until = utcnow() + timedelta(minutes=_LOCKOUT_MINUTES)
            self.failed_attempts = 0

    def register_success(self) -> None:
        self.failed_attempts = 0
        self.locked_until = None
        self.last_login_at = utcnow()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<User {self.username} ({self.role.value})>"


class Session(Base, TimestampMixin):
    """A logged-in device.

    Server-side rather than a signed token, because a shop needs to be able to
    revoke a device that walked out of the building, and a stateless token
    cannot be withdrawn before it expires.
    """

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="sessions")

    @property
    def is_valid(self) -> bool:
        if self.revoked:
            return False
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires > utcnow()

    @staticmethod
    def new_token() -> str:
        return secrets.token_urlsafe(32)
