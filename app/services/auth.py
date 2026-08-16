"""Authentication and role enforcement.

Login takes a username and a PIN and returns an opaque session token. The token
is a random 32-byte value looked up server-side rather than a signed JWT,
because a shop must be able to revoke a device that walked out of the building
and a self-contained token cannot be withdrawn before it expires.

Failed attempts are counted per user and lock the account for fifteen minutes
after five. That lockout, not the PIN length, is what makes a four-digit
credential defensible: it turns ten thousand possibilities into roughly one
guess every three minutes.

Errors are deliberately uniform. A wrong username and a wrong PIN return the
same message, so the endpoint cannot be used to discover who works here.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.core.logging import get_logger
from app.db.base import utcnow
from app.models.enums import UserRole
from app.models.user import Session, User

logger = get_logger(__name__)

#: A shift is long, and re-entering a PIN mid-aisle is friction. A day balances
#: that against a device left on a counter overnight.
SESSION_HOURS = 12


class AuthError(Exception):
    """Login failed. The message is safe to show a user."""


class PermissionDenied(Exception):
    """Authenticated, but not allowed to do this."""


def authenticate(db: DbSession, username: str, pin: str) -> Session:
    """Verify credentials and open a session.

    Raises AuthError with the same wording for every failure mode a caller
    could use for enumeration.
    """
    generic = "That username or PIN is not right."

    user = db.execute(select(User).where(User.username == username.strip())).scalar_one_or_none()

    if user is None:
        # Hash anyway so a missing user does not answer measurably faster than
        # a wrong PIN, which would leak which usernames exist.
        User(username="_", name="_", role=UserRole.STAFF).set_pin(pin)
        raise AuthError(generic)

    if not user.is_active:
        raise AuthError("That account has been switched off. Ask your manager.")

    if user.is_locked:
        raise AuthError("Too many wrong PINs. Try again in a few minutes.")

    if not user.verify_pin(pin):
        user.register_failure()
        db.commit()
        raise AuthError(generic)

    user.register_success()
    session = Session(
        token=Session.new_token(),
        user_id=user.id,
        expires_at=utcnow() + timedelta(hours=SESSION_HOURS),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info("Login: %s (%s)", user.username, user.role.value)
    return session


def resolve_session(db: DbSession, token: Optional[str]) -> Optional[User]:
    """The user behind a token, or None if it is missing, expired or revoked."""
    if not token:
        return None
    session = db.execute(select(Session).where(Session.token == token)).scalar_one_or_none()
    if session is None or not session.is_valid:
        return None
    return session.user if session.user.is_active else None


def revoke(db: DbSession, token: str) -> bool:
    """Log out. Returns whether a live session was actually ended."""
    session = db.execute(select(Session).where(Session.token == token)).scalar_one_or_none()
    if session is None or session.revoked:
        return False
    session.revoked = True
    db.commit()
    return True


#: Who may do what.
#:
#: Expressed as sets rather than a rank order because the roles are not a
#: hierarchy: a coordinator assigns restock work that a manager may never touch,
#: so "manager > coordinator > staff" would be the wrong model.
_MANAGER_ONLY = frozenset({UserRole.MANAGER})
_MANAGER_OR_COORDINATOR = frozenset({UserRole.MANAGER, UserRole.COORDINATOR})
_ANY = frozenset(UserRole)

PERMISSIONS: dict[str, frozenset[UserRole]] = {
    "shelf:create": _MANAGER_ONLY,
    "shelf:allocate": _MANAGER_ONLY,
    "shelf:set_buffer": _MANAGER_ONLY,
    "user:manage": _MANAGER_ONLY,
    "restock:assign": _MANAGER_OR_COORDINATOR,
    "restock:complete": _ANY,
    "batch:receive": _MANAGER_OR_COORDINATOR,
    "stock:place": _ANY,
    "sale:scan": _ANY,
    "shelf:view": _ANY,
}


def may(user: Optional[User], action: str) -> bool:
    if user is None:
        return False
    allowed = PERMISSIONS.get(action)
    if allowed is None:  # unknown action: refuse rather than wave it through
        logger.warning("Unknown permission checked: %s", action)
        return False
    return user.role in allowed


def require(user: Optional[User], action: str) -> User:
    """Assert a permission, or raise. Returns the user so callers can chain."""
    if user is None:
        raise PermissionDenied("You need to sign in first.")
    if not may(user, action):
        raise PermissionDenied(f"A {user.role.value} cannot do that. Ask a manager.")
    return user
