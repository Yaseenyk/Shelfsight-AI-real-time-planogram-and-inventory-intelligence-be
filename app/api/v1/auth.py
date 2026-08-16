"""Sign in, sign out, and "who am I".

The token travels in an `Authorization: Bearer` header rather than a cookie:
the dashboard and the API are separately deployable here, and a bearer header
avoids the cross-site cookie configuration that would otherwise be needed the
moment the client puts them on different hosts.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DbSession

from app.api.deps import get_db
from app.models.enums import UserRole
from app.models.user import User
from app.services import auth as auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    #: Numeric, but carried as a string so a leading zero survives the round trip.
    pin: str = Field(..., min_length=4, max_length=12)


class UserRead(BaseModel):
    id: int
    username: str
    name: str
    role: UserRole

    model_config = {"from_attributes": True}


class LoginResponse(BaseModel):
    token: str
    expires_at: str
    user: UserRead


def current_user(
    authorization: Optional[str] = Header(default=None),
    db: DbSession = Depends(get_db),
) -> Optional[User]:
    """Resolve the caller, or None. Use `require_user` when a route needs one."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return auth_service.resolve_session(db, authorization.split(" ", 1)[1].strip())


def require_user(user: Optional[User] = Depends(current_user)) -> User:
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue.")
    return user


def requires(action: str):
    """Dependency factory guarding a route with a named permission.

        @router.post("/shelves", dependencies=[Depends(requires("shelf:create"))])

    Named actions rather than role checks at the call site: the rule lives in
    one table, so widening a permission is one edit rather than a search for
    every route that happened to compare a role.
    """

    def guard(user: User = Depends(require_user)) -> User:
        try:
            return auth_service.require(user, action)
        except auth_service.PermissionDenied as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc

    return guard


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: DbSession = Depends(get_db)) -> LoginResponse:
    try:
        session = auth_service.authenticate(db, payload.username, payload.pin)
    except auth_service.AuthError as exc:
        # 401 with a uniform message: a wrong username and a wrong PIN must be
        # indistinguishable, or the endpoint lists who works here.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    return LoginResponse(
        token=session.token,
        expires_at=session.expires_at.isoformat(),
        user=UserRead.model_validate(session.user),
    )


@router.get("/me", response_model=UserRead)
def me(user: User = Depends(require_user)) -> UserRead:
    return UserRead.model_validate(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    authorization: Optional[str] = Header(default=None),
    db: DbSession = Depends(get_db),
) -> None:
    if authorization and authorization.lower().startswith("bearer "):
        auth_service.revoke(db, authorization.split(" ", 1)[1].strip())
