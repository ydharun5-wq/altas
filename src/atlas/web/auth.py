"""Bootstrap and session endpoints; there is no open-registration route."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import hmac

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from atlas.web.database import get_db_session
from atlas.web.models import AuditEvent, AuthSession, User
from atlas.web.schemas import BootstrapRequest, LoginRequest, SessionResponse
from atlas.web.security import (
    AuthContext,
    clear_login_failures,
    current_auth,
    ensure_login_allowed,
    hash_password,
    new_session,
    record_login_failure,
    require_csrf,
    require_origin,
    request_settings,
    token_digest,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


def _set_session_cookies(response: Response, raw_session: str, csrf: str, settings) -> None:
    common = {"secure": settings.cookie_secure, "samesite": "strict", "path": "/"}
    response.set_cookie(
        "atlas_session", raw_session, httponly=True,
        max_age=settings.session_ttl_hours * 3600, **common
    )
    response.set_cookie(
        "atlas_csrf", csrf, httponly=False,
        max_age=settings.session_ttl_hours * 3600, **common
    )


def _session_response(user: User, csrf: str, expires_at: datetime) -> SessionResponse:
    return SessionResponse(username=user.username, csrf_token=csrf, expires_at=expires_at)


@router.get("/bootstrap-status")
def bootstrap_status(db: Session = Depends(get_db_session)) -> dict[str, bool]:
    """Expose only whether first-owner setup is available."""
    return {"setup_required": db.get(User, 1) is None}


@router.post("/bootstrap", response_model=SessionResponse, status_code=201)
def bootstrap_owner(
    payload: BootstrapRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db_session),
    settings=Depends(request_settings),
) -> SessionResponse:
    require_origin(request, settings)
    if db.get(User, 1) is not None:
        raise HTTPException(status_code=409, detail="Initial owner is already configured")
    configured = settings.bootstrap_token
    supplied = payload.bootstrap_token.get_secret_value()
    if not configured or not hmac.compare_digest(configured, supplied):
        raise HTTPException(status_code=403, detail="Bootstrap is not authorized")
    user = User(id=1, username=payload.username, password_hash=hash_password(payload.password.get_secret_value()))
    db.add(user)
    try:
        db.flush()
        raw, csrf, session = new_session(db, user, settings)
        db.add(AuditEvent(actor_user_id=user.id, action="owner.bootstrap", resource_type="user", resource_id="1"))
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Initial owner is already configured") from exc
    _set_session_cookies(response, raw, csrf, settings)
    return _session_response(user, csrf, session.expires_at)


@router.post("/login", response_model=SessionResponse)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db_session),
    settings=Depends(request_settings),
) -> SessionResponse:
    require_origin(request, settings)
    ip = request.client.host if request.client else "unknown"
    key = sha256(f"{ip}\0{payload.username}".encode("utf-8")).hexdigest()
    ensure_login_allowed(db, key, settings)
    user = db.scalar(select(User).where(User.username == payload.username))
    valid = verify_password(payload.password.get_secret_value(), user.password_hash if user else None)
    if not user or not valid:
        record_login_failure(db, key, settings)
        raise HTTPException(status_code=401, detail="Invalid username or password")
    clear_login_failures(db, key)
    raw, csrf, session = new_session(db, user, settings)
    db.add(AuditEvent(actor_user_id=user.id, action="auth.login", resource_type="session"))
    db.commit()
    _set_session_cookies(response, raw, csrf, settings)
    return _session_response(user, csrf, session.expires_at)


@router.get("/session", response_model=SessionResponse)
def session_status(auth: AuthContext = Depends(current_auth)) -> SessionResponse:
    return _session_response(auth.user, auth.csrf_token, auth.session.expires_at)


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    response: Response,
    auth: AuthContext = Depends(require_csrf),
    db: Session = Depends(get_db_session),
) -> Response:
    auth.session.revoked_at = datetime.now(timezone.utc)
    db.add(AuditEvent(actor_user_id=auth.user.id, action="auth.logout", resource_type="session", resource_id=str(auth.session.id)))
    db.commit()
    response.delete_cookie("atlas_session", path="/", secure=request.app.state.settings.cookie_secure, httponly=True, samesite="strict")
    response.delete_cookie("atlas_csrf", path="/", secure=request.app.state.settings.cookie_secure, httponly=False, samesite="strict")
    response.status_code = 204
    return response
