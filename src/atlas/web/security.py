"""Private owner authentication, CSRF checks, and shared login throttling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import secrets

from fastapi import Depends, HTTPException, Request, status
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.web.config import Settings
from atlas.web.database import get_db_session
from atlas.web.models import AuthSession, LoginAttempt, User

PASSWORD_HASHER = PasswordHash.recommended()
_DUMMY_HASH = PASSWORD_HASHER.hash("atlas-invalid-login-placeholder")


def token_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def create_secret() -> str:
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class AuthContext:
    user: User
    session: AuthSession
    csrf_token: str


def require_origin(request: Request, settings: Settings) -> None:
    origin = request.headers.get("origin")
    if not origin or origin.rstrip("/") not in settings.allowed_origins:
        raise HTTPException(status_code=403, detail="Untrusted request origin")


def request_settings(request: Request) -> Settings:
    return request.app.state.settings


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def current_auth(
    request: Request,
    db: Session = Depends(get_db_session),
) -> AuthContext:
    raw_token = request.cookies.get("atlas_session")
    if not raw_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    auth_session = db.scalar(select(AuthSession).where(AuthSession.token_hash == token_digest(raw_token)))
    now = datetime.now(timezone.utc)
    if (
        auth_session is None
        or auth_session.revoked_at is not None
        or _utc(auth_session.expires_at) <= now
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or revoked")
    csrf_token = request.cookies.get("atlas_csrf", "")
    if not csrf_token or not hmac.compare_digest(auth_session.csrf_hash, token_digest(csrf_token)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    user = db.get(User, auth_session.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    return AuthContext(user=user, session=auth_session, csrf_token=csrf_token)


def require_csrf(
    request: Request,
    settings: Settings = Depends(request_settings),
    auth: AuthContext = Depends(current_auth),
) -> AuthContext:
    require_origin(request, settings)
    cookie_token = request.cookies.get("atlas_csrf", "")
    header_token = request.headers.get("x-csrf-token", "")
    if (
        not cookie_token
        or not header_token
        or not hmac.compare_digest(cookie_token, header_token)
        or not hmac.compare_digest(auth.csrf_token, header_token)
    ):
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    return auth


def ensure_login_allowed(db: Session, key: str, settings: Settings) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.login_window_seconds)
    attempt = db.get(LoginAttempt, key)
    if attempt is None:
        return
    if _utc(attempt.updated_at) <= cutoff:
        db.delete(attempt)
        db.commit()
        return
    if attempt.failure_count >= settings.login_max_failures:
        raise HTTPException(status_code=429, detail="Too many failed login attempts")


def record_login_failure(db: Session, key: str, settings: Settings) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.login_window_seconds)
    attempt = db.get(LoginAttempt, key)
    if attempt is None or _utc(attempt.updated_at) <= cutoff:
        if attempt is not None:
            db.delete(attempt)
        db.flush()
        attempt = LoginAttempt(key_hash=key, failure_count=1)
        db.add(attempt)
    else:
        attempt.failure_count += 1
    db.commit()


def clear_login_failures(db: Session, key: str) -> None:
    attempt = db.get(LoginAttempt, key)
    if attempt is not None:
        db.delete(attempt)
        db.commit()


def verify_password(password: str, encoded: str | None) -> bool:
    return PASSWORD_HASHER.verify(password, encoded or _DUMMY_HASH)


def hash_password(password: str) -> str:
    return PASSWORD_HASHER.hash(password)


def new_session(db: Session, user: User, settings: Settings) -> tuple[str, str, AuthSession]:
    raw_token = create_secret()
    csrf_token = create_secret()
    now = datetime.now(timezone.utc)
    row = AuthSession(
        user_id=user.id,
        token_hash=token_digest(raw_token),
        csrf_hash=token_digest(csrf_token),
        created_at=now,
        expires_at=now + timedelta(hours=settings.session_ttl_hours),
    )
    db.add(row)
    db.flush()
    return raw_token, csrf_token, row
