from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

import time
import bcrypt
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Cookie, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from database import User, get_db

bearer = HTTPBearer(auto_error=False)

SESSION_COOKIE = "astra_session"
TOKEN_TTL = timedelta(days=30)
BCRYPT_ROUNDS = 12
ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)

# Login Rate Limiting & Lockout Tracker
LOGIN_ATTEMPTS: dict[str, dict] = {}
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 600  # 10 minutes temporary lockout


def record_failed_attempt(key: str) -> tuple[int, int]:
    """Records a failed login attempt for key (IP or identifier). Returns (remaining_attempts, lockout_seconds)."""
    now = time.time()
    entry = LOGIN_ATTEMPTS.get(key, {"attempts": 0, "locked_until": 0, "last_failed": 0})
    if entry.get("locked_until", 0) > now:
        return 0, int(entry["locked_until"] - now)

    if now - entry.get("last_failed", 0) > LOCKOUT_DURATION_SECONDS:
        entry["attempts"] = 0

    entry["attempts"] += 1
    entry["last_failed"] = now

    if entry["attempts"] >= MAX_LOGIN_ATTEMPTS:
        entry["locked_until"] = now + LOCKOUT_DURATION_SECONDS
        LOGIN_ATTEMPTS[key] = entry
        return 0, LOCKOUT_DURATION_SECONDS

    LOGIN_ATTEMPTS[key] = entry
    return max(0, MAX_LOGIN_ATTEMPTS - entry["attempts"]), 0


def check_is_locked(key: str) -> int:
    """Returns remaining lockout seconds if locked, or 0 if free to attempt."""
    now = time.time()
    entry = LOGIN_ATTEMPTS.get(key)
    if not entry:
        return 0
    locked_until = entry.get("locked_until", 0)
    if locked_until > now:
        return int(locked_until - now)
    return 0


def clear_failed_attempts(key: str) -> None:
    LOGIN_ATTEMPTS.pop(key, None)


WEAK_PASSWORDS = frozenset({
    "password", "password1", "password123", "12345678", "123456789", "qwerty",
    "qwerty123", "letmein", "welcome", "admin123", "iloveyou", "abc12345",
    "passw0rd", "11111111", "00000000", "astra123", "changeme",
})
PASSWORD_HINT = "Use at least 8 characters with a letter and a number."


def hash_password(password: str) -> str:
    """Hashes password using secure Argon2id algorithm."""
    return ph.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verifies password using Argon2id with automatic fallback for legacy bcrypt hashes."""
    if not password_hash or password_hash in ("", "none"):
        return False
    if password_hash.startswith("$argon2"):
        try:
            return ph.verify(password_hash, password)
        except (VerifyMismatchError, Exception):
            return False
    # Fallback for legacy bcrypt hashes
    try:
        raw = password.encode("utf-8")[:72]
        return bcrypt.checkpw(raw, password_hash.encode())
    except (ValueError, TypeError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """Returns True if hash should be upgraded to modern Argon2id settings."""
    if not password_hash or not password_hash.startswith("$argon2"):
        return True
    try:
        return ph.check_needs_rehash(password_hash)
    except Exception:
        return True


def validate_password_strength(password: str) -> str | None:
    if not password:
        return "Password cannot be blank."
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if len(password) > 128:
        return "Password must be 128 characters or fewer."
    if password.lower() in WEAK_PASSWORDS or password.lower().strip() in WEAK_PASSWORDS:
        return "This password is too common and weak. Please choose a stronger password."
    if password.isdigit():
        return "Password must include at least one letter."
    if password.isalpha():
        return "Password must include at least one number."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return PASSWORD_HINT
    return None


def create_access_token(user_id: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.now(timezone.utc) + TOKEN_TTL,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def set_session_cookie(response: Response, token: str) -> None:
    secure = settings.app_env.lower() in ("production", "prod")
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=int(TOKEN_TTL.total_seconds()),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


def _decode_user(token: str | None, db: Session) -> User | None:
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            return None
        return db.get(User, user_id)
    except (jwt.PyJWTError, KeyError):
        return None


def _extract_token(
    credentials: HTTPAuthorizationCredentials | None,
    cookie_token: str | None,
    guest_header_token: str | None = None,
) -> str | None:
    if credentials and credentials.credentials:
        return credentials.credentials
    if guest_header_token:
        return guest_header_token
    if cookie_token:
        return cookie_token
    return None


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Annotated[Session, Depends(get_db)],
    astra_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> User:
    user = _decode_user(_extract_token(credentials, astra_session), db)
    if not user or user.role == "guest":
        raise HTTPException(401, "Sign in is required")
    return user


def get_optional_user(
    request: Request,
    response: Response,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Annotated[Session, Depends(get_db)],
    astra_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> User:
    """Return the signed-in user or create an isolated per-device local guest session.

    Strict per-device isolation guarantees that an unauthenticated user's chat is
    only visible to their own browser/device and never shared across devices,
    even when connected to the exact same local network or Wi-Fi.
    """
    guest_header = request.headers.get("X-Guest-Token")
    device_header = request.headers.get("X-Device-Id")
    extracted = _extract_token(credentials, astra_session, guest_header)
    user = _decode_user(extracted, db)
    if user:
        if user.role == "guest":
            user_token = create_access_token(user.id, user.role)
            response.headers["X-Guest-Token"] = user_token
            response.headers["Access-Control-Expose-Headers"] = "X-Guest-Token"
            set_session_cookie(response, user_token)
        return user

    if device_header:
        dev_email = f"guest-{device_header}@device.local"
        existing = db.scalar(select(User).where(User.email == dev_email))
        if existing:
            user_token = create_access_token(existing.id, existing.role)
            response.headers["X-Guest-Token"] = user_token
            response.headers["Access-Control-Expose-Headers"] = "X-Guest-Token"
            set_session_cookie(response, user_token)
            return existing

    guest_id = device_header if device_header else uuid.uuid4().hex
    user = User(
        email=f"guest-{guest_id}@device.local",
        username="Guest",
        role="guest",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    guest_token = create_access_token(user.id, user.role)
    response.headers["X-Guest-Token"] = guest_token
    response.headers["Access-Control-Expose-Headers"] = "X-Guest-Token"
    set_session_cookie(response, guest_token)
    return user


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Administrator access is required")
    return user
