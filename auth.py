from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from config import settings
from database import User, get_db

bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_access_token(user_id: str, role: str) -> str:
    payload = {"sub": user_id, "role": role, "exp": datetime.now(timezone.utc) + timedelta(hours=8)}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if not credentials:
        raise HTTPException(401, "Sign in is required")
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=["HS256"])
        user = db.get(User, payload["sub"])
    except (jwt.PyJWTError, KeyError):
        user = None
    if not user:
        raise HTTPException(401, "Your session has expired")
    return user


GUEST_USER_ID = "guest_default"


def get_optional_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Provides seamless access for normal users without forced sign-in."""
    if credentials and credentials.credentials:
        try:
            payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=["HS256"])
            user = db.get(User, payload.get("sub"))
            if user:
                return user
        except Exception:
            pass
    guest = db.get(User, GUEST_USER_ID)
    if not guest:
        guest = User(id=GUEST_USER_ID, email="guest@aster.local", password_hash="none", role="user")
        db.add(guest)
        try:
            db.commit()
            db.refresh(guest)
        except Exception:
            db.rollback()
            guest = db.get(User, GUEST_USER_ID) or User(id=GUEST_USER_ID, email="guest@aster.local", password_hash="none", role="user")
    return guest


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Administrator access is required")
    return user
