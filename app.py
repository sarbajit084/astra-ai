"""Production-grade Enterprise RAG API.

Stateless architecture:
- Auth / sessions: Stateless JWT
- Metadata & Query Events: SQL (PostgreSQL / SQLite)
- Dense Chunks: Qdrant Vector Store
- LLM Inference: Grok / Groq LPU engine
- Metrics: Prometheus /metrics
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import random
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import urllib.parse

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from auth import (
    check_is_locked,
    clear_failed_attempts,
    clear_session_cookie,
    create_access_token,
    get_current_user,
    get_optional_user,
    hash_password,
    needs_rehash,
    record_failed_attempt,
    require_admin,
    set_session_cookie,
    validate_password_strength,
    verify_password,
)
from config import BUNDLE_DIR, settings
from database import Conversation, Document, QueryEvent, User, UserPreference, get_db, initialize_database
from rag_engine import ProductionRAGService, is_chemistry_query

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger("rag_api")
STATIC_DIR = BUNDLE_DIR / "static"
rag = ProductionRAGService()

# Prometheus Metrics for 200k User Scale Monitoring
REQUEST_COUNT = Counter("aster_query_requests_total", "Total chat query requests", ["status", "model"])
QUERY_LATENCY = Histogram(
    "aster_query_latency_seconds",
    "Query end-to-end latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


# Ensure tables are initialized
initialize_database()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    logger.info(
        "service_started environment=%s provider=%s model=%s grok_configured=%s",
        settings.app_env,
        settings.llm_provider,
        settings.active_model,
        settings.grok_configured,
    )
    yield
    await rag.close()


app = FastAPI(title="Aster Grounded RAG", version="3.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, __: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "Please check your input and try again."})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    logger.exception("unhandled_error")
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})


cors_origins = settings.cors_origins
if cors_origins == ["*"] or not cors_origins or "*" in cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://.*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Guest-Token", "Content-Disposition"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Guest-Token", "Content-Disposition"],
    )


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(self), camera=()"
    response.headers["Access-Control-Expose-Headers"] = "X-Guest-Token"
    if settings.app_env.lower() in ("production", "prod"):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    if not request.url.path.startswith("/preview/"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' data: blob:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "font-src 'self' data: https://fonts.gstatic.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob: https: http:; "
            "connect-src 'self' *; "
            "frame-src 'self' *; "
            "frame-ancestors 'self' *;"
        )
    return response


class MathChallengeResponse(BaseModel):
    challenge_token: str
    num1: int
    num2: int
    operation: str
    question: str


class RegisterPayload(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    username: str = Field(min_length=2, max_length=100)
    phone: str | None = Field(default="", max_length=30)
    password: str = Field(min_length=8, max_length=128)
    password_confirm: str | None = Field(default=None, max_length=128)
    challenge_token: str = Field(min_length=1)
    calculation_result: int
    agreed_to_terms: bool = False


class LoginPayload(BaseModel):
    identifier: str | None = Field(default=None, max_length=255)
    phone: str | None = Field(default=None, max_length=30)
    email: str | None = Field(default=None, max_length=255)
    password: str = Field(min_length=1, max_length=128)
    challenge_token: str = Field(min_length=1)
    calculation_result: int


class ForgotPasswordPayload(BaseModel):
    identifier: str = Field(min_length=1, max_length=255)
    challenge_token: str = Field(min_length=1)
    calculation_result: int


class ResetPasswordPayload(BaseModel):
    reset_token: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)
    new_password_confirm: str | None = Field(default=None, max_length=128)



class PreferencesPayload(BaseModel):
    payload: dict = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    selected_doc_id: str | None = None
    conversation_id: str | None = None
    history: list[dict] | None = Field(default_factory=list)
    incognito: bool = False
    detailed: bool = False
    mode: str = "general"


def token_payload(user: User) -> dict:
    displayName = user.username or (user.email.split("@")[0] if user.email else "User")
    return {
        "access_token": create_access_token(user.id, user.role),
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "username": displayName,
            "phone": user.phone or "",
            "role": user.role,
        },
    }


def auth_success_response(user: User) -> JSONResponse:
    body = token_payload(user)
    response = JSONResponse(content=body)
    set_session_cookie(response, body["access_token"])
    return response


token_response = auth_success_response



def owned_conversation(db: Session, conv_id: str, user_id: str) -> Conversation:
    conv = db.scalar(select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user_id))
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


def owned_document(db: Session, document_id: str, user_id: str) -> Document:
    document = db.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user_id))
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt() -> PlainTextResponse:
    content = (
        "User-agent: *\n"
        "Allow: /\n"
        "Allow: /terms\n"
        "Allow: /privacy\n"
        "Disallow: /api/\n"
        "Disallow: /admin\n"
        "Disallow: /delete-account\n"
        "Disallow: /dashboard\n"
        "Disallow: /chat\n"
        "Disallow: /conversations\n"
        "Disallow: /documents\n"
        "Disallow: /settings\n"
    )
    return PlainTextResponse(content=content, media_type="text/plain")


@app.get("/terms", response_class=FileResponse)
def terms_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "terms.html")


@app.get("/privacy", response_class=FileResponse)
def privacy_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "privacy.html")


@app.get("/delete-account", response_class=FileResponse)
def delete_account_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "delete_account.html")


@app.get("/admin", response_class=FileResponse)
def admin_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus telemetry for 200k simultaneous users monitoring."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "healthy",
        "version": "3.0.0",
        "provider": settings.llm_provider,
        "model": settings.active_model,
        **rag.health(),
    }


# ==========================================
# Calculation Challenge Verification System
# ==========================================
import random
import hmac
import hashlib

def create_math_challenge() -> dict:
    """Generates two 2-digit random numbers (10-99) and signs the result in a secure token.
    Crucially: The token MUST NOT contain the answer in plaintext.
    Instead, it contains a cryptographic digest (HMAC) of the answer + nonce."""
    nonce = uuid.uuid4().hex[:12]
    num1 = random.randint(10, 99)
    num2 = random.randint(10, 99)
    op = random.choice(["+", "-"])
    if op == "+":
        correct_ans = num1 + num2
    else:
        if num1 < num2:
            num1, num2 = num2, num1
        correct_ans = num1 - num2

    ts = int(datetime.now(timezone.utc).timestamp())
    ans_digest = hmac.new(
        settings.jwt_secret.encode(),
        f"{nonce}:{correct_ans}".encode(),
        hashlib.sha256
    ).hexdigest()

    # Sign the challenge metadata together with ans_digest
    msg = f"{nonce}:{num1}:{op}:{num2}:{ts}:{ans_digest}"
    sig = hmac.new(settings.jwt_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
    token = f"{nonce}:{num1}:{op}:{num2}:{ts}:{ans_digest}:{sig}"

    return {
        "challenge_token": token,
        "num1": num1,
        "num2": num2,
        "operation": op,
        "question": f"What is {num1} {op} {num2}?",
    }


def verify_math_challenge(token: str, user_answer: int, max_age_seconds: int = 300) -> bool:
    """Verifies that the math challenge token is authentic, non-expired, and answered correctly.
    Answer is cryptographically verified against the HMAC digest without plaintext exposure."""
    try:
        parts = token.split(":")
        if len(parts) != 7:
            return False
        nonce, num1_str, op, num2_str, ts_str, ans_digest, sig = parts
        msg = f"{nonce}:{num1_str}:{op}:{num2_str}:{ts_str}:{ans_digest}"
        expected_sig = hmac.new(settings.jwt_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return False
        
        ts = int(ts_str)
        now_ts = int(datetime.now(timezone.utc).timestamp())
        if (now_ts - ts) > max_age_seconds or (now_ts - ts) < -60:
            return False
        
        # Verify user_answer produces the exact same ans_digest
        user_digest = hmac.new(
            settings.jwt_secret.encode(),
            f"{nonce}:{user_answer}".encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(ans_digest, user_digest)
    except Exception:
        return False


def normalize_phone(val: str | None) -> str:
    """
    Normalizes phone numbers across India (+91) and international formats consistently.
    Handles spaces, hyphens, parentheses, leading/trailing whitespace, and prefixes.
    Examples:
      '9531711863' -> '+919531711863'
      '+91 95317 11863' -> '+919531711863'
      '+91-95317-11863' -> '+919531711863'
      '09531711863' -> '+919531711863'
      '+1 (555) 019-9234' -> '+15550199234'
    """
    if not val:
        return ""
    val = val.strip()
    digits = re.sub(r"\D", "", val)
    if not digits:
        return ""

    if val.startswith("+"):
        return f"+{digits}"

    # 10-digit standard Indian mobile format (starts with 6, 7, 8, 9)
    if len(digits) == 10:
        if digits[0] in "6789":
            return f"+91{digits}"
        return f"+{digits}"
    elif len(digits) == 11 and digits.startswith("0"):
        return f"+91{digits[1:]}"
    elif len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"

    return f"+{digits}"


@app.get("/api/auth/challenge")
def get_calculation_challenge() -> dict:
    """Issues a calculation challenge between two 2-digit numbers."""
    return create_math_challenge()


@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register_user(payload: RegisterPayload, db: Annotated[Session, Depends(get_db)]) -> dict:
    # 1. Verify terms & conditions agreement
    if not payload.agreed_to_terms:
        raise HTTPException(
            status_code=400,
            detail="You must agree to the Terms & Conditions and Privacy Policy to register."
        )

    # 2. Verify calculation result
    if not verify_math_challenge(payload.challenge_token, payload.calculation_result):
        raise HTTPException(
            status_code=400,
            detail="Incorrect calculation result! Please solve the math problem correctly to register."
        )

    if payload.password_confirm is not None and payload.password != payload.password_confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    # Validate password strength
    strength_err = validate_password_strength(payload.password)
    if strength_err:
        raise HTTPException(status_code=400, detail=strength_err)

    email = payload.email.strip().lower()
    username = payload.username.strip()
    raw_phone = (payload.phone or "").strip()
    phone = normalize_phone(raw_phone) if raw_phone else ""

    # Validate phone length if provided
    if raw_phone:
        p_clean = re.sub(r"\D", "", raw_phone)
        if len(p_clean) < 10:
            raise HTTPException(status_code=400, detail="Please enter a valid phone number with at least 10 digits.")

        # Prevent duplicate accounts with the same phone number
        existing_phone_user = db.scalar(
            select(User).where(
                User.role != "guest",
                User.phone.isnot(None),
                User.phone != "",
                (User.phone == phone) | (User.phone.like(f"%{p_clean[-10:]}"))
            )
        )
        if existing_phone_user and existing_phone_user.password_hash:
            raise HTTPException(
                status_code=400,
                detail="An account with this phone number already exists. Please log in."
            )

    # Check existing account by email or username
    existing = db.scalar(select(User).where((User.email == email) | (User.username == username)))
    if existing and existing.password_hash:
        raise HTTPException(status_code=400, detail="An account with this email or username already exists. Please log in.")

    is_first_user = (db.scalar(select(func.count(User.id))) or 0) == 0
    if existing:
        user = existing
        user.email = email
        user.username = username
        user.phone = phone
        user.password_hash = hash_password(payload.password)
        user.role = "admin" if is_first_user else "user"
    else:
        user = User(
            email=email,
            username=username,
            phone=phone,
            password_hash=hash_password(payload.password),
            role="admin" if is_first_user else "user",
        )
        db.add(user)

    db.commit()
    db.refresh(user)

    logger.info("user_registered_math_verified id=%s email=%s username=%s phone=%s", user.id, user.email, user.username, user.phone)
    return token_response(user)


@app.post("/api/auth/login")
def login_user(request: Request, payload: LoginPayload, db: Annotated[Session, Depends(get_db)]) -> dict:
    ident = (payload.identifier or payload.email or payload.phone or "").strip()
    if not ident:
        raise HTTPException(status_code=400, detail="Please enter your phone number, email, or username.")

    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"{client_ip}:{ident.lower()}"

    # Lockout check
    remaining_lock = check_is_locked(rate_key)
    if remaining_lock > 0:
        minutes = (remaining_lock // 60) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Account temporarily locked due to too many failed attempts. Please try again in {minutes} minutes."
        )

    # 1. Calculation verification
    if not verify_math_challenge(payload.challenge_token, payload.calculation_result):
        raise HTTPException(
            status_code=400,
            detail="Incorrect calculation result! Please solve the math problem correctly to log in."
        )

    user = None
    clean_digits = re.sub(r"\D", "", ident)

    # 1. Phone number match
    if len(clean_digits) >= 7:
        norm_phone = normalize_phone(ident)
        last_10 = clean_digits[-10:]
        user = db.scalar(
            select(User)
            .where(
                User.role != "guest",
                User.phone.isnot(None),
                User.phone != "",
                (User.phone == norm_phone) | (User.phone == ident) | (User.phone.like(f"%{last_10}"))
            )
            .order_by(User.created_at.desc())
        )
        # Deep candidate scan for legacy formatting in DB
        if not user:
            candidates = db.scalars(
                select(User).where(
                    User.role != "guest",
                    User.phone.isnot(None),
                    User.phone != ""
                )
            ).all()
            for cand in candidates:
                cand_digits = re.sub(r"\D", "", cand.phone or "")
                if cand_digits and (cand_digits == clean_digits or cand_digits[-10:] == last_10):
                    user = cand
                    break

    # 2. Email or username match
    if not user:
        search_val = ident.lower()
        user = db.scalar(
            select(User)
            .where(
                User.role != "guest",
                (func.lower(User.email) == search_val) | (func.lower(User.username) == search_val)
            )
            .order_by(User.created_at.desc())
        )

    # 3. Fallback to payload.phone if separate
    if not user and payload.phone:
        p_norm = normalize_phone(payload.phone)
        p_clean = re.sub(r"\D", "", payload.phone)
        if len(p_clean) >= 7:
            user = db.scalar(
                select(User)
                .where(
                    User.role != "guest",
                    User.phone.isnot(None),
                    User.phone != "",
                    (User.phone == p_norm) | (User.phone == payload.phone) | (User.phone.like(f"%{p_clean[-10:]}"))
                )
                .order_by(User.created_at.desc())
            )

    if not user or not user.password_hash:
        rem, lock_secs = record_failed_attempt(rate_key)
        if lock_secs > 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Account temporarily locked due to too many failed attempts. Please try again in 10 minutes."
            )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    if not verify_password(payload.password, user.password_hash):
        rem, lock_secs = record_failed_attempt(rate_key)
        if lock_secs > 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Account temporarily locked due to too many failed attempts. Please try again in 10 minutes."
            )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    # Upgrade password hash to Argon2id if needed
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)
        db.commit()

    clear_failed_attempts(rate_key)

    logger.info("user_login_success id=%s email=%s username=%s phone=%s", user.id, user.email, user.username, user.phone)
    return token_response(user)


@app.post("/api/auth/forgot-password")
def forgot_password(payload: ForgotPasswordPayload, db: Annotated[Session, Depends(get_db)]) -> dict:
    if not verify_math_challenge(payload.challenge_token, payload.calculation_result):
        raise HTTPException(
            status_code=400,
            detail="Incorrect calculation result! Please solve the math problem correctly."
        )

    ident = payload.identifier.strip()
    if not ident:
        raise HTTPException(status_code=400, detail="Please enter your email, username, or phone number.")

    user = None
    clean_digits = re.sub(r"\D", "", ident)
    if len(clean_digits) >= 7:
        last_10 = clean_digits[-10:]
        user = db.scalar(
            select(User)
            .where(
                User.role != "guest",
                User.phone.isnot(None),
                User.phone != "",
                (User.phone == ident) | (User.phone.like(f"%{last_10}"))
            )
            .order_by(User.created_at.desc())
        )

    if not user:
        search_val = ident.lower()
        user = db.scalar(
            select(User)
            .where(
                User.role != "guest",
                (func.lower(User.email) == search_val) | (func.lower(User.username) == search_val)
            )
            .order_by(User.created_at.desc())
        )

    if not user:
        raise HTTPException(status_code=404, detail="No account found with the provided details.")

    reset_token = f"rst_{uuid.uuid4().hex}"
    user.password_reset_token = reset_token
    user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
    db.commit()

    return {
        "success": True,
        "message": "Verification successful. Please enter your new password.",
        "reset_token": reset_token,
    }


@app.post("/api/auth/reset-password")
def reset_password(payload: ResetPasswordPayload, db: Annotated[Session, Depends(get_db)]) -> dict:
    if payload.new_password_confirm is not None and payload.new_password != payload.new_password_confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    # Validate password strength
    strength_err = validate_password_strength(payload.new_password)
    if strength_err:
        raise HTTPException(status_code=400, detail=strength_err)

    user = db.scalar(
        select(User).where(
            User.password_reset_token == payload.reset_token,
            User.password_reset_expires_at > datetime.now(timezone.utc)
        )
    )
    if not user:
        raise HTTPException(status_code=400, detail="Password reset link is invalid or has expired. Please request a new one.")

    # Exact rejection requirement if new password matches existing password
    if user.password_hash and verify_password(payload.new_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Please enter a new password.")

    user.password_hash = hash_password(payload.new_password)
    user.password_reset_token = None
    user.password_reset_expires_at = None
    db.commit()

    return {
        "success": True,
        "message": "Password reset successful! You can now log in with your new password."
    }



@app.post("/api/auth/logout")
def logout_user() -> JSONResponse:
    response = JSONResponse(content={"success": True, "message": "Logged out successfully."})
    clear_session_cookie(response)
    return response


@app.get("/api/auth/me")
def me(user: Annotated[User, Depends(get_current_user)]) -> dict:
    displayName = user.username or user.email.split("@")[0]
    return {
        "id": user.id,
        "email": user.email,
        "username": displayName,
        "phone": user.phone or "",
        "role": user.role,
    }


class DeleteAccountPayload(BaseModel):
    confirm_text: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=128)


@app.post("/api/account/delete")
def delete_account(
    payload: DeleteAccountPayload,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> JSONResponse:
    if payload.confirm_text.strip() != "DELETE MY ACCOUNT":
        raise HTTPException(
            status_code=400,
            detail='You must type "DELETE MY ACCOUNT" exactly to confirm deletion.'
        )

    if not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Incorrect password. Account deletion cannot proceed."
        )

    user_id = user.id

    # 1. Purge all user query events
    db.execute(delete(QueryEvent).where(QueryEvent.user_id == user_id))

    # 2. Purge user preferences
    db.execute(delete(UserPreference).where(UserPreference.user_id == user_id))

    # 3. Purge all user conversations
    db.execute(delete(Conversation).where(Conversation.user_id == user_id))

    # 4. Purge all user documents and vector embeddings
    docs = db.scalars(select(Document).where(Document.owner_id == user_id)).all()
    for doc in docs:
        try:
            rag.delete_document(doc.id, user_id)
        except Exception:
            pass
        db.delete(doc)

    # 5. Purge the user record
    db.delete(user)
    db.commit()

    logger.info("account_permanently_deleted user_id=%s", user_id)

    response = JSONResponse(content={
        "success": True,
        "message": "Your account and all associated data have been permanently deleted."
    })
    clear_session_cookie(response)
    return response


# ==========================================
# Conversation History Navigation Endpoints
# ==========================================
@app.get("/api/conversations")
def get_conversations(user: Annotated[User, Depends(get_optional_user)], db: Annotated[Session, Depends(get_db)]) -> dict:
    convs = db.scalars(
        select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.updated_at.desc())
    ).all()

    valid_convs = []
    has_changes = False

    for conv in convs:
        raw_msgs = list(conv.messages or [])
        # Filter out temporary messages on revisit/listing
        perm_msgs = [m for m in raw_msgs if not m.get("temporary", False)]

        # If a conversation contained ONLY temporary messages (or was empty), auto-delete it from history
        if not perm_msgs:
            db.delete(conv)
            has_changes = True
        else:
            if len(perm_msgs) != len(raw_msgs):
                conv.messages = perm_msgs
                flag_modified(conv, "messages")
                has_changes = True
            valid_convs.append(conv.public())

    if has_changes:
        db.commit()

    return {"conversations": valid_convs}


@app.post("/api/conversations")
def create_conversation(user: Annotated[User, Depends(get_optional_user)], db: Annotated[Session, Depends(get_db)]) -> dict:
    conv = Conversation(user_id=user.id, title="New Conversation", messages=[])
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv.public()


@app.get("/api/conversations/{conv_id}")
def get_conversation_detail(conv_id: str, user: Annotated[User, Depends(get_optional_user)], db: Annotated[Session, Depends(get_db)]) -> dict:
    conv = db.scalar(select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user.id))
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    raw_msgs = list(conv.messages or [])
    # Filter out temporary messages when conversation is reopened/revisited
    perm_msgs = [m for m in raw_msgs if not m.get("temporary", False)]

    # If conversation had only temporary messages, auto-delete and return 404
    if not perm_msgs and raw_msgs:
        db.delete(conv)
        db.commit()
        raise HTTPException(status_code=404, detail="Conversation contained only temporary messages and was removed")

    if len(perm_msgs) != len(raw_msgs):
        conv.messages = perm_msgs
        flag_modified(conv, "messages")
        db.commit()

    return conv.public()


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str, user: Annotated[User, Depends(get_optional_user)], db: Annotated[Session, Depends(get_db)]) -> dict:
    conv = db.scalar(select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user.id))
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return {"success": True}


@app.get("/api/documents")
def documents(user: Annotated[User, Depends(get_optional_user)], db: Annotated[Session, Depends(get_db)]) -> dict:
    rows = db.scalars(
        select(Document).where(Document.owner_id == user.id).order_by(Document.created_at.desc())
    ).all()
    return {"documents": [row.public() for row in rows], "total_documents": len(rows)}


@app.post("/api/upload")
async def upload_document(
    user: Annotated[User, Depends(get_optional_user)],
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(...),
) -> dict:
    filename = file.filename or "uploaded-file.txt"
    extension = Path(filename).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(400, f"Unsupported file type '{extension}'. Supported: {', '.join(settings.allowed_extensions)}")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file cannot be empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(400, "File size exceeds the 400 MB limit. Please upload a file smaller than or equal to 400 MB.")

    document = Document(owner_id=user.id, name=Path(filename).name, status="processing")
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        result = await rag.ingest_async(document.id, user.id, filename, content)
        document.status = "ready"
        document.chunk_count = result["chunks_count"]
        document.character_count = result["characters"]
        db.commit()
        return {"success": True, "document": document.public()}
    except Exception as exc:
        document.status = "failed"
        db.commit()
        logger.exception("document_ingestion_failed document_id=%s", document.id)
        raise HTTPException(422, f"The document could not be processed: {exc}") from exc


@app.delete("/api/documents/{document_id}")
def delete_document(
    document_id: str,
    user: Annotated[User, Depends(get_optional_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    document = db.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if not document:
        raise HTTPException(404, "Document not found")
    rag.delete_document(document.id, user.id)
    db.delete(document)
    db.commit()
    return {"success": True}


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    user: Annotated[User, Depends(get_optional_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    started = time.perf_counter()
    if request.selected_doc_id:
        allowed = db.scalar(
            select(Document.id).where(Document.id == request.selected_doc_id, Document.owner_id == user.id)
        )
        if not allowed:
            raise HTTPException(404, "Selected document not found")

    clean_message = request.message.strip()

    # Normalize mode — only accept known modes; anything else falls back to 'general'
    effective_mode = request.mode if request.mode in ("general", "code") else "general"

    # Load or initialize conversation session strictly scoped to user (prevent IDOR)
    conv = None
    if request.conversation_id:
        existing_conv = db.get(Conversation, request.conversation_id)
        if existing_conv:
            if existing_conv.user_id != user.id:
                raise HTTPException(403, "You do not have permission to access this conversation.")
            conv = existing_conv

    if not conv:
        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title=clean_message[:40] + ("…" if len(clean_message) > 40 else ""),
            messages=[],
        )
        db.add(conv)
        db.flush()

    conversation_history: list[dict] = []
    if conv and conv.messages:
        conversation_history = list(conv.messages)

    # If client has longer in-memory history (e.g. rapid multi-turn before db sync), prioritize/merge
    if request.history and len(request.history) > len(conversation_history):
        conversation_history = list(request.history)

    try:
        answer = await rag.answer(
            clean_message,
            user.id,
            request.selected_doc_id,
            history=conversation_history,
            detailed=request.detailed,
            mode=effective_mode,
        )
    except Exception as exc:
        logger.exception("chat_failed user_id=%s", user.id)
        REQUEST_COUNT.labels(status="error", model="unknown").inc()
        raise HTTPException(503, f"The answer service encountered an issue: {exc}") from exc

    # Accurate, positive monotonic latency computation
    total_ms = max(5.0, round((time.perf_counter() - started) * 1000, 1))
    timings = {**answer.get("timings_ms", {}), "total": total_ms}

    # Record Prometheus metrics
    QUERY_LATENCY.observe(total_ms / 1000.0)
    REQUEST_COUNT.labels(status="success", model=answer["model_used"]).inc()

    # Only log QueryEvent for non-temporary messages
    if not request.incognito:
        event = QueryEvent(
            user_id=user.id,
            document_id=request.selected_doc_id,
            query=clean_message,
            rewritten_query=answer["rewritten_query"],
            answer=answer["answer"],
            sources=answer["sources"],
            retrieval_count=len(answer["sources"]),
            latency_ms=total_ms,
            timings=timings,
            model=answer["model_used"],
        )
        db.add(event)

    # Detect chemistry problem intent
    is_chem = bool(answer.get("is_chemistry", False) or is_chemistry_query(clean_message, answer.get("answer", "")))

    # Persist message to user's conversation session with temporary state flag
    curr_msgs = list(conv.messages or [])
    if not curr_msgs or conv.title == "New Conversation":
        conv.title = clean_message[:40] + ("…" if len(clean_message) > 40 else "")

    is_temp = bool(request.incognito)
    user_msg_entry = {"role": "user", "text": clean_message, "meta": ""}
    assistant_msg_entry = {
        "role": "assistant",
        "text": answer["answer"],
        "meta": str(total_ms),
        "sources": answer["sources"],
        "is_chemistry": is_chem,
        "research_trace": answer.get("research_trace"),
    }
    if is_temp:
        user_msg_entry["temporary"] = True
        assistant_msg_entry["temporary"] = True

    curr_msgs.append(user_msg_entry)
    curr_msgs.append(assistant_msg_entry)
    conv.messages = curr_msgs
    flag_modified(conv, "messages")
    conv.updated_at = datetime.now(timezone.utc)

    db.commit()

    return {
        **answer,
        "is_chemistry": is_chem,
        "conversation_id": conv.id,
        "latency_ms": total_ms,
        "timings_ms": timings,
    }


# In-memory storage for interactive live website previews
WEB_PREVIEWS: dict[str, dict] = {}

class WebPreviewPayload(BaseModel):
    html: str = ""
    css: str = ""
    js: str = ""
    title: str = "Astra Live Website Demo"

def _assemble_complete_html(html_code: str, css_code: str, js_code: str, title: str = "Astra Live Demo") -> str:
    import re
    cleaned_html = html_code.strip()
    cleaned_css = css_code.strip()
    cleaned_js = js_code.strip()

    has_html = bool(re.search(r"<html\b", cleaned_html, re.I))
    has_head = bool(re.search(r"<head\b", cleaned_html, re.I))
    has_body = bool(re.search(r"<body\b", cleaned_html, re.I))

    style_tag = f"\n<style>\n{cleaned_css}\n</style>\n" if cleaned_css else ""
    script_tag = f"\n<script>\n{cleaned_js}\n</script>\n" if cleaned_js else ""

    font_head_tags = (
        '  <link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">\n'
        '  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">\n'
    )

    if has_html:
        doc = cleaned_html
        # Remove dead relative link tags like <link rel="stylesheet" href="styles.css"> or script.js so they don't 404
        doc = re.sub(r'<link[^>]+href=["\'](?:styles?\.css|main\.css|style\.css)["\'][^>]*>', '', doc, flags=re.I)
        doc = re.sub(r'<script[^>]+src=["\'](?:scripts?\.js|main\.js|app\.js)["\'][^>]*>\s*</script>', '', doc, flags=re.I)
        if "fonts.googleapis.com" not in doc:
            if has_head:
                doc = re.sub(r"(<head[^>]*>)", f"\\1\n{font_head_tags}", doc, count=1, flags=re.I)
        if style_tag:
            if has_head:
                doc = re.sub(r"(</head>)", f"{style_tag}\\1", doc, count=1, flags=re.I)
            else:
                doc = re.sub(r"(<html[^>]*>)", f"\\1\n<head>{font_head_tags}{style_tag}</head>", doc, count=1, flags=re.I)
        if script_tag:
            if has_body:
                doc = re.sub(r"(</body>)", f"{script_tag}\\1", doc, count=1, flags=re.I)
            else:
                doc += script_tag
        return doc

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
{font_head_tags}
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      padding: 0;
      font-family: 'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      line-height: 1.5;
    }}
    {cleaned_css}
  </style>
</head>
<body>
  {cleaned_html}
  {script_tag}
</body>
</html>"""

@app.post("/api/preview")
def create_web_preview(payload: WebPreviewPayload) -> dict:
    """Stores generated HTML/CSS/JS website code for live preview in a new Chrome tab."""
    preview_id = str(uuid.uuid4())[:12]
    complete_html = _assemble_complete_html(payload.html, payload.css, payload.js, payload.title)
    WEB_PREVIEWS[preview_id] = {
        "html": complete_html,
        "created_at": time.time(),
    }
    if len(WEB_PREVIEWS) > 200:
        oldest_key = min(WEB_PREVIEWS.keys(), key=lambda k: WEB_PREVIEWS[k]["created_at"])
        WEB_PREVIEWS.pop(oldest_key, None)
    return {"preview_id": preview_id, "url": f"/preview/{preview_id}"}

@app.get("/preview/{preview_id}", response_class=HTMLResponse)
def render_web_preview(preview_id: str) -> HTMLResponse:
    """Renders the generated website live in a standalone browser tab."""
    preview_data = WEB_PREVIEWS.get(preview_id)
    if not preview_data:
        raise HTTPException(404, "Preview expired or not found. Please regenerate or click Live Demo again.")
    return HTMLResponse(content=preview_data["html"])


@app.get("/api/images/{filename}")
def get_generated_image(filename: str) -> FileResponse:
    """Securely serve locally generated AI artwork."""
    safe_filename = Path(filename).name
    image_path = settings.image_dir / safe_filename
    if not image_path.is_file():
        raise HTTPException(404, "Image not found")
    media_type = "image/png" if safe_filename.endswith(".png") else "image/jpeg"
    return FileResponse(image_path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/admin/dashboard")
def admin_dashboard(admin: Annotated[User, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]) -> dict:
    total_users = db.scalar(select(func.count(User.id))) or 0
    total_documents = db.scalar(select(func.count(Document.id))) or 0
    total_queries = db.scalar(select(func.count(QueryEvent.id))) or 0
    avg_latency = db.scalar(select(func.avg(QueryEvent.latency_ms))) or 0.0

    recent = db.scalars(
        select(QueryEvent).order_by(QueryEvent.created_at.desc()).limit(30)
    ).all()

    return {
        "totals": {
            "users": total_users,
            "documents": total_documents,
            "queries": total_queries,
            "average_latency_ms": max(1.0, round(float(avg_latency), 1)),
        },
        "recent_queries": [event.public() for event in recent],
    }


@app.get("/api/admin/usage")
def admin_usage(admin: Annotated[User, Depends(require_admin)], db: Annotated[Session, Depends(get_db)]) -> dict:
    """Usage patterns breakdown for administrative oversight."""
    # Past 7 days query distribution
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    recent_events = db.scalars(
        select(QueryEvent).where(QueryEvent.created_at >= seven_days_ago)
    ).all()

    # Hourly / daily breakdown
    daily_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    latencies: list[float] = []

    for ev in recent_events:
        day_str = ev.created_at.strftime("%Y-%m-%d") if ev.created_at else "Unknown"
        daily_counts[day_str] = daily_counts.get(day_str, 0) + 1
        model_counts[ev.model] = model_counts.get(ev.model, 0) + 1
        if ev.latency_ms:
            latencies.append(ev.latency_ms)

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    return {
        "daily_counts": daily_counts,
        "model_counts": model_counts,
        "percentiles": {
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
        },
        "total_analyzed": len(recent_events),
    }


@app.get("/api/admin/queries")
def admin_queries(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Provides admin full visibility into all user searches, questions, and responses."""
    events = db.scalars(
        select(QueryEvent).order_by(QueryEvent.created_at.desc()).offset(offset).limit(limit)
    ).all()
    total = db.scalar(select(func.count(QueryEvent.id))) or 0
    return {
        "total": total,
        "queries": [
            {
                "id": ev.id,
                "user_id": ev.user_id,
                "query": ev.query,
                "rewritten_query": ev.rewritten_query,
                "answer": ev.answer,
                "sources_count": ev.retrieval_count,
                "latency_ms": round(ev.latency_ms, 1),
                "model": ev.model,
                "created_at": ev.created_at.isoformat() if ev.created_at else "",
            }
            for ev in events
        ],
    }


@app.get("/healthcheck")
@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "app": "astra-ai", "version": "3.0.0"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
