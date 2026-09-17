"""Production-grade Enterprise RAG API.

Stateless architecture:
- Auth / sessions: Stateless JWT
- Metadata & Query Events: SQL (PostgreSQL / SQLite)
- Dense Chunks: Qdrant Vector Store
- LLM Inference: Grok / Groq LPU engine
- Metrics: Prometheus /metrics
"""
from __future__ import annotations

import base64
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
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status, Cookie
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from collections import defaultdict
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from auth import (
    bearer,
    check_is_locked,
    clear_failed_attempts,
    clear_session_cookie,
    create_access_token,
    get_current_user,
    get_optional_user,
    hash_password,
    needs_rehash,
    record_failed_attempt,
    revoke_user_sessions,
    set_session_cookie,
    validate_password_strength,
    verify_password,
)
from config import BUNDLE_DIR, settings
from database import Conversation, DailyUsage, Document, QueryEvent, User, UserPreference, get_db, initialize_database
from rag_engine import ProductionRAGService, is_chemistry_query

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger("rag_api")
STATIC_DIR = BUNDLE_DIR / "static"
rag = ProductionRAGService()

# Prometheus Metrics for Scale Monitoring
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


# Strict Production CORS Configuration
cors_origins = [o for o in (settings.cors_origins or []) if o and o != "*"]
if not cors_origins:
    if settings.app_env.lower() in ("production", "prod"):
        cors_origins = []
    else:
        cors_origins = ["http://localhost:8000", "http://127.0.0.1:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Guest-Token", "Content-Disposition"],
)


# Production-grade in-memory sliding window rate limiter
class InMemoryRateLimiter:
    def __init__(self) -> None:
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int = 60) -> bool:
        now = time.time()
        window_start = now - window_seconds
        history = [t for t in self.requests[key] if t > window_start]
        if len(history) >= max_requests:
            self.requests[key] = history
            return False
        history.append(now)
        self.requests[key] = history
        return True


rate_limiter = InMemoryRateLimiter()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path

    if path.startswith("/static/") or path in ("/", "/favicon.ico", "/terms", "/privacy", "/robots.txt"):
        return await call_next(request)

    if path.startswith("/api/auth/"):
        if not rate_limiter.is_allowed(f"auth:{client_ip}", max_requests=25, window_seconds=60):
            return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down and try again."})
    elif path.startswith("/api/chat"):
        if not rate_limiter.is_allowed(f"chat:{client_ip}", max_requests=40, window_seconds=60):
            return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down and try again."})
    elif path.startswith("/api/upload"):
        if not rate_limiter.is_allowed(f"upload:{client_ip}", max_requests=15, window_seconds=60):
            return JSONResponse(status_code=429, content={"detail": "Too many upload requests. Please wait a moment."})
    elif path.startswith("/api/"):
        if not rate_limiter.is_allowed(f"api:{client_ip}", max_requests=120, window_seconds=60):
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please try again later."})

    return await call_next(request)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(self), camera=(self)"
    response.headers["Access-Control-Expose-Headers"] = "X-Guest-Token"
    if settings.app_env.lower() in ("production", "prod"):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    if request.url.path.startswith("/preview/"):
        response.headers["Content-Security-Policy"] = (
            "sandbox allow-scripts allow-forms allow-modals; "
            "default-src 'self' data: blob: https:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "font-src 'self' data: https://fonts.gstatic.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob: https: http:; "
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' data: blob:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "font-src 'self' data: https://fonts.gstatic.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob: https: http:; "
            "connect-src 'self'; "
            "frame-src 'self'; "
            "frame-ancestors 'self';"
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
    image_data: str | None = None
    image_url: str | None = None
    selected_doc_id: str | None = None
    conversation_id: str | None = None
    history: list[dict] | None = Field(default_factory=list)
    incognito: bool = False
    detailed: bool = False
    mode: str = "general"


def token_payload(user: User) -> dict:
    displayName = user.username or (user.email.split("@")[0] if user.email else "User")
    return {
        "access_token": create_access_token(user.id, user.role, getattr(user, "token_version", 1)),
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


def get_user_token_status(user: User, client_ip: str, db: Session) -> dict:
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_logged_in = bool(user and user.role != "guest")
    daily_limit = 20 if is_logged_in else 5

    now_utc = datetime.now(timezone.utc)
    tomorrow_utc = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc) + timedelta(days=1)
    reset_seconds = max(0, int((tomorrow_utc - now_utc).total_seconds()))

    if is_logged_in:
        ident = f"user:{user.id}"
        record = db.scalar(select(DailyUsage).where(DailyUsage.identifier == ident, DailyUsage.usage_date == today_str))
        used = record.tokens_used if record else 0
    else:
        ident_u = f"guest_user:{user.id}" if user else "guest_unknown"
        ident_ip = f"guest_ip:{client_ip}"
        rec_u = db.scalar(select(DailyUsage).where(DailyUsage.identifier == ident_u, DailyUsage.usage_date == today_str))
        rec_ip = db.scalar(select(DailyUsage).where(DailyUsage.identifier == ident_ip, DailyUsage.usage_date == today_str))
        used_u = rec_u.tokens_used if rec_u else 0
        used_ip = rec_ip.tokens_used if rec_ip else 0
        used = max(used_u, used_ip)

    remaining = max(0, daily_limit - used)
    out_of_tokens = used >= daily_limit

    return {
        "is_logged_in": is_logged_in,
        "daily_limit": daily_limit,
        "tokens_used": used,
        "tokens_remaining": remaining,
        "out_of_tokens": out_of_tokens,
        "reset_seconds": reset_seconds,
    }


def consume_user_token(user: User, client_ip: str, db: Session) -> dict:
    status = get_user_token_status(user, client_ip, db)
    if status["out_of_tokens"]:
        msg = (
            "Out of tokens. You have used all 20 tokens for today."
            if status["is_logged_in"]
            else "Out of tokens. You have used all 5 free guest tokens for today. Please log in to get 20 tokens per day."
        )
        raise HTTPException(
            status_code=429,
            detail=msg,
        )

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_logged_in = status["is_logged_in"]

    if is_logged_in:
        ident = f"user:{user.id}"
        rec = db.scalar(select(DailyUsage).where(DailyUsage.identifier == ident, DailyUsage.usage_date == today_str))
        if not rec:
            rec = DailyUsage(identifier=ident, usage_date=today_str, tokens_used=1)
            db.add(rec)
        else:
            rec.tokens_used += 1
            rec.updated_at = datetime.now(timezone.utc)
    else:
        ident_u = f"guest_user:{user.id}"
        ident_ip = f"guest_ip:{client_ip}"
        for ident in (ident_u, ident_ip):
            rec = db.scalar(select(DailyUsage).where(DailyUsage.identifier == ident, DailyUsage.usage_date == today_str))
            if not rec:
                rec = DailyUsage(identifier=ident, usage_date=today_str, tokens_used=1)
                db.add(rec)
            else:
                rec.tokens_used += 1
                rec.updated_at = datetime.now(timezone.utc)

    db.commit()
    return get_user_token_status(user, client_ip, db)


@app.get("/api/tokens/status")
def get_tokens_status(
    request: Request,
    user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
) -> dict:
    client_ip = request.client.host if request.client else "unknown"
    return get_user_token_status(user, client_ip, db)


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


@app.get("/metrics")
def metrics(request: Request) -> Response:
    """Internal metrics telemetry restricted to local loopback interface."""
    client_ip = request.client.host if request.client else ""
    if client_ip not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(status_code=404, detail="Not found")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/health")
def health() -> dict:
    return {"status": "healthy"}


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
      '9876543210' -> '+919876543210'
      '+91 98765 43210' -> '+919876543210'
      '+91-98765-43210' -> '+919876543210'
      '09876543210' -> '+919876543210'
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
def register_user(payload: RegisterPayload, db: Session = Depends(get_db)) -> dict:
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

    if existing:
        user = existing
        user.email = email
        user.username = username
        user.phone = phone
        user.password_hash = hash_password(payload.password)
        user.role = "user"
        user.token_version = (user.token_version or 1) + 1
    else:
        user = User(
            email=email,
            username=username,
            phone=phone,
            password_hash=hash_password(payload.password),
            role="user",
            token_version=1,
        )
        db.add(user)

    db.commit()
    db.refresh(user)

    logger.info("user_registered id=%s", user.id)
    return token_response(user)


@app.post("/api/auth/login")
def login_user(request: Request, payload: LoginPayload, db: Session = Depends(get_db)) -> dict:
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

    logger.info("user_login_success id=%s", user.id)
    return token_response(user)


@app.post("/api/auth/forgot-password")
def forgot_password(payload: ForgotPasswordPayload, db: Session = Depends(get_db)) -> dict:
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

    GENERIC_RESPONSE = "If an account matching the provided details exists, password reset instructions have been dispatched."
    if not user:
        return {"success": True, "message": GENERIC_RESPONSE}

    reset_token = f"rst_{uuid.uuid4().hex}"
    user.password_reset_token = reset_token
    user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
    db.commit()

    if settings.smtp_host and settings.smtp_user and user.email:
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(f"Your password reset token is: {reset_token}\nThis token expires in 15 minutes.")
            msg["Subject"] = "Password Reset Request - Astra"
            msg["From"] = settings.smtp_from
            msg["To"] = user.email
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as s:
                if settings.smtp_use_tls:
                    s.starttls()
                if settings.smtp_password:
                    s.login(settings.smtp_user, settings.smtp_password)
                s.send_message(msg)
        except Exception as e_smtp:
            logger.warning("smtp_dispatch_failed error=%s", e_smtp)
    elif settings.app_env.lower() in ("development", "dev"):
        logger.info("[DEV ONLY] password_reset_token for user_id=%s: %s", user.id, reset_token)

    response_payload = {"success": True, "message": GENERIC_RESPONSE}
    # In non-production local development without mailer configured, provide dev_reset_token for testing
    if settings.app_env.lower() in ("development", "dev") and not (settings.smtp_host or settings.fast2sms_api_key):
        response_payload["dev_reset_token"] = reset_token

    return response_payload


@app.post("/api/auth/reset-password")
def reset_password(payload: ResetPasswordPayload, db: Session = Depends(get_db)) -> dict:
    if payload.new_password_confirm is not None and payload.new_password != payload.new_password_confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

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

    if user.password_hash and verify_password(payload.new_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Please enter a new password.")

    user.password_hash = hash_password(payload.new_password)
    user.password_reset_token = None
    user.password_reset_expires_at = None
    # Invalidate all prior sessions on password reset
    user.token_version = (user.token_version or 1) + 1
    db.commit()

    return {
        "success": True,
        "message": "Password reset successful! You can now log in with your new password."
    }


@app.post("/api/auth/logout")
def logout_user(
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    astra_session: str | None = Cookie(default=None, alias="astra_session"),
) -> JSONResponse:
    from auth import _decode_user, _extract_token
    token = _extract_token(credentials, astra_session)
    user = _decode_user(token, db)
    if user:
        revoke_user_sessions(user, db)
        logger.info("user_logged_out_sessions_revoked user_id=%s", user.id)

    response = JSONResponse(content={"success": True, "message": "Logged out successfully."})
    clear_session_cookie(response)
    return response


@app.get("/api/auth/me")
def me(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    displayName = user.username or user.email.split("@")[0]
    client_ip = request.client.host if request.client else "unknown"
    tokens = get_user_token_status(user, client_ip, db)
    return {
        "id": user.id,
        "email": user.email,
        "username": displayName,
        "phone": user.phone or "",
        "role": user.role,
        "tokens": tokens,
    }


# ==========================================
# Conversation History Navigation Endpoints
# ==========================================
@app.get("/api/conversations")
def get_conversations(user: User = Depends(get_optional_user), db: Session = Depends(get_db)) -> dict:
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
def create_conversation(user: User = Depends(get_optional_user), db: Session = Depends(get_db)) -> dict:
    conv = Conversation(user_id=user.id, title="New Conversation", messages=[])
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv.public()


@app.get("/api/conversations/{conv_id}")
def get_conversation_detail(conv_id: str, user: User = Depends(get_optional_user), db: Session = Depends(get_db)) -> dict:
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
def delete_conversation(conv_id: str, user: User = Depends(get_optional_user), db: Session = Depends(get_db)) -> dict:
    conv = db.scalar(select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user.id))
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return {"success": True}


@app.get("/api/documents")
def documents(user: User = Depends(get_optional_user), db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(
        select(Document).where(Document.owner_id == user.id).order_by(Document.created_at.desc())
    ).all()
    return {"documents": [row.public() for row in rows], "total_documents": len(rows)}


def validate_uploaded_file_safety(filename: str, content: bytes) -> None:
    if not content:
        raise HTTPException(400, "Uploaded file cannot be empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(400, "File size exceeds the allowed limit.")

    raw_filename = Path(filename).name
    # Disallow directory traversal sequences and control characters
    if not raw_filename or ".." in raw_filename or "/" in raw_filename or "\\" in raw_filename:
        raise HTTPException(400, "Invalid filename provided.")

    extension = Path(raw_filename).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(400, f"Unsupported file type '{extension}'. Supported: {', '.join(settings.allowed_extensions)}")

    # Reject executable binary file headers immediately
    if content.startswith(b"MZ") or content.startswith(b"\x7fELF") or content.startswith(b"\xca\xfe\xba\xbe") or content.startswith(b"\xfe\xed\xfa"):
        raise HTTPException(400, "Executable binary files are strictly prohibited.")

    # Validate file signatures (magic bytes)
    if extension == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(400, "Invalid PDF file: corrupted or mismatched file signature.")
    elif extension == ".png":
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(400, "Invalid PNG file: corrupted or mismatched file signature.")
    elif extension in (".jpg", ".jpeg"):
        if not content.startswith(b"\xff\xd8\xff"):
            raise HTTPException(400, "Invalid JPEG file: corrupted or mismatched file signature.")
    elif extension == ".webp":
        if not (content.startswith(b"RIFF") and b"WEBP" in content[:16]):
            raise HTTPException(400, "Invalid WebP file: corrupted or mismatched file signature.")
    elif extension in (".zip", ".docx", ".xlsx", ".pptx"):
        if not (content.startswith(b"PK\x03\x04") or content.startswith(b"PK\x05\x06") or content.startswith(b"PK\x07\x08")):
            raise HTTPException(400, f"Invalid archive/document file '{extension}': corrupted or mismatched file signature.")
    elif extension in (".gif",):
        if not (content.startswith(b"GIF87a") or content.startswith(b"GIF89a")):
            raise HTTPException(400, "Invalid GIF file signature.")
    elif extension in (".bmp",):
        if not content.startswith(b"BM"):
            raise HTTPException(400, "Invalid BMP file signature.")
    else:
        # Text/code/markdown/json files: check for binary null bytes
        sample = content[:4096]
        if b"\x00" in sample:
            raise HTTPException(400, f"File '{extension}' contains prohibited binary control characters.")


@app.post("/api/upload")
async def upload_document(
    user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
) -> dict:
    filename = file.filename or "uploaded-file.txt"
    content = await file.read()
    validate_uploaded_file_safety(filename, content)

    # Sanitize document name for safe display
    safe_name = re.sub(r'[^\w\s.-]', '_', Path(filename).name)[:120]
    document = Document(owner_id=user.id, name=safe_name, status="processing")
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
        raise HTTPException(422, "The document could not be processed. Please ensure it is a valid, uncorrupted document.") from exc


@app.delete("/api/documents/{document_id}")
def delete_document(
    document_id: str,
    user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
) -> dict:
    document = db.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if not document:
        raise HTTPException(404, "Document not found")
    rag.delete_document(document.id, user.id)
    db.delete(document)
    db.commit()
    return {"success": True}


@app.post("/api/chat/upload-image")
async def upload_chat_image(
    request: Request,
    user: User = Depends(get_optional_user),
) -> dict:
    """Upload an image from camera snapshot or file picker for multimodal chat."""
    content_type = request.headers.get("content-type", "")
    raw_bytes = b""
    orig_filename = "photo.jpg"

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file:
            raise HTTPException(400, "No file field found in form data.")
        raw_bytes = await uploaded_file.read()
        orig_filename = getattr(uploaded_file, "filename", "upload.jpg")
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON payload.")
        image_data = body.get("image_data", "")
        orig_filename = body.get("filename", "camera.jpg")
        if not image_data:
            raise HTTPException(400, "Missing image_data.")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(image_data)
        except Exception:
            raise HTTPException(400, "Invalid base64 image data.")

    if not raw_bytes:
        raise HTTPException(400, "Empty image uploaded.")
    if len(raw_bytes) > 25 * 1024 * 1024:
        raise HTTPException(400, "Image exceeds 25 MB limit.")

    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(raw_bytes))
        im.verify()
    except Exception:
        raise HTTPException(400, "Invalid image format.")

    filename = f"chat_{user.id[:8]}_{uuid.uuid4().hex[:12]}.jpg"
    local_path = settings.image_dir / filename
    im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if max(im.size) > 1600:
        im.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    im.save(local_path, format="JPEG", quality=90)

    image_url = f"/api/images/{filename}"
    return {
        "success": True,
        "url": image_url,
        "image_url": image_url,
        "filename": filename,
    }


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    req: Request,
    user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
) -> dict:
    started = time.perf_counter()
    client_ip = req.client.host if req.client else "unknown"

    # Pre-check token availability before heavy inference
    current_status = get_user_token_status(user, client_ip, db)
    if current_status["out_of_tokens"]:
        msg = (
            "Out of tokens. You have used all 20 tokens for today."
            if current_status["is_logged_in"]
            else "Out of tokens. You have used all 5 free guest tokens for today. Please log in to get 20 tokens per day."
        )
        raise HTTPException(
            status_code=429,
            detail=msg,
        )

    if request.selected_doc_id:
        allowed = db.scalar(
            select(Document.id).where(Document.id == request.selected_doc_id, Document.owner_id == user.id)
        )
        if not allowed:
            raise HTTPException(404, "Selected document not found")

    clean_message = request.message.strip()

    # Normalize mode — only accept known modes; anything else falls back to 'general'
    effective_mode = request.mode if request.mode in ("general", "code") else "general"

    # Persist base64 image data to local image file if present
    image_url_to_save = request.image_url
    if request.image_data and not image_url_to_save:
        try:
            from PIL import Image
            import io
            import base64
            clean_b64 = re.sub(r"^data:image/[^;]+;base64,", "", request.image_data)
            raw_b = base64.b64decode(clean_b64)
            im = Image.open(io.BytesIO(raw_b)).convert("RGB")
            if max(im.size) > 1600:
                im.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            saved_fn = f"chat_{user.id[:8]}_{uuid.uuid4().hex[:12]}.jpg"
            im.save(settings.image_dir / saved_fn, format="JPEG", quality=90)
            image_url_to_save = f"/api/images/{saved_fn}"
        except Exception as e_save_img:
            logger.warning("save_chat_image_failed error=%s", e_save_img)

    # Load or initialize conversation session strictly scoped to user (prevent IDOR and existence disclosure)
    conv = None
    if request.conversation_id:
        existing_conv = db.scalar(
            select(Conversation).where(
                Conversation.id == request.conversation_id,
                Conversation.user_id == user.id,
            )
        )
        if not existing_conv:
            raise HTTPException(404, "Conversation not found")
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
            image_data=request.image_data,
            image_url=image_url_to_save,
        )
    except Exception as exc:
        logger.exception("chat_failed user_id=%s", user.id)
        REQUEST_COUNT.labels(status="error", model="unknown").inc()
        raise HTTPException(503, "The assistant service is temporarily unavailable. Please try again.") from exc

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

    # Persist message to user's conversation session with temporary state flag and image references
    curr_msgs = list(conv.messages or [])
    if not curr_msgs or conv.title == "New Conversation":
        conv.title = clean_message[:40] + ("…" if len(clean_message) > 40 else "")

    is_temp = bool(request.incognito)
    user_msg_entry = {
        "role": "user",
        "text": clean_message,
        "image_url": image_url_to_save,
        "meta": ""
    }
    assistant_msg_entry = {
        "role": "assistant",
        "text": answer["answer"],
        "meta": str(total_ms),
        "sources": answer["sources"],
        "image_url": answer.get("image_url"),
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

    token_status = consume_user_token(user, client_ip, db)

    return {
        **answer,
        "image_url": image_url_to_save or answer.get("image_url"),
        "is_chemistry": is_chem,
        "conversation_id": conv.id,
        "latency_ms": total_ms,
        "timings_ms": timings,
        "token_status": token_status,
        "tokens_remaining": token_status["tokens_remaining"],
        "daily_limit": token_status["daily_limit"],
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
        '  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\n'
        '  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>\n'
        '  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>\n'
    )

    if has_html:
        doc = cleaned_html
        # Remove dead relative link tags like <link rel="stylesheet" href="styles.css"> or script.js so they don't 404
        doc = re.sub(r'<link[^>]+href=["\'](?:styles?\.css|main\.css|style\.css)["\'][^>]*>', '', doc, flags=re.I)
        doc = re.sub(r'<script[^>]+src=["\'](?:scripts?\.js|main\.js|app\.js)["\'][^>]*>\s*</script>', '', doc, flags=re.I)
        if "three.min.js" not in doc:
            if has_head:
                doc = re.sub(r"(<head[^>]*>)", f"\\1\n{font_head_tags}", doc, count=1, flags=re.I)
            else:
                doc = re.sub(r"(<html[^>]*>)", f"\\1\n<head>{font_head_tags}</head>", doc, count=1, flags=re.I)
                has_head = True
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
    html, body {{
      overflow-x: hidden;
      width: 100%;
      max-width: 100vw;
    }}
    body {{
      margin: 0;
      padding: 0;
      font-family: 'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      line-height: 1.5;
    }}
    canvas {{
      display: block;
      max-width: 100%;
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
    # Strict regex validation: only valid image filenames allowed
    if not re.match(r"^[a-zA-Z0-9_\-]+\.(?:jpg|jpeg|png|webp)$", safe_filename, re.I):
        raise HTTPException(404, "Image not found")
    image_path = (settings.image_dir / safe_filename).resolve()
    if not str(image_path).startswith(str(settings.image_dir.resolve())):
        raise HTTPException(404, "Image not found")
    if not image_path.is_file():
        raise HTTPException(404, "Image not found")
    media_type = (
        "image/png" if safe_filename.lower().endswith(".png")
        else ("image/webp" if safe_filename.lower().endswith(".webp") else "image/jpeg")
    )
    return FileResponse(image_path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/healthcheck")
@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
