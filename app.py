"""Production-grade Enterprise RAG API.

Stateless architecture:
- Auth / sessions: Stateless JWT
- Metadata & Query Events: SQL (PostgreSQL / SQLite)
- Dense Chunks: Qdrant Vector Store
- LLM Inference: Grok / Groq LPU engine
- Metrics: Prometheus /metrics
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from auth import (
    clear_session_cookie,
    create_access_token,
    get_current_user,
    get_optional_user,
    hash_password,
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
if cors_origins == ["*"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


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


class LoginPayload(BaseModel):
    phone: str | None = Field(default=None, max_length=30)
    email: str | None = Field(default=None, max_length=255)
    identifier: str | None = Field(default=None, max_length=255)
    password: str = Field(min_length=1, max_length=128)
    challenge_token: str = Field(min_length=1)
    calculation_result: int


class GoogleAuthPayload(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    name: str | None = None
    sub: str | None = None
    id_token: str | None = None


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
    """Generates two 2-digit random numbers (10-99) and signs the result in a secure token."""
    num1 = random.randint(10, 99)
    num2 = random.randint(10, 99)
    # Randomly choose addition or subtraction (or multiplication if friendly)
    op = random.choice(["+", "-"])
    if op == "+":
        correct_ans = num1 + num2
    else:
        # Keep result non-negative for convenience
        if num1 < num2:
            num1, num2 = num2, num1
        correct_ans = num1 - num2

    ts = int(datetime.now(timezone.utc).timestamp())
    msg = f"{num1}:{op}:{num2}:{correct_ans}:{ts}"
    sig = hmac.new(settings.jwt_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
    token = f"{msg}:{sig}"

    return {
        "challenge_token": token,
        "num1": num1,
        "num2": num2,
        "operation": op,
        "question": f"What is {num1} {op} {num2}?",
    }


def verify_math_challenge(token: str, user_answer: int, max_age_seconds: int = 300) -> bool:
    """Verifies that the math challenge token is authentic, non-expired, and answered correctly."""
    try:
        parts = token.split(":")
        if len(parts) != 6:
            return False
        num1, op, num2, expected_ans_str, ts_str, sig = parts
        msg = f"{num1}:{op}:{num2}:{expected_ans_str}:{ts_str}"
        expected_sig = hmac.new(settings.jwt_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return False
        
        ts = int(ts_str)
        now_ts = int(datetime.now(timezone.utc).timestamp())
        if (now_ts - ts) > max_age_seconds:
            return False
        
        return user_answer == int(expected_ans_str)
    except Exception:
        return False


def normalize_phone(val: str | None) -> str:
    if not val:
        return ""
    return val.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")


@app.get("/api/auth/challenge")
def get_calculation_challenge() -> dict:
    """Issues a calculation challenge between two 2-digit numbers."""
    return create_math_challenge()


@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register_user(payload: RegisterPayload, db: Annotated[Session, Depends(get_db)]) -> dict:
    # 1. Verify calculation result
    if not verify_math_challenge(payload.challenge_token, payload.calculation_result):
        raise HTTPException(
            status_code=400,
            detail="Incorrect calculation result! Please solve the math problem correctly to register."
        )

    email = payload.email.strip().lower()
    username = payload.username.strip()
    phone = normalize_phone(payload.phone)

    # Check existing
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

    logger.info("user_registered_math_verified id=%s email=%s username=%s", user.id, user.email, user.username)
    return token_response(user)


@app.post("/api/auth/login")
def login_user(payload: LoginPayload, db: Annotated[Session, Depends(get_db)]) -> dict:
    # 1. Verify calculation result
    if not verify_math_challenge(payload.challenge_token, payload.calculation_result):
        raise HTTPException(
            status_code=400,
            detail="Incorrect calculation result! Please solve the math problem correctly to log in."
        )

    phone = normalize_phone(payload.phone)
    email = (payload.email or "").strip().lower()
    ident = (payload.identifier or "").strip()

    if ident:
        if "@" in ident:
            email = ident.lower()
        elif ident.isdigit() and len(ident) >= 7:
            phone = normalize_phone(ident)
        else:
            # treat as username or email
            email = ident.lower()

    user = None
    if ident or email:
        search_val = (ident or email).strip().lower()
        user = db.scalar(select(User).where((func.lower(User.email) == search_val) | (func.lower(User.username) == search_val)))
    if not user and phone:
        user = db.scalar(select(User).where(User.phone == phone))

    if not user:
        raise HTTPException(status_code=404, detail="No account found with the provided details. Please register first.")

    logger.info("user_login_math_verified id=%s email=%s username=%s", user.id, user.email, user.username)
    return token_response(user)


@app.post("/api/auth/google")
def google_auth(payload: GoogleAuthPayload, db: Annotated[Session, Depends(get_db)]) -> dict:
    """Seamless One-Click Google Authentication."""
    email = payload.email.strip().lower()
    user = db.scalar(select(User).where(User.email == email))

    if not user:
        is_first_user = (db.scalar(select(func.count(User.id))) or 0) == 0
        username = payload.name or email.split("@")[0]
        user = User(
            email=email,
            username=username,
            role="admin" if is_first_user else "user",
            password_hash=hash_password(f"google-oauth-{uuid.uuid4()}"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("google_user_registered email=%s", email)
    else:
        if payload.name and not user.username:
            user.username = payload.name
            db.commit()
            db.refresh(user)

    return token_response(user)


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
    if not content or len(content) > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes // 1024 // 1024
        raise HTTPException(400, f"File must be between 1 byte and {max_mb} MB")

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

    # ── Backend Code Mode Enforcement ────────────────────────────────────────────
    # If user is in Quick or Extended mode (not code mode) and asks for code → reject.
    if effective_mode != "code":
        code_verbs = re.compile(
            r"\b(write|create|build|implement|generate|make|code|program|develop|script|design)\b",
            re.IGNORECASE,
        )
        code_langs = re.compile(
            r"\b(python|javascript|typescript|java|c\+\+|cpp|c#|csharp|rust|go|golang|php|ruby|kotlin|swift|sql|bash|shell|html|css|react|node|django|flask|express|vue|angular)\b",
            re.IGNORECASE,
        )
        code_nouns = re.compile(
            r"\b(function|class|method|algorithm|snippet|program|script|code|api|endpoint|component|module|library|implementation|binary search|linked list|factorial|fibonacci|sort|recursion|loop|array|stack|queue|tree|graph)\b",
            re.IGNORECASE,
        )
        has_verb = bool(code_verbs.search(clean_message))
        has_lang = bool(code_langs.search(clean_message))
        has_noun = bool(code_nouns.search(clean_message))

        is_code_request = (
            (has_verb and (has_lang or has_noun))
            or (has_lang and has_noun)
            or bool(re.search(r"\b(give me|show me|write)\s+(a\s+)?(code|program|script|implementation|function|class)\b", clean_message, re.IGNORECASE))
            or bool(re.search(r"\b(debug|fix|refactor|explain this code|optimize this code)\b", clean_message, re.IGNORECASE))
        )

        if is_code_request:
            return {
                "answer": "Code can only be genearte in code mode",
                "sources": [],
                "metrics": {},
                "latency_ms": 1.0,
                "timings_ms": {"total": 1.0},
                "conversation_id": request.conversation_id,
                "model_used": "Astra",
                "rewritten_query": clean_message,
            }

    # Load or initialize conversation session for human-like multi-turn context
    conv = None
    if request.conversation_id:
        conv = db.get(Conversation, request.conversation_id)
        if conv:
            # If conversation exists but belonged to previous guest/user, adopt it safely for current user
            if conv.user_id != user.id:
                conv.user_id = user.id

    if not conv:
        conv_id = request.conversation_id
        # Double check if conv_id already exists in db
        if conv_id and db.get(Conversation, conv_id):
            conv = db.get(Conversation, conv_id)
            conv.user_id = user.id
        else:
            conv = Conversation(
                id=conv_id or str(uuid.uuid4()),
                user_id=user.id,
                title=clean_message[:40] + ("…" if len(clean_message) > 40 else ""),
                messages=[],
            )
            db.add(conv)
            try:
                db.flush()
            except Exception:
                db.rollback()
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

    if has_html:
        doc = cleaned_html
        if style_tag:
            if has_head:
                doc = re.sub(r"(</head>)", f"{style_tag}\\1", doc, count=1, flags=re.I)
            else:
                doc = re.sub(r"(<html[^>]*>)", f"\\1\n<head>{style_tag}</head>", doc, count=1, flags=re.I)
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
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 0;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
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
