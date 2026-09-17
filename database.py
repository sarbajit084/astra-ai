from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config import settings

engine_kwargs: dict = {"pool_pre_ping": True}
if settings.database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_size"] = settings.db_pool_size
    engine_kwargs["max_overflow"] = settings.db_max_overflow

engine = create_engine(settings.database_url, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def new_id() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), default="")
    otp: Mapped[str | None] = mapped_column(String(10), nullable=True)
    otp_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="user", index=True)
    password_reset_token: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    password_reset_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    google_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    token_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="New Conversation")
    chat_type: Mapped[str] = mapped_column(String(20), default="normal", index=True)
    messages: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
        Index("ix_conversations_user_type", "user_id", "chat_type"),
    )

    def public(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "chat_type": self.chat_type or "normal",
            "messages": self.messages or [],
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(30), default="processing")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    character_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_documents_owner_created", "owner_id", "created_at"),
    )

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "chunks_count": self.chunk_count,
            "chars_count": self.character_count,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


class UserPreference(Base):
    __tablename__ = "user_preferences"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def public(self) -> dict:
        return dict(self.payload or {})


class QueryEvent(Base):
    __tablename__ = "query_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True, index=True)
    query: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, index=True)
    timings: Mapped[dict] = mapped_column(JSON, default=dict)
    model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_query_events_user_created", "user_id", "created_at"),
    )

    def public(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "document_id": self.document_id,
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_count": self.retrieval_count,
            "latency_ms": round(self.latency_ms, 1),
            "timings_ms": self.timings,
            "model": self.model,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
    # Ensure any missing columns from older SQLite database versions are gracefully migrated
    with engine.connect() as conn:
        for col_def in [
            "ALTER TABLE users ADD COLUMN username VARCHAR(100)",
            "ALTER TABLE users ADD COLUMN phone VARCHAR(30)",
            "ALTER TABLE users ADD COLUMN otp VARCHAR(10)",
            "ALTER TABLE users ADD COLUMN otp_expires_at DATETIME",
            "ALTER TABLE users ADD COLUMN password_reset_token VARCHAR(128)",
            "ALTER TABLE users ADD COLUMN password_reset_expires_at DATETIME",
            "ALTER TABLE users ADD COLUMN google_id VARCHAR(128)",
            "ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0",
            "ALTER TABLE users ADD COLUMN locked_until DATETIME",
            "ALTER TABLE users ADD COLUMN token_version INTEGER DEFAULT 1",
            "ALTER TABLE conversations ADD COLUMN chat_type VARCHAR(20) DEFAULT 'normal'",
        ]:
            try:
                conn.execute(text(col_def))
                conn.commit()
            except Exception:
                pass


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
