from __future__ import annotations

import base64
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Detect base directory (support both PyInstaller bundle and regular python script)
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    BUNDLE_DIR = Path(sys._MEIPASS)
    APP_DIR = Path(sys.executable).parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    APP_DIR = BUNDLE_DIR

# Load .env from App directory or Bundle directory
if (APP_DIR / ".env").exists():
    load_dotenv(APP_DIR / ".env", override=False)
elif (BUNDLE_DIR / ".env").exists():
    load_dotenv(BUNDLE_DIR / ".env", override=False)
else:
    load_dotenv(override=False)


def env_list(name: str, default: str) -> list[str]:
    return [value.strip() for value in os.getenv(name, default).split(",") if value.strip()]


import secrets

@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "production")
    database_url: str = os.getenv("DATABASE_URL", "")
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    jwt_access_token_ttl_seconds: int = int(os.getenv("JWT_ACCESS_TOKEN_TTL_SECONDS", "86400"))  # 24 hours
    cors_origins: list[str] = None  # type: ignore[assignment]
    
    # LLM Provider Configuration
    # Accepts xAI Grok (xai-...), Grok API key (GROK_API_KEY), or Groq Cloud (gsk_...)
    xai_api_key: str = os.getenv("GROK_API_KEY", os.getenv("XAI_API_KEY", ""))
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    grok_model: str = os.getenv("GROK_MODEL", "openai/gpt-oss-120b")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    # Embedding and Reranking Configuration
    use_neural_models: bool = os.getenv("USE_NEURAL_MODELS", "false").lower() in ("true", "1", "yes")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    
    # Qdrant Vector DB Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    
    # Email (SMTP) Dispatch Configuration
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "noreply@aster.local"))
    smtp_use_tls: bool = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

    # SMS (Fast2SMS / Twilio Gateway) Dispatch Configuration
    fast2sms_api_key: str = os.getenv("FAST2SMS_API_KEY", "")
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")

    # Scalability & Concurrency Settings
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "20"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "40"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(400 * 1024 * 1024)))
    allowed_extensions: frozenset[str] = frozenset({
        # Documents
        ".pdf", ".doc", ".docx", ".txt", ".rtf", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
        # Spreadsheets
        ".xls", ".xlsx", ".ods",
        # Presentations
        ".ppt", ".pptx", ".odp",
        # Images (OCR & vision indexing)
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff",
        # Code / Source files
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".h", ".hpp", ".c", ".cs",
        ".go", ".rs", ".php", ".rb", ".sql", ".html", ".css", ".scss", ".sh", ".toml", ".ini", ".log",
        # Archives
        ".zip",
    })

    # Persona & Professional/Formal Settings
    research_mode: bool = os.getenv("RESEARCH_MODE", "false").lower() in ("true", "1", "yes")
    humor_level: str = os.getenv("HUMOR_LEVEL", "off")
    humor_temperature: float = float(os.getenv("HUMOR_TEMPERATURE", "0.2"))

    # Multimodal Vision Configuration
    vision_model: str = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
    inpainting_provider: str = os.getenv("INPAINTING_PROVIDER", "auto")

    # Image Generation Configuration
    image_model: str = os.getenv("IMAGE_MODEL", "flux")
    image_width: int = int(os.getenv("IMAGE_WIDTH", "1024"))
    image_height: int = int(os.getenv("IMAGE_HEIGHT", "1024"))

    def __post_init__(self):
        # Validate or securely generate JWT_SECRET
        INSECURE_SECRETS = {
            "production-secret-aster-enterprise-secure-2026-key-salt",
            "your-production-super-secret-jwt-key-change-this-now",
            "changeme",
            "secret",
            "",
        }
        current_secret = (self.jwt_secret or "").strip()
        is_prod = self.app_env.lower() in ("production", "prod")
        if current_secret in INSECURE_SECRETS or len(current_secret) < 32:
            if is_prod:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: "
                    "JWT_SECRET must be explicitly set to a cryptographically strong key "
                    "(minimum 32 characters) in production mode."
                )
            # Ephemeral random secret for development if not provided or insecure
            ephemeral = secrets.token_urlsafe(48)
            object.__setattr__(self, "jwt_secret", ephemeral)
        else:
            object.__setattr__(self, "jwt_secret", current_secret)

        cors_val = os.getenv("CORS_ORIGINS", "").strip()
        if not cors_val or cors_val == "*":
            # For development, allow local dev servers; never open wildcard in production
            if is_prod:
                object.__setattr__(self, "cors_origins", [])
            else:
                object.__setattr__(self, "cors_origins", ["http://localhost:8000", "http://127.0.0.1:8000"])
        else:
            parsed = [v.strip() for v in cors_val.split(",") if v.strip() and v.strip() != "*"]
            object.__setattr__(self, "cors_origins", parsed)
        if not self.database_url:
            db_file = self.data_dir / "aster.db"
            object.__setattr__(self, "database_url", f"sqlite:///{db_file.as_posix()}")

    @property
    def active_llm_key(self) -> str:
        """Returns the active key, preferring xai_api_key if set, then groq_api_key."""
        if self.xai_api_key.strip():
            return self.xai_api_key.strip()
        if self.groq_api_key.strip():
            return self.groq_api_key.strip()
        return ""

    @property
    def llm_provider(self) -> str:
        """Auto-detect provider based on key format."""
        key = self.active_llm_key
        if key.startswith("gsk_"):
            return "groq"
        if key.startswith("xai-"):
            return "xai"
        if key.startswith("sk-or-"):
            return "openrouter"
        if key.startswith("sk-"):
            return "openai"
        if self.groq_api_key:
            return "groq"
        return "xai"

    @property
    def llm_endpoint(self) -> str:
        if self.llm_provider == "groq":
            return "https://api.groq.com/openai/v1/chat/completions"
        if self.llm_provider == "openrouter":
            return "https://openrouter.ai/api/v1/chat/completions"
        if self.llm_provider == "openai":
            return "https://api.openai.com/v1/chat/completions"
        return "https://api.x.ai/v1/chat/completions"

    @property
    def active_model(self) -> str:
        if self.llm_provider == "groq":
            return self.groq_model or "openai/gpt-oss-120b"
        return self.grok_model or "grok-2-latest"

    @property
    def grok_configured(self) -> bool:
        return bool(self.active_llm_key)

    @property
    def data_dir(self) -> Path:
        path = APP_DIR / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def image_dir(self) -> Path:
        path = self.data_dir / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
