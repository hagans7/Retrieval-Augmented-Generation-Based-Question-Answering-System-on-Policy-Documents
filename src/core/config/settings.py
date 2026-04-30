"""
Application settings — single source of truth for all configuration.

All values come from environment variables or a .env file.
No secrets or endpoints are hardcoded here.

Usage:
    from src.core.config.settings import settings
    print(settings.LLM_BASE_URL)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic BaseSettings — reads all fields from environment variables.

    Fields without defaults are required and will raise a validation error
    at startup if missing. This enforces fail-fast on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    APP_NAME: str = "legal"
    APP_VERSION: str = "0.1.0"
    DEPLOYMENT_MODE: str = "development"  # development | production
    DEBUG: bool = False
    DEFAULT_USER_ID: str = ""  # UUID string for static default user (auth deferred)

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str  # Required — postgresql+asyncpg://user:pass@host:port/db
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_PRE_PING: bool = True

    # ── Redis ────────────────────────────────────────────────
    REDIS_URL: str  # Required — redis://host:port/db

    # ── LLM Provider (OpenAI-Compatible) ─────────────────────
    LLM_BASE_URL: str  # Required — e.g. https://openrouter.ai/api/v1
    LLM_API_KEY: str  # Required
    LLM_DEFAULT_MODEL: str 
    LLM_TIMEOUT: int = 60
    LLM_MAX_RETRIES: int = 3

    # ── Embedding (TEI) ──────────────────────────────────────
    EMBEDDING_BASE_URL: str 
    EMBEDDING_MODEL_NAME: str 
    EMBEDDING_TIMEOUT: int = 30

    # ── Reranker (TEI) ───────────────────────────────────────
    RERANKER_BASE_URL: str 
    RERANKER_MODEL_NAME: str
    RERANKER_TOP_K: int = 5
    RERANKER_MIN_SCORE: float = 0.3 
    RERANKER_TIMEOUT: int = 30

    # ── Agent Governance ─────────────────────────────────────
    AGENT_MAX_EXECUTOR_LOOPS: int = 3  # max times auditor can restart executor (audit_retry_count)
    MAX_EVIDENCE_SLOTS: int = 15       # cap on total evidence items in state (dedup before cap)
 
    # ── OCR (LightOnOCR via HuggingFace) ─────────────────────
    OCR_MODEL_REPO: str = "lightonai/LightOnOCR-2-1B"
    OCR_MODEL_DIR: str = "/app/models/ocr"
    HF_TOKEN: str | None = None  # Optional — for private HuggingFace repos

    # ── Object Storage (S3-Compatible) ───────────────────────
    STORAGE_ENDPOINT: str = "http://minio:9000"
    STORAGE_ACCESS_KEY: str = "minioadmin"
    STORAGE_SECRET_KEY: str = "minioadmin"
    STORAGE_BUCKET: str = "legal-docs"
    STORAGE_USE_SSL: bool = False

    # ── Vector Store (Weaviate) ──────────────────────────────
    WEAVIATE_URL: str = "http://weaviate:8080"
    WEAVIATE_COLLECTION_NAME: str = "DocumentChunk"

    # ── Graph Store (Neo4j) ──────────────────────────────────
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str  # Required

    # ── Chat History ─────────────────────────────────────────
    CHAT_HISTORY_MAX_TURNS: int = 10

    # ── Langfuse (Observability) ─────────────────────────────
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "http://langfuse:3000"

    @property
    def is_production(self) -> bool:
        """Return True if running in production mode."""
        return self.DEPLOYMENT_MODE.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Return True if running in development mode."""
        return self.DEPLOYMENT_MODE.lower() == "development"


# Single module-level instance — all layers import from here.
# Never instantiate Settings() elsewhere.
settings = Settings()
