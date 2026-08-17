"""Central application configuration.

Every secret is read from the environment (or a local ``.env`` file). Nothing
sensitive is ever hard-coded, and nothing sensitive is ever returned by an API
route -- see :meth:`Settings.public_snapshot`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Runtime configuration for the RAGX backend."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Application --------------------------------------------------------
    app_name: str = "RAGX"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = False

    # -- CORS ---------------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # -- Optional API-key gate on mutating routes ---------------------------
    # Empty means "open" (fine for local development). Set it in production.
    ragx_api_key: str = ""

    # -- Cloud LLM providers (NO local LLMs are supported by design) --------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_reasoning_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"

    primary_llm_provider: Literal["gemini", "groq"] = "gemini"
    fallback_llm_provider: Literal["gemini", "groq", "none"] = "groq"
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2

    # Pricing (USD per 1M tokens) used for *estimated* cost accounting only.
    gemini_input_cost_per_mtok: float = 0.10
    gemini_output_cost_per_mtok: float = 0.40
    groq_input_cost_per_mtok: float = 0.59
    groq_output_cost_per_mtok: float = 0.79

    # -- Embeddings ---------------------------------------------------------
    # "gemini"    -> Gemini embedding API (production default)
    # "hashing"   -> deterministic offline embedder, DEV/TEST ONLY. It carries
    #                no semantic knowledge; it exists so the pipeline, tests and
    #                CI can run without network access. Never use in production.
    embedding_provider: Literal["gemini", "hashing"] = "gemini"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32

    # -- Vector store (Qdrant) ---------------------------------------------
    # When ``qdrant_url`` is empty the client runs in embedded local mode,
    # persisting to ``qdrant_path``. Both are genuine Qdrant.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = str(BACKEND_ROOT / "data" / "qdrant")
    qdrant_collection: str = "ragx_chunks"

    # -- Graph store (Neo4j, with an embedded NetworkX fallback) -----------
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    graph_fallback_path: str = str(BACKEND_ROOT / "data" / "graph" / "graph.json")

    # -- Relational database ------------------------------------------------
    # PostgreSQL in production; SQLite is the zero-infrastructure dev default.
    database_url: str = ""
    postgres_host: str = ""
    postgres_port: int = 5432
    postgres_user: str = "ragx"
    postgres_password: str = ""
    postgres_db: str = "ragx"
    db_echo: bool = False

    # -- Object storage -----------------------------------------------------
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_path: str = str(BACKEND_ROOT / "data" / "objects")
    s3_bucket: str = ""
    s3_endpoint_url: str = ""
    s3_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # -- Ingestion ----------------------------------------------------------
    max_upload_mb: int = 50
    allowed_extensions: str = ".pdf,.docx,.txt,.md,.csv,.png,.jpg,.jpeg,.webp,.tiff"
    chunk_target_tokens: int = 380
    chunk_overlap_tokens: int = 60
    chunk_min_tokens: int = 60
    enable_ocr: bool = True
    ocr_min_chars_per_page: int = 120
    extract_entities: bool = True
    entity_extraction_max_chunks: int = 60

    # -- Retrieval ----------------------------------------------------------
    default_top_k: int = 8
    candidate_pool_size: int = 40
    rrf_k: int = 60
    rerank_enabled: bool = True
    rerank_top_n: int = 24
    min_relevance_score: float = 0.28
    corrective_relevance_floor: float = 0.45
    corrective_max_rounds: int = 2
    agentic_max_steps: int = 6
    agentic_max_subqueries: int = 4
    retrieval_timeout_seconds: float = 60.0

    # -- Verification -------------------------------------------------------
    verification_enabled: bool = True
    min_claim_support_score: float = 0.5
    insufficient_evidence_threshold: float = 0.35

    # -- Caching ------------------------------------------------------------
    cache_enabled: bool = True
    cache_ttl_seconds: int = 900
    cache_max_entries: int = 512

    # ---------------------------------------------------------------- helpers
    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("qdrant_path", "storage_local_path", "graph_fallback_path")
    @classmethod
    def _resolve_data_path(cls, value: str) -> str:
        """Anchor relative data paths to the backend directory.

        Users start the server from either the repository root or ``backend/``,
        and may keep ``.env`` in either place. Resolving against the current
        working directory would silently create a second, empty data tree on the
        wrong side of that choice -- so relative paths are always anchored to
        ``backend/`` instead. A leading ``./backend/`` is tolerated for configs
        written relative to the repository root.
        """
        if not value:
            return value
        path = Path(value)
        if path.is_absolute():
            return str(path)
        parts = path.parts
        if parts and parts[0] == "backend":
            path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        return str((BACKEND_ROOT / path).resolve())

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extension_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def sqlalchemy_url(self) -> str:
        """Resolve the async SQLAlchemy URL.

        Precedence: explicit ``DATABASE_URL`` -> assembled PostgreSQL DSN ->
        local SQLite file. The SQLite path keeps the project runnable with no
        infrastructure while remaining a real, migrated relational store.
        """
        if self.database_url:
            url = self.database_url
            # Accept the common sync-style DSNs and upgrade them to async.
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif url.startswith("sqlite://") and "+aiosqlite" not in url:
                url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)
            return url
        if self.postgres_host:
            return (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        db_file = BACKEND_ROOT / "data" / "ragx.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{db_file.as_posix()}"

    @property
    def uses_postgres(self) -> bool:
        return self.sqlalchemy_url.startswith("postgresql")

    def ensure_directories(self) -> None:
        for path in (
            Path(self.storage_local_path),
            Path(self.qdrant_path),
            Path(self.graph_fallback_path).parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ safe
    def public_snapshot(self) -> dict:
        """Non-sensitive configuration safe to hand to the frontend.

        No API keys, passwords, DSNs or credentials appear here.
        """
        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "llm": {
                "primary_provider": self.primary_llm_provider,
                "fallback_provider": self.fallback_llm_provider,
                "gemini_model": self.gemini_model,
                "gemini_embedding_model": self.gemini_embedding_model,
                "groq_model": self.groq_model,
                "groq_fast_model": self.groq_fast_model,
                "gemini_configured": bool(self.gemini_api_key),
                "groq_configured": bool(self.groq_api_key),
            },
            "embeddings": {
                "provider": self.embedding_provider,
                "dimension": self.embedding_dimension,
                "development_only": self.embedding_provider == "hashing",
            },
            "storage": {
                "vector_store": "qdrant",
                "vector_mode": "server" if self.qdrant_url else "embedded",
                "graph_store": "neo4j" if self.neo4j_uri else "networkx-embedded",
                "relational": "postgresql" if self.uses_postgres else "sqlite",
                "objects": self.storage_backend,
            },
            "retrieval": {
                "default_top_k": self.default_top_k,
                "candidate_pool_size": self.candidate_pool_size,
                "rerank_enabled": self.rerank_enabled,
                "min_relevance_score": self.min_relevance_score,
                "corrective_relevance_floor": self.corrective_relevance_floor,
                "corrective_max_rounds": self.corrective_max_rounds,
                "agentic_max_steps": self.agentic_max_steps,
            },
            "verification": {
                "enabled": self.verification_enabled,
                "min_claim_support_score": self.min_claim_support_score,
                "insufficient_evidence_threshold": self.insufficient_evidence_threshold,
            },
            "ingestion": {
                "max_upload_mb": self.max_upload_mb,
                "allowed_extensions": sorted(self.allowed_extension_set),
                "ocr_enabled": self.enable_ocr,
                "entity_extraction": self.extract_entities,
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


def reload_settings() -> Settings:
    """Clear the cache and re-read the environment (used by tests)."""
    get_settings.cache_clear()
    return get_settings()


# Convenience for modules that only need read access at import time.
settings = get_settings()

__all__ = ["Settings", "get_settings", "reload_settings", "settings", "BACKEND_ROOT", "PROJECT_ROOT", "os"]
