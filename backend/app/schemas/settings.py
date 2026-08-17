"""Settings and system-status schemas.

Nothing in this module carries a secret. Provider *configuration state* is
exposed as a boolean; keys, DSNs and passwords never leave the backend.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProviderStatus(BaseModel):
    provider: str
    configured: bool = False
    model: str | None = None
    fast_model: str | None = None
    embedding_model: str | None = None
    multimodal: bool = False
    kind: Literal["cloud"] = "cloud"
    healthy: bool | None = None
    status_text: str = "unknown"
    latency_ms: float | None = None
    error: str | None = None


class LLMStatusResponse(BaseModel):
    providers: list[ProviderStatus] = Field(default_factory=list)
    primary: str = "gemini"
    fallback: str = "groq"
    any_configured: bool = False
    local_llms_supported: bool = False
    note: str = "RAGX uses cloud LLM APIs only. Local model runtimes are not supported."


class EmbeddingStatus(BaseModel):
    provider: str
    model: str | None = None
    dimension: int = 768
    production_ready: bool = True
    healthy: bool | None = None
    status_text: str = "unknown"
    warning: str | None = None


class StorageStatus(BaseModel):
    vector_store: dict[str, Any] = Field(default_factory=dict)
    graph_store: dict[str, Any] = Field(default_factory=dict)
    bm25_index: dict[str, Any] = Field(default_factory=dict)
    relational: dict[str, Any] = Field(default_factory=dict)
    object_storage: dict[str, Any] = Field(default_factory=dict)


class RetrievalSettings(BaseModel):
    default_top_k: int = 8
    candidate_pool_size: int = 40
    rerank_enabled: bool = True
    min_relevance_score: float = 0.28
    corrective_relevance_floor: float = 0.45
    corrective_max_rounds: int = 2
    agentic_max_steps: int = 6


class VerificationSettings(BaseModel):
    enabled: bool = True
    min_claim_support_score: float = 0.5
    insufficient_evidence_threshold: float = 0.35


class IngestionSettings(BaseModel):
    max_upload_mb: int = 50
    allowed_extensions: list[str] = Field(default_factory=list)
    ocr_enabled: bool = True
    ocr_available: bool = False
    entity_extraction: bool = True
    chunk_target_tokens: int = 380
    chunk_overlap_tokens: int = 60


class StrategyInfo(BaseModel):
    name: str
    label: str
    description: str
    uses_llm: bool = False


class SettingsResponse(BaseModel):
    app_name: str
    version: str
    environment: str
    llm: LLMStatusResponse
    embeddings: EmbeddingStatus
    storage: StorageStatus
    retrieval: RetrievalSettings
    verification: VerificationSettings
    ingestion: IngestionSettings
    strategies: list[StrategyInfo] = Field(default_factory=list)
    cache: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RetrievalSettingsUpdate(BaseModel):
    """Runtime-tunable retrieval knobs. Credentials are never accepted here."""

    default_top_k: int | None = Field(default=None, ge=1, le=30)
    candidate_pool_size: int | None = Field(default=None, ge=5, le=200)
    rerank_enabled: bool | None = None
    min_relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    corrective_relevance_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    corrective_max_rounds: int | None = Field(default=None, ge=0, le=5)
    agentic_max_steps: int | None = Field(default=None, ge=1, le=10)
    verification_enabled: bool | None = None
    insufficient_evidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    primary_llm_provider: Literal["gemini", "groq"] | None = None
    fallback_llm_provider: Literal["gemini", "groq", "none"] | None = None


__all__ = [
    "ProviderStatus",
    "LLMStatusResponse",
    "EmbeddingStatus",
    "StorageStatus",
    "RetrievalSettings",
    "VerificationSettings",
    "IngestionSettings",
    "StrategyInfo",
    "SettingsResponse",
    "RetrievalSettingsUpdate",
]
