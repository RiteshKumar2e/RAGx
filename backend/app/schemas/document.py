"""Document and knowledge-base schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel


class ProcessingStep(BaseModel):
    status: str = "pending"
    duration_ms: float = 0.0
    detail: str | None = None


class DocumentSummary(ORMModel):
    id: str
    filename: str
    title: str | None = None
    file_type: str
    size_bytes: int = 0
    status: str
    status_detail: str | None = None
    error_message: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    table_count: int = 0
    figure_count: int = 0
    entity_count: int = 0
    token_count: int = 0
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None = None
    processing_ms: float = 0.0
    processing_steps: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _status_value(cls, v: Any) -> str:
        return v.value if hasattr(v, "value") else str(v)


class ChunkPreview(ORMModel):
    id: str
    ordinal: int
    content: str
    modality: str
    token_count: int = 0
    page_number: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_path: list[str] = Field(default_factory=list)
    figure_label: str | None = None
    table_label: str | None = None
    asset_key: str | None = None
    indexed_in_vector_store: bool = False

    @field_validator("modality", mode="before")
    @classmethod
    def _modality_value(cls, v: Any) -> str:
        return v.value if hasattr(v, "value") else str(v)


class EntityPreview(ORMModel):
    id: str
    name: str
    entity_type: str
    description: str | None = None
    salience: float = 0.0
    mention_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)


class DocumentDetail(DocumentSummary):
    mime_type: str | None = None
    checksum: str | None = None
    storage_backend: str | None = None
    source_url: str | None = None
    doc_metadata: dict[str, Any] = Field(default_factory=dict)
    outline: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[ChunkPreview] = Field(default_factory=list)
    entities: list[EntityPreview] = Field(default_factory=list)
    modality_breakdown: dict[str, int] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    items: list[DocumentSummary]
    total: int
    page: int = 1
    page_size: int = 20
    status_counts: dict[str, int] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    duplicate_of: str | None = None


class BulkUploadResponse(BaseModel):
    uploaded: list[UploadResponse] = Field(default_factory=list)
    rejected: list[dict[str, str]] = Field(default_factory=list)


class ReindexResponse(BaseModel):
    document_id: str
    status: str
    message: str


class KnowledgeBaseStats(BaseModel):
    total_documents: int = 0
    indexed_documents: int = 0
    processing_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    total_pages: int = 0
    total_tables: int = 0
    total_figures: int = 0
    total_entities: int = 0
    total_relations: int = 0
    vectors_indexed: int = 0
    bm25_documents: int = 0
    storage_bytes: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    by_modality: dict[str, int] = Field(default_factory=dict)


class WebIngestRequest(BaseModel):
    url: str
    title: str | None = None

    @field_validator("url")
    @classmethod
    def _http_only(cls, v: str) -> str:
        if not v.lower().startswith(("http://", "https://")):
            raise ValueError("Only http:// and https:// URLs can be ingested.")
        return v


__all__ = [
    "ProcessingStep",
    "DocumentSummary",
    "DocumentDetail",
    "ChunkPreview",
    "EntityPreview",
    "DocumentListResponse",
    "UploadResponse",
    "BulkUploadResponse",
    "ReindexResponse",
    "KnowledgeBaseStats",
    "WebIngestRequest",
]
