"""Document, chunk and entity ORM models.

Large binaries never land in PostgreSQL -- ``Document.object_key`` points at the
object-storage backend. Only metadata, extracted text and provenance live here.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    GRAPH_INDEXING = "graph_indexing"
    READY = "ready"
    FAILED = "failed"


class ChunkModality(str, enum.Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"
    OCR = "ocr"
    CODE = "code"


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "documents"

    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    mime_type: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str | None] = mapped_column(String(64), index=True)

    # Object storage pointer -- the file body itself is never stored in SQL.
    object_key: Mapped[str | None] = mapped_column(String(512))
    storage_backend: Mapped[str | None] = mapped_column(String(32))

    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus, native_enum=False, length=32),
        default=DocumentStatus.UPLOADED,
        nullable=False,
        index=True,
    )
    status_detail: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)

    # Progress ledger consumed by the Knowledge Base UI's step list.
    processing_steps: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    figure_count: Mapped[int] = mapped_column(Integer, default=0)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    doc_metadata: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    outline: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    source_url: Mapped[str | None] = mapped_column(String(1024))

    indexed_at: Mapped[datetime | None] = mapped_column()
    processing_ms: Mapped[float] = mapped_column(Float, default=0.0)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", lazy="selectin"
    )
    entities: Mapped[list["DocumentEntity"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_documents_status_created", "status", "created_at"),)


class Chunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A retrievable unit of evidence with full source provenance."""

    __tablename__ = "chunks"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    modality: Mapped[ChunkModality] = mapped_column(
        SAEnum(ChunkModality, native_enum=False, length=16),
        default=ChunkModality.TEXT,
        nullable=False,
        index=True,
    )

    # --- Citation provenance ------------------------------------------------
    page_number: Mapped[int | None] = mapped_column(Integer, index=True)
    page_end: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(512))
    section_path: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    figure_label: Mapped[str | None] = mapped_column(String(64))
    table_label: Mapped[str | None] = mapped_column(String(64))
    bbox: Mapped[list[Any] | None] = mapped_column(JSONType)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)

    # Rendered artefact for figures/tables (object-storage key).
    asset_key: Mapped[str | None] = mapped_column(String(512))

    embedding_model: Mapped[str | None] = mapped_column(String(128))
    indexed_in_vector_store: Mapped[bool] = mapped_column(default=False)

    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_doc_ordinal", "document_id", "ordinal"),
        Index("ix_chunks_doc_page", "document_id", "page_number"),
    )

    # -- convenience -------------------------------------------------------
    @property
    def citation_label(self) -> str:
        parts = [self.document.filename if self.document else self.document_id]
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        if self.figure_label:
            parts.append(self.figure_label)
        if self.table_label:
            parts.append(self.table_label)
        return " · ".join(parts)


class DocumentEntity(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """An entity mention extracted from a document, mirrored into the graph store."""

    __tablename__ = "document_entities"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    normalized_name: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(64), default="CONCEPT", index=True)
    description: Mapped[str | None] = mapped_column(Text)
    salience: Mapped[float] = mapped_column(Float, default=0.0)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    chunk_ids: Mapped[list[Any]] = mapped_column(JSONType, default=list)

    document: Mapped[Document] = relationship(back_populates="entities")

    __table_args__ = (
        UniqueConstraint("document_id", "normalized_name", name="uq_document_entity"),
    )


class EntityRelation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A typed relation between two entities, extracted from a chunk."""

    __tablename__ = "entity_relations"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_name: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    target_name: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    relation_type: Mapped[str] = mapped_column(String(96), default="RELATED_TO", index=True)
    evidence_chunk_id: Mapped[str | None] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    context: Mapped[str | None] = mapped_column(Text)


__all__ = [
    "Document",
    "Chunk",
    "DocumentEntity",
    "EntityRelation",
    "DocumentStatus",
    "ChunkModality",
]
