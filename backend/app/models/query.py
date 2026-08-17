"""Query history, retrieval logs and per-answer verification records."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin


class Conversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "conversations"

    title: Mapped[str] = mapped_column(String(320), default="New research thread")
    user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), index=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)

    queries: Mapped[list["QueryRecord"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="QueryRecord.created_at"
    )


class QueryRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One end-to-end RAGX execution: analysis, routing, retrieval, answer."""

    __tablename__ = "queries"

    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id"), index=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)

    # --- Adaptive router decision ------------------------------------------
    analysis: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    intent: Mapped[str | None] = mapped_column(String(48), index=True)
    complexity: Mapped[str | None] = mapped_column(String(24), index=True)
    strategies: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    routing_reason: Mapped[str | None] = mapped_column(Text)
    routing_mode: Mapped[str | None] = mapped_column(String(32))

    # --- Retrieval ----------------------------------------------------------
    retrieved_chunk_ids: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    retrieval_scores: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, default=0)
    documents_used: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    retrieval_calls: Mapped[int] = mapped_column(Integer, default=0)
    corrective_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    corrective_rounds: Mapped[int] = mapped_column(Integer, default=0)
    agentic_used: Mapped[bool] = mapped_column(Boolean, default=False)
    reranked: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- Generation ---------------------------------------------------------
    llm_provider: Mapped[str | None] = mapped_column(String(32), index=True)
    llm_model: Mapped[str | None] = mapped_column(String(96))
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # --- Verification -------------------------------------------------------
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_label: Mapped[str | None] = mapped_column(String(16))
    claims_total: Mapped[int] = mapped_column(Integer, default=0)
    claims_supported: Mapped[int] = mapped_column(Integer, default=0)
    citation_coverage: Mapped[float] = mapped_column(Float, default=0.0)
    insufficient_evidence: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # --- Timing / observability --------------------------------------------
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    generation_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    trace: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    citations: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    evidence: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    verification: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    status: Mapped[str] = mapped_column(String(24), default="completed", index=True)
    error_message: Mapped[str | None] = mapped_column(Text)

    conversation: Mapped[Conversation | None] = relationship(back_populates="queries")
    retrieval_logs: Mapped[list["RetrievalLog"]] = relationship(
        back_populates="query", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_queries_created_status", "created_at", "status"),)


class RetrievalLog(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One strategy invocation inside a query -- the unit of retrieval analytics."""

    __tablename__ = "retrieval_logs"

    query_id: Mapped[str] = mapped_column(
        ForeignKey("queries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    strategy: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    round_index: Mapped[int] = mapped_column(Integer, default=0)
    effective_query: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    results_returned: Mapped[int] = mapped_column(Integer, default=0)
    top_score: Mapped[float] = mapped_column(Float, default=0.0)
    mean_score: Mapped[float] = mapped_column(Float, default=0.0)
    chunk_ids: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    scores: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    was_corrective: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    query: Mapped[QueryRecord] = relationship(back_populates="retrieval_logs")


__all__ = ["Conversation", "QueryRecord", "RetrievalLog"]
