"""Evaluation experiment ORM models.

An :class:`EvaluationRun` is one strategy evaluated over one benchmark subset.
:class:`EvaluationResult` holds the per-question record. Every number displayed
in the Evaluation dashboard comes from these tables -- if no run exists, the UI
shows an empty state rather than invented figures.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin


class EvaluationRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "evaluation_runs"

    name: Mapped[str] = mapped_column(String(200), default="benchmark run")
    dataset: Mapped[str] = mapped_column(String(120), default="ragx_benchmark")
    dataset_version: Mapped[str | None] = mapped_column(String(32))
    strategy: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(24), default="running", index=True)
    notes: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    question_count: Mapped[int] = mapped_column(Integer, default=0)
    completed_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)

    # --- Aggregated metrics (means over completed questions) ---------------
    recall_at_k: Mapped[float | None] = mapped_column(Float)
    precision_at_k: Mapped[float | None] = mapped_column(Float)
    mrr: Mapped[float | None] = mapped_column(Float)
    ndcg_at_k: Mapped[float | None] = mapped_column(Float)
    context_relevance: Mapped[float | None] = mapped_column(Float)

    answer_relevance: Mapped[float | None] = mapped_column(Float)
    faithfulness: Mapped[float | None] = mapped_column(Float)
    groundedness: Mapped[float | None] = mapped_column(Float)
    citation_accuracy: Mapped[float | None] = mapped_column(Float)

    avg_latency_ms: Mapped[float | None] = mapped_column(Float)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float)
    avg_total_tokens: Mapped[float | None] = mapped_column(Float)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    avg_retrieval_calls: Mapped[float | None] = mapped_column(Float)
    corrective_retrievals: Mapped[int] = mapped_column(Integer, default=0)
    abstention_rate: Mapped[float | None] = mapped_column(Float)

    k: Mapped[int] = mapped_column(Integer, default=8)
    error_message: Mapped[str | None] = mapped_column(Text)

    results: Mapped[list["EvaluationResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationResult(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "evaluation_results"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(48), index=True)
    answer: Mapped[str | None] = mapped_column(Text)

    strategies_used: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    retrieved_chunk_ids: Mapped[list[Any]] = mapped_column(JSONType, default=list)
    relevant_chunk_ids: Mapped[list[Any]] = mapped_column(JSONType, default=list)

    recall_at_k: Mapped[float | None] = mapped_column(Float)
    precision_at_k: Mapped[float | None] = mapped_column(Float)
    reciprocal_rank: Mapped[float | None] = mapped_column(Float)
    ndcg_at_k: Mapped[float | None] = mapped_column(Float)
    context_relevance: Mapped[float | None] = mapped_column(Float)

    answer_relevance: Mapped[float | None] = mapped_column(Float)
    faithfulness: Mapped[float | None] = mapped_column(Float)
    groundedness: Mapped[float | None] = mapped_column(Float)
    citation_accuracy: Mapped[float | None] = mapped_column(Float)

    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_calls: Mapped[int] = mapped_column(Integer, default=0)
    corrective_rounds: Mapped[int] = mapped_column(Integer, default=0)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)

    judge_detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)

    run: Mapped[EvaluationRun] = relationship(back_populates="results")


__all__ = ["EvaluationRun", "EvaluationResult"]
