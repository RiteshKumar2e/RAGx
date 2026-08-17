"""Evaluation and analytics schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel

EVAL_STRATEGIES = Literal[
    "naive", "hybrid", "hyde", "multimodal", "corrective", "graph", "adaptive", "agentic", "ragx"
]


class BenchmarkQuestion(BaseModel):
    id: str
    question: str
    category: str
    difficulty: str = "moderate"
    expects_abstention: bool = False
    notes: str = ""
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None


class BenchmarkInfo(BaseModel):
    name: str
    version: str
    description: str
    question_count: int
    categories: dict[str, int] = Field(default_factory=dict)
    has_relevance_labels: bool = False
    questions: list[BenchmarkQuestion] = Field(default_factory=list)


class EvaluationRunRequest(BaseModel):
    strategies: list[EVAL_STRATEGIES] = Field(
        default_factory=lambda: ["naive", "hybrid", "adaptive"],
        description="Strategies to benchmark. 'ragx' is the full adaptive + verification pipeline.",
    )
    dataset: str = "ragx_benchmark"
    categories: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=200)
    k: int = Field(default=8, ge=1, le=30)
    judge_generation: bool = Field(
        default=True,
        description="Run LLM judges for faithfulness / answer relevance / context relevance. "
        "Requires a configured provider; retrieval metrics are computed either way.",
    )
    name: str | None = None
    notes: str | None = None


class EvaluationResultItem(ORMModel):
    id: str
    question_id: str
    question: str
    category: str | None = None
    answer: str | None = None
    strategies_used: list[str] = Field(default_factory=list)
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    reciprocal_rank: float | None = None
    ndcg_at_k: float | None = None
    context_relevance: float | None = None
    answer_relevance: float | None = None
    faithfulness: float | None = None
    groundedness: float | None = None
    citation_accuracy: float | None = None
    latency_ms: float = 0.0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    retrieval_calls: int = 0
    corrective_rounds: int = 0
    abstained: bool = False
    error_message: str | None = None


class EvaluationRunSummary(ORMModel):
    id: str
    name: str
    dataset: str
    strategy: str
    status: str
    question_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    k: int = 8
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    context_relevance: float | None = None
    answer_relevance: float | None = None
    faithfulness: float | None = None
    groundedness: float | None = None
    citation_accuracy: float | None = None
    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    avg_total_tokens: float | None = None
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    avg_retrieval_calls: float | None = None
    corrective_retrievals: int = 0
    abstention_rate: float | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class EvaluationRunDetail(EvaluationRunSummary):
    config: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    results: list[EvaluationResultItem] = Field(default_factory=list)


class EvaluationComparison(BaseModel):
    """Side-by-side comparison of the latest completed run per strategy."""

    runs: list[EvaluationRunSummary] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    best_by_metric: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime
    has_data: bool = False
    message: str = ""


class EvaluationStartResponse(BaseModel):
    run_ids: list[str]
    strategies: list[str]
    question_count: int
    message: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- analytics
class TimeseriesPoint(BaseModel):
    date: str
    value: float = 0.0
    count: int = 0


class DistributionItem(BaseModel):
    label: str
    value: int = 0
    percentage: float = 0.0


class DashboardStats(BaseModel):
    total_documents: int = 0
    indexed_documents: int = 0
    processing_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    total_queries: int = 0
    queries_last_7_days: int = 0
    avg_retrieval_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    most_used_strategy: str | None = None
    corrective_rate: float = 0.0
    abstention_rate: float = 0.0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    evaluation_runs: int = 0


class AnalyticsResponse(BaseModel):
    stats: DashboardStats
    queries_over_time: list[TimeseriesPoint] = Field(default_factory=list)
    latency_over_time: list[TimeseriesPoint] = Field(default_factory=list)
    strategy_usage: list[DistributionItem] = Field(default_factory=list)
    complexity_distribution: list[DistributionItem] = Field(default_factory=list)
    intent_distribution: list[DistributionItem] = Field(default_factory=list)
    confidence_distribution: list[DistributionItem] = Field(default_factory=list)
    provider_usage: list[DistributionItem] = Field(default_factory=list)
    recent_queries: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "BenchmarkQuestion",
    "BenchmarkInfo",
    "EvaluationRunRequest",
    "EvaluationResultItem",
    "EvaluationRunSummary",
    "EvaluationRunDetail",
    "EvaluationComparison",
    "EvaluationStartResponse",
    "TimeseriesPoint",
    "DistributionItem",
    "DashboardStats",
    "AnalyticsResponse",
]
