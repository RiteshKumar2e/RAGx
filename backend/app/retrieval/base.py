"""The retrieval strategy contract.

Every strategy -- Naive, Hybrid, HyDE, Multimodal, Corrective, Graph, Adaptive,
Agentic -- implements the same three-argument interface:

    await strategy.retrieve(query, context, config) -> RetrievalResult

This is what lets the Adaptive Router treat strategies as interchangeable parts,
call one or several, and compose their outputs without special-casing any of
them. Composite strategies (Corrective, Adaptive, Agentic) are themselves
``RetrievalStrategy`` implementations that call other strategies.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from app.core.logging import TraceRecorder

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession


class StrategyName(str, Enum):
    NAIVE = "naive"
    HYBRID = "hybrid"
    HYDE = "hyde"
    MULTIMODAL = "multimodal"
    CORRECTIVE = "corrective"
    GRAPH = "graph"
    ADAPTIVE = "adaptive"
    AGENTIC = "agentic"

    @property
    def label(self) -> str:
        return {
            "naive": "Naive RAG",
            "hybrid": "Hybrid RAG",
            "hyde": "HyDE",
            "multimodal": "Multimodal RAG",
            "corrective": "Corrective RAG",
            "graph": "Graph RAG",
            "adaptive": "Adaptive RAG",
            "agentic": "Agentic RAG",
        }[self.value]


@dataclass
class RetrievedChunk:
    """A retrieved piece of evidence with full provenance.

    ``strategy_scores`` records what each contributing strategy scored this
    chunk, which is what the "Why this answer?" panel displays and what fusion
    uses to explain a ranking.
    """

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    score: float = 0.0
    modality: str = "text"
    page_number: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_path: list[str] = field(default_factory=list)
    figure_label: str | None = None
    table_label: str | None = None
    asset_key: str | None = None
    ordinal: int = 0
    token_count: int = 0
    rank: int = 0
    sources: list[str] = field(default_factory=list)
    strategy_scores: dict[str, float] = field(default_factory=dict)
    graph_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_source(self, strategy: str, score: float) -> None:
        if strategy not in self.sources:
            self.sources.append(strategy)
        self.strategy_scores[strategy] = round(max(score, self.strategy_scores.get(strategy, 0.0)), 6)

    @property
    def citation(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page": self.page_number,
            "page_end": self.page_end,
            "section": self.section,
            "section_path": self.section_path,
            "figure": self.figure_label,
            "table": self.table_label,
            "modality": self.modality,
            "asset_key": self.asset_key,
        }

    @property
    def location(self) -> str:
        parts: list[str] = []
        if self.page_number:
            parts.append(
                f"p.{self.page_number}"
                if not self.page_end or self.page_end == self.page_number
                else f"pp.{self.page_number}-{self.page_end}"
            )
        if self.section:
            parts.append(self.section)
        if self.figure_label:
            parts.append(self.figure_label)
        if self.table_label:
            parts.append(self.table_label)
        return " · ".join(parts)

    def to_dict(self, include_content: bool = True) -> dict[str, Any]:
        payload = {
            **self.citation,
            "score": round(self.score, 4),
            "rank": self.rank,
            "sources": self.sources,
            "strategy_scores": self.strategy_scores,
            "location": self.location,
            "graph_path": self.graph_path,
            "token_count": self.token_count,
        }
        if include_content:
            payload["content"] = self.content
        return payload


@dataclass
class RetrievalConfig:
    """Per-call retrieval knobs. Defaults come from application settings."""

    top_k: int = 8
    candidate_pool: int = 40
    document_ids: list[str] | None = None
    modalities: list[str] | None = None
    rerank: bool = True
    min_score: float = 0.0
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    graph_depth: int = 2
    include_neighbors: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def copy_with(self, **updates: Any) -> "RetrievalConfig":
        payload = {
            "top_k": self.top_k,
            "candidate_pool": self.candidate_pool,
            "document_ids": self.document_ids,
            "modalities": self.modalities,
            "rerank": self.rerank,
            "min_score": self.min_score,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "graph_depth": self.graph_depth,
            "include_neighbors": self.include_neighbors,
            "extra": dict(self.extra),
        }
        payload.update(updates)
        return RetrievalConfig(**payload)


@dataclass
class RetrievalContext:
    """Everything a strategy needs beyond the query string itself."""

    session: "AsyncSession"
    trace: TraceRecorder
    analysis: Any | None = None  # QueryAnalysis; kept loose to avoid a cycle
    history: list[dict[str, str]] = field(default_factory=list)
    document_titles: list[str] = field(default_factory=list)
    round_index: int = 0

    def child(self, round_index: int) -> "RetrievalContext":
        return RetrievalContext(
            session=self.session,
            trace=self.trace,
            analysis=self.analysis,
            history=self.history,
            document_titles=self.document_titles,
            round_index=round_index,
        )


@dataclass
class RetrievalResult:
    """The uniform return type of every strategy."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    strategy: str = "naive"
    strategies_used: list[str] = field(default_factory=list)
    effective_query: str = ""
    latency_ms: float = 0.0
    retrieval_calls: int = 1
    corrective_rounds: int = 0
    reranked: bool = False
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategies_used:
            self.strategies_used = [self.strategy]

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def scores(self) -> list[float]:
        return [c.score for c in self.chunks]

    @property
    def top_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def document_ids(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.document_id not in seen:
                seen.append(chunk.document_id)
        return seen

    def rerank_positions(self) -> None:
        self.chunks.sort(key=lambda c: c.score, reverse=True)
        for index, chunk in enumerate(self.chunks, start=1):
            chunk.rank = index

    def truncate(self, limit: int) -> "RetrievalResult":
        self.rerank_positions()
        self.chunks = self.chunks[:limit]
        return self

    def to_dict(self, include_content: bool = False) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "strategies_used": self.strategies_used,
            "effective_query": self.effective_query,
            "latency_ms": round(self.latency_ms, 2),
            "retrieval_calls": self.retrieval_calls,
            "corrective_rounds": self.corrective_rounds,
            "reranked": self.reranked,
            "chunk_count": len(self.chunks),
            "top_score": round(self.top_score, 4),
            "mean_score": round(self.mean_score, 4),
            "notes": self.notes,
            "diagnostics": self.diagnostics,
            "chunks": [c.to_dict(include_content) for c in self.chunks],
        }


class RetrievalStrategy(ABC):
    """Base class for every retrieval strategy."""

    name: StrategyName = StrategyName.NAIVE
    description: str = ""
    #: Whether this strategy calls an LLM (used for cost-aware routing).
    uses_llm: bool = False

    @abstractmethod
    async def retrieve(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult: ...

    # -- shared instrumentation -------------------------------------------
    async def run(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult:
        """Execute with timing and trace recording. Callers should use this."""
        started = time.perf_counter()
        result = await self.retrieve(query, context, config)
        result.latency_ms = result.latency_ms or (time.perf_counter() - started) * 1000
        result.rerank_positions()
        context.trace.record_retrieval(
            strategy=self.name.value,
            duration_ms=result.latency_ms,
            chunk_ids=result.chunk_ids,
            scores=result.scores,
            effective_query=result.effective_query[:200] or query[:200],
            round_index=context.round_index,
        )
        return result

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "label": self.name.label,
            "description": self.description,
            "uses_llm": self.uses_llm,
        }


__all__ = [
    "StrategyName",
    "RetrievedChunk",
    "RetrievalConfig",
    "RetrievalContext",
    "RetrievalResult",
    "RetrievalStrategy",
]
