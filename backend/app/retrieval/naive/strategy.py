"""Naive RAG -- the baseline.

    query -> embedding -> Qdrant top-K -> chunks

No rewriting, no fusion, no grading. It is deliberately the simplest possible
pipeline: it is both the fast path for genuinely simple questions and the
control condition every other strategy is measured against in the evaluation
suite.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.indexing.vector_store import get_vector_store
from app.llm.embeddings import get_embedding_provider
from app.retrieval.base import (
    RetrievalConfig,
    RetrievalContext,
    RetrievalResult,
    RetrievalStrategy,
    StrategyName,
)
from app.retrieval.loader import hydrate

log = get_logger("ragx.retrieval.naive")


class NaiveRAG(RetrievalStrategy):
    name = StrategyName.NAIVE
    description = "Single-shot dense vector search over chunk embeddings."
    uses_llm = False

    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.embedder = get_embedding_provider()

    async def retrieve(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult:
        settings = get_settings()
        vector = await self.embedder.embed_query(query)

        hits = await self.vector_store.search(
            vector=vector,
            limit=max(config.top_k, min(config.candidate_pool, settings.candidate_pool_size)),
            document_ids=config.document_ids,
            modalities=config.modalities,
        )
        scored = [(hit.chunk_id, hit.score) for hit in hits]
        chunks = await hydrate(context.session, scored, self.name.value)

        if config.min_score > 0:
            chunks = [c for c in chunks if c.score >= config.min_score]

        result = RetrievalResult(
            chunks=chunks[: config.top_k],
            strategy=self.name.value,
            effective_query=query,
            diagnostics={
                "vector_hits": len(hits),
                "embedding_provider": self.embedder.name,
                "embedding_dimension": self.embedder.dimension,
            },
        )
        result.rerank_positions()
        return result


__all__ = ["NaiveRAG"]
