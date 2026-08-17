"""Hybrid RAG -- dense + BM25 with rank fusion.

Dense retrieval generalises across paraphrase but is unreliable on rare exact
tokens: model names ("MobileNetV2"), dataset identifiers ("NEU-DET"), version
strings and API symbols often live in a sparse region of embedding space. BM25
handles exactly those, and fails on paraphrase. Running both and fusing with RRF
covers each one's blind spot.

The router raises ``sparse_weight`` when the query analysis reports a high
keyword requirement, and ``dense_weight`` when it reports a high semantic
requirement -- so "hybrid" is a tunable blend, not a fixed 50/50.
"""

from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.core.logging import get_logger
from app.indexing.bm25_index import get_bm25_index
from app.indexing.vector_store import get_vector_store
from app.llm.embeddings import get_embedding_provider
from app.retrieval.base import (
    RetrievalConfig,
    RetrievalContext,
    RetrievalResult,
    RetrievalStrategy,
    StrategyName,
)
from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.loader import hydrate

log = get_logger("ragx.retrieval.hybrid")


class HybridRAG(RetrievalStrategy):
    name = StrategyName.HYBRID
    description = "Dense vector search fused with BM25 keyword search via reciprocal rank fusion."
    uses_llm = False

    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.bm25 = get_bm25_index()
        self.embedder = get_embedding_provider()

    async def retrieve(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult:
        settings = get_settings()
        pool = max(config.top_k * 3, min(config.candidate_pool, settings.candidate_pool_size))

        async def _dense() -> list[tuple[str, float]]:
            vector = await self.embedder.embed_query(query)
            hits = await self.vector_store.search(
                vector=vector,
                limit=pool,
                document_ids=config.document_ids,
                modalities=config.modalities,
            )
            return [(h.chunk_id, h.score) for h in hits]

        async def _sparse() -> list[tuple[str, float]]:
            hits = await self.bm25.search(
                query=query,
                limit=pool,
                document_ids=config.document_ids,
                modalities=config.modalities,
            )
            return [(h.chunk_id, h.score) for h in hits]

        # Both retrievers run concurrently -- neither depends on the other.
        dense_scored, sparse_scored = await asyncio.gather(_dense(), _sparse())

        dense_chunks = await hydrate(context.session, dense_scored, "dense")
        sparse_chunks = await hydrate(context.session, sparse_scored, "bm25")

        fused = reciprocal_rank_fusion(
            [("dense", dense_chunks), ("bm25", sparse_chunks)],
            weights={"dense": config.dense_weight, "bm25": config.sparse_weight},
        )
        fused = deduplicate(fused)

        if config.min_score > 0:
            fused = [c for c in fused if c.score >= config.min_score]

        for chunk in fused:
            chunk.add_source(self.name.value, chunk.score)

        result = RetrievalResult(
            chunks=fused[: config.top_k],
            strategy=self.name.value,
            strategies_used=[self.name.value],
            effective_query=query,
            retrieval_calls=2,
            diagnostics={
                "dense_hits": len(dense_scored),
                "sparse_hits": len(sparse_scored),
                "dense_weight": config.dense_weight,
                "sparse_weight": config.sparse_weight,
                "fusion": "reciprocal_rank_fusion",
                "bm25_index_size": self.bm25.size,
                "overlap": len(
                    {c for c, _ in dense_scored} & {c for c, _ in sparse_scored}
                ),
            },
        )
        result.rerank_positions()
        return result


__all__ = ["HybridRAG"]
