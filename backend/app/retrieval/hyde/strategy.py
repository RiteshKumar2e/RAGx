"""HyDE -- Hypothetical Document Embeddings.

    query -> LLM writes a plausible answer passage -> embed that passage
          -> retrieve real evidence near it

The premise: a question and its answer are lexically and structurally different
objects, so a question embedding sits some distance from the passages that
answer it. A *hypothetical answer* lives in the same neighbourhood as the real
one, which makes it a better retrieval probe for conceptual queries.

HyDE costs one LLM call, so the router only selects it when the query is
semantically hard -- not for lookups where the question's own terms already
match the document's wording.

The hypothetical passage is never shown to the user and never enters the answer
context; it is discarded once retrieval completes.
"""

from __future__ import annotations

import asyncio

from app.core.cache import analysis_cache, cache_key
from app.core.config import get_settings
from app.core.logging import get_logger
from app.indexing.vector_store import get_vector_store
from app.llm.base import Message
from app.llm.embeddings import get_embedding_provider
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import HYDE_SYSTEM, HYDE_USER
from app.retrieval.base import (
    RetrievalConfig,
    RetrievalContext,
    RetrievalResult,
    RetrievalStrategy,
    StrategyName,
)
from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.loader import hydrate

log = get_logger("ragx.retrieval.hyde")


class HyDERAG(RetrievalStrategy):
    name = StrategyName.HYDE
    description = (
        "Generates a hypothetical answer passage, embeds it, and retrieves real evidence "
        "near it. Effective for conceptual queries whose wording differs from the source text."
    )
    uses_llm = True

    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.embedder = get_embedding_provider()
        self.gateway = get_gateway()

    async def generate_hypothesis(self, query: str, context: RetrievalContext) -> str:
        key = cache_key("hyde", query)
        cached = await analysis_cache.get(key)
        if cached is not None:
            return cached
        if not self.gateway.any_configured:
            return ""
        try:
            response = await self.gateway.complete(
                [Message.system(HYDE_SYSTEM), Message.user(HYDE_USER.format(question=query))],
                Purpose.HYDE,
                temperature=0.35,
                max_output_tokens=320,
                trace=context.trace,
            )
        except Exception as exc:
            log.warning("hyde.generation_failed", error=str(exc)[:160])
            return ""
        hypothesis = response.text.strip()
        await analysis_cache.set(key, hypothesis)
        return hypothesis

    async def retrieve(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult:
        settings = get_settings()
        pool = max(config.top_k * 3, min(config.candidate_pool, settings.candidate_pool_size))

        hypothesis = await self.generate_hypothesis(query, context)

        if not hypothesis:
            # Without an LLM, HyDE has nothing to add over dense retrieval; be
            # honest about that instead of pretending the strategy ran.
            vector = await self.embedder.embed_query(query)
            hits = await self.vector_store.search(
                vector=vector,
                limit=config.top_k,
                document_ids=config.document_ids,
                modalities=config.modalities,
            )
            chunks = await hydrate(context.session, [(h.chunk_id, h.score) for h in hits], self.name.value)
            result = RetrievalResult(
                chunks=chunks,
                strategy=self.name.value,
                effective_query=query,
                notes=["Hypothesis generation was unavailable; fell back to dense retrieval."],
                diagnostics={"hypothesis_generated": False},
            )
            result.rerank_positions()
            return result

        # Retrieve with both probes. The original query keeps HyDE honest when
        # the hypothesis drifts off-topic.
        hypothesis_vector, query_vector = await asyncio.gather(
            self.embedder.embed_query(hypothesis), self.embedder.embed_query(query)
        )
        hypothesis_hits, query_hits = await asyncio.gather(
            self.vector_store.search(
                vector=hypothesis_vector,
                limit=pool,
                document_ids=config.document_ids,
                modalities=config.modalities,
            ),
            self.vector_store.search(
                vector=query_vector,
                limit=max(config.top_k, pool // 2),
                document_ids=config.document_ids,
                modalities=config.modalities,
            ),
        )

        hypothesis_chunks = await hydrate(
            context.session, [(h.chunk_id, h.score) for h in hypothesis_hits], "hyde_hypothesis"
        )
        query_chunks = await hydrate(
            context.session, [(h.chunk_id, h.score) for h in query_hits], "hyde_original"
        )

        fused = reciprocal_rank_fusion(
            [("hyde_hypothesis", hypothesis_chunks), ("hyde_original", query_chunks)],
            weights={"hyde_hypothesis": 0.65, "hyde_original": 0.35},
        )
        fused = deduplicate(fused)
        for chunk in fused:
            chunk.add_source(self.name.value, chunk.score)

        result = RetrievalResult(
            chunks=fused[: config.top_k],
            strategy=self.name.value,
            effective_query=query,
            retrieval_calls=2,
            diagnostics={
                "hypothesis_generated": True,
                "hypothesis_chars": len(hypothesis),
                "hypothesis_preview": hypothesis[:280],
                "hypothesis_hits": len(hypothesis_hits),
                "original_query_hits": len(query_hits),
            },
            notes=["Retrieval was probed with a generated hypothetical passage as well as the original query."],
        )
        result.rerank_positions()
        return result


__all__ = ["HyDERAG"]
