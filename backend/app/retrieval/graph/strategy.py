"""Graph RAG -- entity-and-relationship retrieval.

Vector search answers "what does the corpus say about X". It cannot answer
"how does X relate to Y" when no single passage states the relation, because
the connection only exists across passages. Graph RAG recovers those by:

1. Resolving the query's entities against the knowledge graph (LLM-extracted
   entities from ingestion, plus surface-form matching).
2. Traversing outward to bounded depth, or finding the path between two named
   entities for an explicit relationship question.
3. Returning the chunks each traversed edge was extracted from -- so a graph hit
   is still ordinary, citable evidence, not an unverifiable assertion.

Because the graph is sparser than the vector index, dense results are blended in
so a graph miss degrades to normal retrieval rather than an empty answer.
"""

from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.core.text import technical_terms
from app.indexing.graph_store import GraphPath, get_graph_store
from app.indexing.vector_store import get_vector_store
from app.llm.embeddings import get_embedding_provider
from app.retrieval.base import (
    RetrievalConfig,
    RetrievalContext,
    RetrievalResult,
    RetrievalStrategy,
    RetrievedChunk,
    StrategyName,
)
from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.loader import hydrate

log = get_logger("ragx.retrieval.graph")


class GraphRAG(RetrievalStrategy):
    name = StrategyName.GRAPH
    description = (
        "Resolves query entities in the knowledge graph, traverses typed relations, and "
        "returns the source chunks behind each edge. Handles relationship and multi-hop queries."
    )
    uses_llm = False

    def __init__(self) -> None:
        self.graph = get_graph_store()
        self.vector_store = get_vector_store()
        self.embedder = get_embedding_provider()

    # ------------------------------------------------------------- entities
    async def resolve_entities(self, query: str, context: RetrievalContext) -> list[dict]:
        """Find graph entities named by the query.

        Candidates come from the query analysis when available (the LLM already
        identified them) and from surface-form technical terms otherwise. Each
        candidate is matched against the graph; only real nodes survive.
        """
        candidates: list[str] = []
        analysis = context.analysis
        if analysis is not None:
            candidates.extend(getattr(analysis, "entities", []) or [])
            candidates.extend(getattr(analysis, "key_terms", []) or [])
        candidates.extend(technical_terms(query))

        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            key = candidate.lower().strip()
            if key and key not in seen and len(key) > 1:
                seen.add(key)
                ordered.append(candidate)

        if not ordered:
            # No named entity in the query -- fall back to a free-text lookup.
            return await self.graph.search_entities(query, limit=6)

        results = await asyncio.gather(
            *(self.graph.search_entities(term, limit=3) for term in ordered[:8]),
            return_exceptions=True,
        )
        resolved: dict[str, dict] = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            for entity in item:
                if entity["id"] not in resolved:
                    resolved[entity["id"]] = entity
        return sorted(resolved.values(), key=lambda e: -e.get("degree", 0))[:8]

    # ------------------------------------------------------------- traversal
    async def _traverse(
        self, entities: list[dict], query: str, config: RetrievalContext | RetrievalConfig, depth: int
    ) -> list[GraphPath]:
        names = [e["name"] for e in entities]
        if len(names) >= 2:
            # An explicit two-entity question: find how they actually connect.
            pair_results = await asyncio.gather(
                *(
                    self.graph.paths_between(names[0], other, max_depth=max(depth, 3))
                    for other in names[1:4]
                ),
                return_exceptions=True,
            )
            paths: list[GraphPath] = []
            for item in pair_results:
                if isinstance(item, list):
                    paths.extend(item)
            if paths:
                neighborhood = await self.graph.neighborhood(names, depth=depth, limit=40)
                return (paths + neighborhood)[:80]
        return await self.graph.neighborhood(names, depth=depth, limit=60)

    async def retrieve(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult:
        entities = await self.resolve_entities(query, context)

        if not entities:
            return await self._fallback(
                query,
                context,
                config,
                note="No entities from this query were found in the knowledge graph; "
                     "dense retrieval was used instead.",
                diagnostics={"entities_resolved": 0, "graph_backend": self.graph.backend},
            )

        depth = max(1, min(config.graph_depth, 4))
        paths = await self._traverse(entities, query, config, depth)

        # Score each evidence chunk by the best path that surfaced it.
        chunk_scores: dict[str, float] = {}
        chunk_paths: dict[str, str] = {}
        for path in paths:
            description = path.describe()
            for chunk_id in path.chunk_ids:
                if path.score > chunk_scores.get(chunk_id, 0.0):
                    chunk_scores[chunk_id] = path.score
                    chunk_paths[chunk_id] = description

        # Entities also carry the chunks they were first mentioned in.
        for entity in entities:
            for chunk_id in (entity.get("chunk_ids") or [])[:6]:
                chunk_scores.setdefault(chunk_id, 0.55)
                chunk_paths.setdefault(chunk_id, f"mention: {entity['name']}")

        if not chunk_scores:
            return await self._fallback(
                query,
                context,
                config,
                note=(
                    f"Matched {len(entities)} graph entities but no relation carried citable "
                    "source text; dense retrieval was used instead."
                ),
                diagnostics={
                    "entities_resolved": len(entities),
                    "entities": [e["name"] for e in entities],
                    "paths_found": len(paths),
                    "graph_backend": self.graph.backend,
                },
            )

        graph_chunks = await hydrate(
            context.session,
            sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True)[: config.candidate_pool],
            "graph",
        )
        for chunk in graph_chunks:
            chunk.graph_path = chunk_paths.get(chunk.chunk_id)
            chunk.metadata["graph_evidence"] = True

        # Blend in dense results so the answer is not limited to what the
        # extractor happened to turn into edges.
        dense_chunks = await self._dense(query, context, config)

        fused = reciprocal_rank_fusion(
            [("graph", graph_chunks), ("dense", dense_chunks)],
            weights={"graph": 1.0, "dense": 0.5},
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
                "graph_backend": self.graph.backend,
                "entities_resolved": len(entities),
                "entities": [e["name"] for e in entities],
                "paths_found": len(paths),
                "traversal_depth": depth,
                "graph_chunks": len(graph_chunks),
                "top_paths": [p.describe() for p in paths[:6]],
                "multi_hop_paths": sum(1 for p in paths if len(p.relations) > 1),
            },
        )
        result.rerank_positions()
        return result

    # -------------------------------------------------------------- helpers
    async def _dense(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> list[RetrievedChunk]:
        vector = await self.embedder.embed_query(query)
        hits = await self.vector_store.search(
            vector=vector,
            limit=max(config.top_k, config.candidate_pool // 2),
            document_ids=config.document_ids,
            modalities=config.modalities,
        )
        return await hydrate(context.session, [(h.chunk_id, h.score) for h in hits], "dense")

    async def _fallback(
        self,
        query: str,
        context: RetrievalContext,
        config: RetrievalConfig,
        note: str,
        diagnostics: dict,
    ) -> RetrievalResult:
        chunks = await self._dense(query, context, config)
        for chunk in chunks:
            chunk.add_source(self.name.value, chunk.score)
        result = RetrievalResult(
            chunks=chunks[: config.top_k],
            strategy=self.name.value,
            effective_query=query,
            notes=[note],
            diagnostics={**diagnostics, "fell_back_to_dense": True},
        )
        result.rerank_positions()
        return result


__all__ = ["GraphRAG"]
