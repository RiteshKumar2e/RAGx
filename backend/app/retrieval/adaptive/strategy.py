"""Adaptive RAG -- analyse, route, execute, fuse.

    query -> QueryAnalyzer -> AdaptiveRouter -> selected strategies
          -> parallel execution -> fusion -> rerank -> evidence

This strategy is the system's default entry point. It is deliberately thin: all
the intelligence lives in the analyzer and the router, and all the retrieval
work lives in the strategies it delegates to. Its own job is orchestration --
running the selected strategies concurrently, fusing their outputs while
preserving per-strategy provenance, and carrying the routing explanation
through to the response.
"""

from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.core.logging import get_logger
from app.indexing.graph_store import get_graph_store
from app.retrieval.adaptive.analyzer import QueryAnalysis, QueryAnalyzer
from app.retrieval.adaptive.router import AdaptiveRouter, RoutingDecision
from app.retrieval.base import (
    RetrievalConfig,
    RetrievalContext,
    RetrievalResult,
    RetrievalStrategy,
    StrategyName,
)
from app.retrieval.fusion import deduplicate, enforce_document_diversity, reciprocal_rank_fusion
from app.retrieval.registry import get_strategy, wrap_corrective
from app.retrieval.rerank import rerank

log = get_logger("ragx.retrieval.adaptive")


class AdaptiveRAG(RetrievalStrategy):
    name = StrategyName.ADAPTIVE
    description = (
        "Analyses the query, selects the minimum set of retrieval strategies that can answer "
        "it, runs them in parallel and fuses the results."
    )
    uses_llm = True

    def __init__(self) -> None:
        self.analyzer = QueryAnalyzer()
        self.router = AdaptiveRouter()

    # ------------------------------------------------------------- planning
    async def plan(
        self,
        query: str,
        context: RetrievalContext,
        *,
        document_count: int = 0,
        forced_strategies: list[str] | None = None,
        overrides: dict | None = None,
        analysis: QueryAnalysis | None = None,
    ) -> tuple[QueryAnalysis, RoutingDecision]:
        """Produce the analysis and routing decision without retrieving.

        Exposed separately so ``/api/v1/query/analyze`` can show users what the
        router *would* do, at the cost of a single analysis call.
        """
        with context.trace.span("query_analysis", category="analysis"):
            if analysis is None:
                analysis = await self.analyzer.analyze(
                    query,
                    history=context.history,
                    document_titles=context.document_titles,
                    trace=context.trace,
                )
        context.analysis = analysis

        graph = get_graph_store()
        graph_available = True
        try:
            stats = await graph.stats()
            graph_available = stats.get("entities", 0) > 0
        except Exception:
            graph_available = False

        decision = self.router.route(
            analysis,
            document_count=document_count,
            graph_available=graph_available,
            multimodal_available=True,
            forced_strategies=forced_strategies,
            overrides=overrides,
        )
        if not graph_available and StrategyName.GRAPH in decision.strategies:
            decision.rules_fired.append(
                {
                    "rule": "graph_unavailable",
                    "reason": "The knowledge graph holds no entities yet, so graph hits will fall back to dense retrieval.",
                }
            )
        return analysis, decision

    # ------------------------------------------------------------- execution
    async def execute(
        self, query: str, context: RetrievalContext, decision: RoutingDecision
    ) -> RetrievalResult:
        """Run a routing decision. Used by the query service after planning."""
        settings = get_settings()
        config = decision.config

        if decision.use_agentic:
            agentic = get_strategy(StrategyName.AGENTIC)
            result = await agentic.run(query, context, config)
            result.strategies_used = list(
                dict.fromkeys(decision.strategy_values + result.strategies_used)
            )
            result.diagnostics["routing"] = decision.as_dict()
            return result

        selected = [decision.primary, *decision.parallel]

        async def _run(name: StrategyName) -> RetrievalResult | Exception:
            strategy: RetrievalStrategy = get_strategy(name)
            if decision.use_corrective and name is decision.primary:
                strategy = wrap_corrective(strategy)
            try:
                return await strategy.run(query, context, config)
            except Exception as exc:  # a single strategy failing must not sink the query
                log.warning("adaptive.strategy_failed", strategy=name.value, error=str(exc)[:200])
                return exc

        # Selected strategies are independent, so they run concurrently.
        outcomes = await asyncio.gather(*(_run(name) for name in selected))

        results: list[tuple[StrategyName, RetrievalResult]] = []
        failures: list[str] = []
        for name, outcome in zip(selected, outcomes):
            if isinstance(outcome, RetrievalResult):
                results.append((name, outcome))
            else:
                failures.append(f"{name.label} failed: {str(outcome)[:120]}")

        if not results:
            return RetrievalResult(
                chunks=[],
                strategy=self.name.value,
                strategies_used=decision.strategy_values,
                effective_query=query,
                notes=failures or ["No retrieval strategy returned results."],
                diagnostics={"routing": decision.as_dict(), "failures": failures},
            )

        # -- fuse ------------------------------------------------------------
        if len(results) == 1:
            merged = results[0][1]
            chunks = merged.chunks
            fusion_method = "single_strategy"
        else:
            ranked_lists = [(name.value, result.chunks) for name, result in results]
            weights = {name.value: (1.0 if name is decision.primary else 0.7) for name, _ in results}
            chunks = deduplicate(reciprocal_rank_fusion(ranked_lists, weights=weights))
            fusion_method = "reciprocal_rank_fusion"

        max_per_document = config.extra.get("max_per_document")
        if max_per_document:
            chunks = enforce_document_diversity(chunks, int(max_per_document))

        # -- rerank ----------------------------------------------------------
        rerank_diagnostics: dict = {"applied": False}
        if config.rerank and chunks:
            chunks, rerank_diagnostics = await rerank(
                query,
                chunks,
                use_llm=bool(context.analysis and context.analysis.requires_verification),
                prefer_modalities=config.modalities,
                top_n=settings.rerank_top_n,
                trace=context.trace,
            )
            rerank_diagnostics["applied"] = True

        combined = RetrievalResult(
            chunks=chunks[: config.top_k],
            strategy=self.name.value,
            strategies_used=list(
                dict.fromkeys(
                    [s for _, r in results for s in r.strategies_used] + decision.strategy_values
                )
            ),
            effective_query=query,
            retrieval_calls=sum(r.retrieval_calls for _, r in results),
            corrective_rounds=max((r.corrective_rounds for _, r in results), default=0),
            reranked=rerank_diagnostics.get("applied", False),
            notes=[note for _, r in results for note in r.notes] + failures,
            diagnostics={
                "routing": decision.as_dict(),
                "fusion": fusion_method,
                "rerank": rerank_diagnostics,
                "per_strategy": {
                    name.value: {
                        "chunks": len(result.chunks),
                        "top_score": round(result.top_score, 4),
                        "latency_ms": round(result.latency_ms, 2),
                        **{
                            k: v
                            for k, v in result.diagnostics.items()
                            if k not in {"routing", "per_strategy"}
                        },
                    }
                    for name, result in results
                },
                "failures": failures,
            },
        )
        combined.rerank_positions()
        return combined

    # ------------------------------------------------------------- interface
    async def retrieve(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult:
        """Full adaptive pass. Honours ``config`` as routing overrides."""
        analysis, decision = await self.plan(
            query,
            context,
            document_count=len(context.document_titles),
            overrides={
                "top_k": config.top_k,
                "document_ids": config.document_ids,
            },
        )
        return await self.execute(query, context, decision)


__all__ = ["AdaptiveRAG"]
