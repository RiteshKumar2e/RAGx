"""Corrective RAG -- retrieve, evaluate, repair.

    retrieve -> grade -> if poor: diagnose -> rewrite -> re-retrieve -> merge
             -> re-grade (bounded rounds)

Corrective RAG is a *wrapper*: it takes any other strategy as its inner
retriever, which is why the Adaptive Router can compose "Hybrid + Corrective" or
"Graph + Corrective" without new code.

The repair action is chosen from the diagnosis rather than applied blindly:

* ``off_topic`` / ``empty``   -> rewrite the query and retrieve again
* ``too_general``            -> rewrite *and* widen the candidate pool
* ``partially_relevant``     -> keep the results, widen the pool and rerank
* ``wrong_document``         -> drop the document filter and retrieve again

If every round still fails, the result is returned with
``insufficient_evidence`` set, and the generator abstains instead of answering
from weak context. Failing loudly is the point of this strategy.
"""

from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.base import (
    RetrievalConfig,
    RetrievalContext,
    RetrievalResult,
    RetrievalStrategy,
    StrategyName,
)
from app.retrieval.corrective.grading import Diagnosis, GradingResult, grade_retrieval
from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.query_rewrite import llm_rewrites
from app.retrieval.rerank import rerank

log = get_logger("ragx.retrieval.corrective")


class CorrectiveRAG(RetrievalStrategy):
    name = StrategyName.CORRECTIVE
    description = (
        "Grades retrieval quality and, when it is poor, diagnoses the failure, rewrites the "
        "query and retrieves again -- rather than passing weak context to the model."
    )
    uses_llm = True

    def __init__(self, base_strategy: RetrievalStrategy):
        self.base = base_strategy

    async def retrieve(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult:
        settings = get_settings()
        floor = settings.corrective_relevance_floor
        max_rounds = max(0, settings.corrective_max_rounds)

        result = await self.base.run(query, context, config)
        grading = await grade_retrieval(query, result.chunks, floor, trace=context.trace)

        rounds = 0
        history: list[dict] = [
            {"round": 0, "query": query, "grade": grading.as_dict(), "results": len(result.chunks)}
        ]
        total_calls = result.retrieval_calls
        applied_actions: list[str] = []

        while grading.is_poor and rounds < max_rounds:
            rounds += 1
            action, next_config = self._plan_repair(grading.diagnosis, config)
            applied_actions.append(action)

            context.trace.record_corrective(
                round_index=rounds,
                reason=grading.diagnosis.value,
                action=action,
                overall_score=grading.overall,
                floor=floor,
                hint=grading.diagnosis.hint,
            )

            rewrite = await llm_rewrites(
                query, result.chunks, grading.diagnosis.hint, trace=context.trace
            )
            if not rewrite.rewrites:
                result.notes.append("Retrieval quality was poor but no useful rewrite could be generated.")
                break

            child = context.child(round_index=rounds)
            retries = await asyncio.gather(
                *(self.base.run(r, child, next_config) for r in rewrite.rewrites[:3]),
                return_exceptions=True,
            )

            ranked_lists = [("original", result.chunks)]
            for index, retry in enumerate(retries):
                if isinstance(retry, RetrievalResult):
                    total_calls += retry.retrieval_calls
                    ranked_lists.append((f"rewrite_{index + 1}", retry.chunks))
                elif isinstance(retry, Exception):
                    log.warning("corrective.retry_failed", error=str(retry)[:160])

            if len(ranked_lists) == 1:
                break

            weights = {"original": 0.7}
            weights.update({f"rewrite_{i + 1}": 1.0 for i in range(len(ranked_lists) - 1)})
            merged = deduplicate(reciprocal_rank_fusion(ranked_lists, weights=weights))

            result.chunks = merged[: max(config.top_k, config.top_k)]
            result.rerank_positions()

            grading = await grade_retrieval(query, result.chunks, floor, trace=context.trace)
            history.append(
                {
                    "round": rounds,
                    "action": action,
                    "rewrites": rewrite.rewrites[:3],
                    "rewrite_strategy": rewrite.strategy,
                    "grade": grading.as_dict(),
                    "results": len(result.chunks),
                }
            )

            if not grading.is_poor:
                result.notes.append(
                    f"Corrective retrieval round {rounds} recovered usable evidence "
                    f"(quality {grading.overall:.2f} ≥ {floor:.2f})."
                )
                break

        # A final rerank pass -- corrective rounds change the candidate mix.
        if config.rerank and result.chunks:
            result.chunks, rerank_diagnostics = await rerank(
                query,
                result.chunks,
                use_llm=rounds > 0,
                prefer_modalities=config.modalities,
                top_n=settings.rerank_top_n,
                trace=context.trace,
            )
            result.reranked = True
            result.diagnostics["rerank"] = rerank_diagnostics

        result.chunks = result.chunks[: config.top_k]
        result.rerank_positions()

        insufficient = grading.is_poor and grading.overall < settings.insufficient_evidence_threshold
        if insufficient:
            result.notes.append(
                "Retrieval quality remained below the confidence floor after correction; "
                "the answer will state that evidence is insufficient."
            )

        result.strategy = self.name.value
        if self.base.name.value not in result.strategies_used:
            result.strategies_used.insert(0, self.base.name.value)
        if self.name.value not in result.strategies_used:
            result.strategies_used.append(self.name.value)
        result.corrective_rounds = rounds
        result.retrieval_calls = total_calls
        result.diagnostics.update(
            {
                "corrective": {
                    "rounds": rounds,
                    "max_rounds": max_rounds,
                    "triggered": rounds > 0,
                    "actions": applied_actions,
                    "floor": floor,
                    "final_grade": grading.as_dict(),
                    "history": history,
                    "insufficient_evidence": insufficient,
                },
                "base_strategy": self.base.name.value,
            }
        )
        return result

    # ---------------------------------------------------------------- repair
    @staticmethod
    def _plan_repair(diagnosis: Diagnosis, config: RetrievalConfig) -> tuple[str, RetrievalConfig]:
        """Map a failure diagnosis to a concrete repair action."""
        if diagnosis is Diagnosis.EMPTY:
            return (
                "rewrite_and_broaden",
                config.copy_with(
                    candidate_pool=config.candidate_pool * 2,
                    modalities=None,
                    document_ids=None,
                    min_score=0.0,
                ),
            )
        if diagnosis is Diagnosis.OFF_TOPIC:
            return "rewrite_query", config.copy_with(min_score=0.0)
        if diagnosis is Diagnosis.TOO_GENERAL:
            return (
                "rewrite_and_widen_pool",
                config.copy_with(
                    candidate_pool=int(config.candidate_pool * 1.75),
                    top_k=config.top_k + 4,
                ),
            )
        if diagnosis is Diagnosis.WRONG_DOCUMENT:
            return "drop_document_filter", config.copy_with(document_ids=None, min_score=0.0)
        return (
            "widen_and_rerank",
            config.copy_with(candidate_pool=int(config.candidate_pool * 1.5), rerank=True),
        )


__all__ = ["CorrectiveRAG", "GradingResult"]
