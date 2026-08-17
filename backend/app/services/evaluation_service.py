"""Evaluation orchestration: create runs, execute them, compare results."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.db.session import session_scope
from app.evaluation.benchmark import list_benchmarks, load_benchmark
from app.evaluation.runner import get_evaluation_runner
from app.llm.gateway import get_gateway
from app.llm.embeddings import get_embedding_provider
from app.models.document import Document, DocumentStatus
from app.models.evaluation import EvaluationResult, EvaluationRun

log = get_logger("ragx.evaluation.service")

COMPARISON_METRICS = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "context_relevance",
    "faithfulness",
    "answer_relevance",
    "groundedness",
    "citation_accuracy",
    "avg_latency_ms",
    "p95_latency_ms",
    "avg_total_tokens",
    "estimated_cost_usd",
    "avg_retrieval_calls",
    "abstention_rate",
]

# Metrics where lower is better.
LOWER_IS_BETTER = {
    "avg_latency_ms",
    "p95_latency_ms",
    "avg_total_tokens",
    "estimated_cost_usd",
    "avg_retrieval_calls",
}


class EvaluationService:
    def __init__(self) -> None:
        self.runner = get_evaluation_runner()
        self._running: set[str] = set()

    # ------------------------------------------------------------- benchmark
    def benchmark(self, name: str = "ragx_benchmark") -> dict[str, Any]:
        return load_benchmark(name).as_info()

    def datasets(self) -> list[str]:
        return list_benchmarks()

    # ------------------------------------------------------------------- run
    async def create_runs(
        self, session: AsyncSession, request: Any
    ) -> dict[str, Any]:
        indexed = await session.scalar(
            select(func.count(Document.id)).where(Document.status == DocumentStatus.READY)
        )
        if not indexed:
            raise ValidationError(
                "No documents are indexed. Upload and process at least one document before "
                "running an evaluation -- otherwise every strategy would score zero and the "
                "comparison would be meaningless."
            )

        benchmark = load_benchmark(request.dataset)
        questions = benchmark.filter(request.categories, request.limit)
        if not questions:
            raise ValidationError("The selected categories matched no benchmark questions.")

        warnings: list[str] = []
        gateway = get_gateway()
        if not gateway.any_configured:
            warnings.append(
                "No LLM provider is configured. Retrieval metrics will be computed, but "
                "faithfulness, answer relevance and context relevance will be unavailable, "
                "and pooled relevance labels cannot be produced."
            )
        if not get_embedding_provider().production_ready:
            warnings.append(
                "The development hashing embedder is active, so these results measure lexical "
                "retrieval only and must not be reported as semantic-retrieval benchmarks."
            )
        if not benchmark.has_manual_labels and gateway.any_configured:
            warnings.append(
                "This dataset ships no manual relevance labels, so Recall@K, Precision@K, MRR "
                "and nDCG are computed against pooled LLM judgements over the union of all "
                "evaluated strategies' results."
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        run_ids: dict[str, str] = {}
        for strategy in request.strategies:
            run = EvaluationRun(
                name=request.name or f"{strategy} · {timestamp}",
                dataset=benchmark.name,
                dataset_version=benchmark.version,
                strategy=strategy,
                status="running",
                notes=request.notes,
                question_count=len(questions),
                k=request.k,
                config={
                    "k": request.k,
                    "categories": request.categories,
                    "limit": request.limit,
                    "judge_generation": request.judge_generation,
                    "embedding_provider": get_embedding_provider().name,
                    "embedding_production_ready": get_embedding_provider().production_ready,
                    "llm_configured": gateway.any_configured,
                    "warnings": warnings,
                },
            )
            session.add(run)
            await session.flush()
            run_ids[strategy] = run.id

        return {
            "run_ids": list(run_ids.values()),
            "strategies": list(run_ids.keys()),
            "question_count": len(questions),
            "message": (
                f"Started {len(run_ids)} evaluation run(s) over {len(questions)} questions. "
                "Results appear on the Evaluation page as each run completes."
            ),
            "warnings": warnings,
            "_run_ids_map": run_ids,
        }

    async def execute_runs(self, run_ids_map: dict[str, str], request: Any) -> None:
        """Background execution. Marks runs failed on error rather than hanging."""
        key = ",".join(sorted(run_ids_map.values()))
        if key in self._running:
            return
        self._running.add(key)
        try:
            await self.runner.run(
                strategies=list(run_ids_map.keys()),
                dataset=request.dataset,
                categories=request.categories,
                limit=request.limit,
                k=request.k,
                judge_generation=request.judge_generation,
                name=request.name,
                notes=request.notes,
                run_ids=run_ids_map,
            )
        except Exception as exc:
            log.error("evaluation.run_failed", error=str(exc), exc_info=True)
            async with session_scope() as session:
                for run_id in run_ids_map.values():
                    run = await session.get(EvaluationRun, run_id)
                    if run is not None and run.status == "running":
                        run.status = "failed"
                        run.error_message = str(exc)[:500]
        finally:
            self._running.discard(key)

    # ------------------------------------------------------------------ read
    async def list_runs(
        self, session: AsyncSession, limit: int = 50, strategy: str | None = None
    ) -> list[EvaluationRun]:
        stmt = select(EvaluationRun).order_by(desc(EvaluationRun.created_at)).limit(limit)
        if strategy:
            stmt = stmt.where(EvaluationRun.strategy == strategy)
        return list(await session.scalars(stmt))

    async def get_run(self, session: AsyncSession, run_id: str) -> EvaluationRun:
        run = await session.scalar(
            select(EvaluationRun).options(selectinload(EvaluationRun.results)).where(EvaluationRun.id == run_id)
        )
        if run is None:
            raise NotFoundError(f"Evaluation run '{run_id}' was not found.")
        return run

    async def delete_run(self, session: AsyncSession, run_id: str) -> dict[str, Any]:
        run = await session.get(EvaluationRun, run_id)
        if run is None:
            raise NotFoundError(f"Evaluation run '{run_id}' was not found.")
        await session.delete(run)
        return {"ok": True, "message": f"Evaluation run '{run.name}' was deleted."}

    async def comparison(self, session: AsyncSession) -> dict[str, Any]:
        """Latest completed run per strategy, side by side."""
        runs = list(
            await session.scalars(
                select(EvaluationRun)
                .where(EvaluationRun.status == "completed")
                .order_by(desc(EvaluationRun.created_at))
            )
        )
        latest: dict[str, EvaluationRun] = {}
        for run in runs:
            latest.setdefault(run.strategy, run)

        selected = list(latest.values())
        if not selected:
            return {
                "runs": [],
                "metrics": COMPARISON_METRICS,
                "best_by_metric": {},
                "generated_at": datetime.now(timezone.utc),
                "has_data": False,
                "message": (
                    "No evaluation has been run yet. Metrics appear here only after a real "
                    "benchmark run completes -- nothing on this page is pre-populated."
                ),
            }

        best: dict[str, str] = {}
        for metric in COMPARISON_METRICS:
            candidates = [
                (getattr(run, metric), run.strategy)
                for run in selected
                if getattr(run, metric, None) is not None
            ]
            if not candidates:
                continue
            chooser = min if metric in LOWER_IS_BETTER else max
            best[metric] = chooser(candidates, key=lambda item: item[0])[1]

        return {
            "runs": selected,
            "metrics": COMPARISON_METRICS,
            "best_by_metric": best,
            "generated_at": datetime.now(timezone.utc),
            "has_data": True,
            "message": "",
        }

    async def results(
        self, session: AsyncSession, run_id: str, limit: int = 200
    ) -> list[EvaluationResult]:
        await self.get_run(session, run_id)
        return list(
            await session.scalars(
                select(EvaluationResult)
                .where(EvaluationResult.run_id == run_id)
                .order_by(EvaluationResult.question_id)
                .limit(limit)
            )
        )


_service: EvaluationService | None = None


def get_evaluation_service() -> EvaluationService:
    global _service
    if _service is None:
        _service = EvaluationService()
    return _service


__all__ = ["EvaluationService", "get_evaluation_service", "COMPARISON_METRICS", "LOWER_IS_BETTER"]
