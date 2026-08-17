"""Experiment runner.

Executes a benchmark against one or more strategies under identical conditions
and records every result. The protocol has three phases so that all strategies
are scored against the *same* relevance labels:

**Phase A -- execution.** Each (strategy, question) pair runs the full query
pipeline with strategies pinned, so the only thing varying between conditions is
the retrieval strategy. Latency, tokens, cost, retrieval calls and corrective
rounds are captured per run. Generation judges (faithfulness, answer relevance,
context relevance) run here because they only depend on that run's own output.

**Phase B -- pooled labelling.** Every strategy's top-N results for a question
are pooled and judged once (see ``app.evaluation.benchmark.RelevancePool``).

**Phase C -- scoring.** Retrieval metrics are computed for every run against
those shared labels, then aggregated per strategy and persisted.

Conditions compared
-------------------
``naive`` … ``agentic``
    A single fixed strategy, router bypassed. These are the controls.
``adaptive``
    Adaptive routing with the verification layer disabled -- isolates the
    contribution of routing alone.
``ragx``
    The full system: adaptive routing *plus* corrective retrieval and evidence
    verification. The difference between ``adaptive`` and ``ragx`` is the
    measured contribution of the verification layer.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import session_scope
from app.evaluation.benchmark import Benchmark, BenchmarkQuestion, RelevancePool, load_benchmark
from app.evaluation.metrics.generation import evaluate_generation
from app.evaluation.metrics.retrieval import compute_retrieval_metrics, mean_ignoring_none
from app.evaluation.metrics.system import aggregate_system_metrics
from app.llm.gateway import get_gateway
from app.models.evaluation import EvaluationResult, EvaluationRun
from app.retrieval.base import RetrievedChunk
from app.schemas.query import QueryRequest
from app.services.query_service import get_query_service

log = get_logger("ragx.evaluation.runner")

# Conditions that pin a single strategy (router bypassed).
FIXED_STRATEGIES = {"naive", "hybrid", "hyde", "multimodal", "corrective", "graph", "agentic"}
ADAPTIVE_CONDITIONS = {"adaptive", "ragx"}


@dataclass
class RunRecord:
    """One (strategy, question) execution."""

    question: BenchmarkQuestion
    strategy: str
    answer: str = ""
    strategies_used: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    retrieval_calls: int = 0
    llm_calls: int = 0
    corrective_rounds: int = 0
    abstained: bool = False
    judge: Any = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None

    def as_system_record(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "retrieval_calls": self.retrieval_calls,
            "llm_calls": self.llm_calls,
            "corrective_rounds": self.corrective_rounds,
            "abstained": self.abstained,
            "error": self.error,
        }


class EvaluationRunner:
    def __init__(self) -> None:
        self.query_service = get_query_service()

    # ------------------------------------------------------------------ main
    async def run(
        self,
        strategies: list[str],
        *,
        dataset: str = "ragx_benchmark",
        categories: list[str] | None = None,
        limit: int | None = None,
        k: int = 8,
        judge_generation: bool = True,
        name: str | None = None,
        notes: str | None = None,
        run_ids: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        settings = get_settings()
        benchmark: Benchmark = load_benchmark(dataset)
        questions = benchmark.filter(categories, limit)
        if not questions:
            return {"error": "The selected filters matched no benchmark questions."}

        judge_generation = judge_generation and get_gateway().any_configured
        pool = RelevancePool(pool_depth=max(k, 10))

        log.info(
            "evaluation.started",
            strategies=strategies,
            questions=len(questions),
            k=k,
            judges=judge_generation,
        )

        # ---------------------------------------------------- Phase A
        all_records: dict[str, list[RunRecord]] = {}
        for strategy in strategies:
            records = await self._run_strategy(strategy, questions, k, judge_generation, pool)
            all_records[strategy] = records
            if run_ids and strategy in run_ids:
                await self._update_progress(run_ids[strategy], len(records), records)

        # ---------------------------------------------------- Phase B
        label_summary: dict[str, Any] = {"method": "manual" if benchmark.has_manual_labels else "none"}
        if not benchmark.has_manual_labels and judge_generation:
            await pool.judge_all(questions)
            label_summary = pool.summary()

        # ---------------------------------------------------- Phase C
        output: dict[str, Any] = {
            "dataset": benchmark.name,
            "dataset_version": benchmark.version,
            "k": k,
            "question_count": len(questions),
            "label_source": label_summary,
            "strategies": {},
        }

        for strategy, records in all_records.items():
            for record in records:
                labels = (
                    record.question.relevant_chunk_ids
                    if record.question.relevant_chunk_ids
                    else pool.labels(record.question.id)
                )
                record.metrics = compute_retrieval_metrics(record.retrieved_chunk_ids, labels, k)
                record.metrics["labels_available"] = bool(labels)

            summary = self._aggregate(strategy, records, k, benchmark)
            output["strategies"][strategy] = summary

            if run_ids and strategy in run_ids:
                await self._persist(run_ids[strategy], records, summary, label_summary)

        log.info("evaluation.completed", strategies=list(all_records.keys()))
        return output

    # -------------------------------------------------------------- phase A
    async def _run_strategy(
        self,
        strategy: str,
        questions: list[BenchmarkQuestion],
        k: int,
        judge_generation: bool,
        pool: RelevancePool,
    ) -> list[RunRecord]:
        records: list[RunRecord] = []
        for question in questions:
            record = await self._run_one(strategy, question, k, judge_generation)
            records.append(record)
            if record.chunks:
                pool.add(question.id, record.chunks)
        return records

    async def _run_one(
        self, strategy: str, question: BenchmarkQuestion, k: int, judge_generation: bool
    ) -> RunRecord:
        record = RunRecord(question=question, strategy=strategy)
        started = time.perf_counter()

        forced = None if strategy in ADAPTIVE_CONDITIONS else [strategy]
        # ``adaptive`` isolates routing by disabling the verification layer;
        # ``ragx`` is routing + verification, so the delta between them is the
        # measured contribution of verification.
        verify = strategy != "adaptive"

        request = QueryRequest(
            question=question.question,
            strategies=forced,
            top_k=k,
            include_evidence=True,
            include_trace=True,
            verify=verify,
        )

        try:
            async with session_scope() as session:
                response = await self.query_service.execute(session, request)
        except Exception as exc:
            record.error = str(exc)[:300]
            record.latency_ms = (time.perf_counter() - started) * 1000
            log.warning(
                "evaluation.question_failed",
                strategy=strategy,
                question_id=question.id,
                error=record.error,
            )
            return record

        why = response.get("why") or {}
        retrieval = why.get("retrieval") or {}
        generation = why.get("generation") or {}
        trace = response.get("trace") or {}

        record.answer = response.get("answer", "")
        record.strategies_used = response.get("strategies", [])
        record.verification = why.get("verification") or {}
        record.abstained = bool(response.get("abstained"))
        record.latency_ms = float(response.get("total_latency_ms", 0.0))
        record.prompt_tokens = int(trace.get("prompt_tokens", 0))
        record.completion_tokens = int(trace.get("completion_tokens", 0))
        record.total_tokens = int(trace.get("total_tokens", 0))
        record.estimated_cost_usd = float(trace.get("estimated_cost_usd", 0.0))
        record.retrieval_calls = int(retrieval.get("retrieval_calls", 0))
        record.llm_calls = len(trace.get("llm_calls", []))
        record.corrective_rounds = int(retrieval.get("corrective_rounds", 0))

        # Rebuild lightweight chunk objects for metric computation and pooling.
        record.chunks = [
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                document_id=item.get("document_id", ""),
                document_name=item.get("document_name", ""),
                content=item.get("content") or item.get("excerpt", ""),
                score=float(item.get("relevance", 0.0)),
                modality=item.get("modality", "text"),
                page_number=item.get("page"),
                section=item.get("section"),
                figure_label=item.get("figure"),
                table_label=item.get("table"),
            )
            for item in (response.get("evidence") or [])
        ]
        record.retrieved_chunk_ids = [c.chunk_id for c in record.chunks]

        if judge_generation:
            record.judge = await evaluate_generation(
                question.question,
                record.answer,
                record.chunks,
                record.verification,
                run_judges=True,
            )
        else:
            record.judge = await evaluate_generation(
                question.question, record.answer, record.chunks, record.verification, run_judges=False
            )

        return record

    # -------------------------------------------------------------- phase C
    @staticmethod
    def _aggregate(
        strategy: str, records: list[RunRecord], k: int, benchmark: Benchmark
    ) -> dict[str, Any]:
        completed = [r for r in records if not r.error]
        system = aggregate_system_metrics([r.as_system_record() for r in records])

        def judged(attribute: str) -> float | None:
            return mean_ignoring_none(
                [getattr(r.judge, attribute, None) for r in completed if r.judge is not None]
            )

        # Abstention correctness: did the adversarial questions abstain, and did
        # the answerable ones avoid abstaining?
        adversarial = [r for r in completed if r.question.expects_abstention]
        answerable = [r for r in completed if not r.question.expects_abstention]
        correct_abstentions = sum(1 for r in adversarial if r.abstained)
        false_abstentions = sum(1 for r in answerable if r.abstained)

        by_category: dict[str, dict[str, Any]] = {}
        for record in completed:
            bucket = by_category.setdefault(
                record.question.category,
                {"count": 0, "recall": [], "faithfulness": [], "latency": [], "abstained": 0},
            )
            bucket["count"] += 1
            bucket["recall"].append(record.metrics.get("recall_at_k"))
            bucket["faithfulness"].append(getattr(record.judge, "faithfulness", None))
            bucket["latency"].append(record.latency_ms)
            bucket["abstained"] += 1 if record.abstained else 0

        category_summary = {
            name: {
                "count": data["count"],
                "recall_at_k": mean_ignoring_none(data["recall"]),
                "faithfulness": mean_ignoring_none(data["faithfulness"]),
                "avg_latency_ms": round(sum(data["latency"]) / len(data["latency"]), 2)
                if data["latency"]
                else 0.0,
                "abstention_rate": round(data["abstained"] / data["count"], 4) if data["count"] else 0.0,
            }
            for name, data in by_category.items()
        }

        labels_available = any(r.metrics.get("labels_available") for r in completed)

        return {
            "strategy": strategy,
            "question_count": len(records),
            "completed_count": len(completed),
            "failed_count": len(records) - len(completed),
            "k": k,
            "labels_available": labels_available,
            "retrieval": {
                "recall_at_k": mean_ignoring_none([r.metrics.get("recall_at_k") for r in completed]),
                "precision_at_k": mean_ignoring_none([r.metrics.get("precision_at_k") for r in completed]),
                "mrr": mean_ignoring_none([r.metrics.get("reciprocal_rank") for r in completed]),
                "ndcg_at_k": mean_ignoring_none([r.metrics.get("ndcg_at_k") for r in completed]),
                "context_relevance": judged("context_relevance"),
            },
            "generation": {
                "faithfulness": judged("faithfulness"),
                "answer_relevance": judged("answer_relevance"),
                "groundedness": judged("groundedness"),
                "citation_accuracy": judged("citation_accuracy"),
            },
            "system": system.as_dict(),
            "abstention": {
                "total": system.abstentions,
                "rate": system.abstention_rate,
                "adversarial_questions": len(adversarial),
                "correct_abstentions": correct_abstentions,
                "correct_abstention_rate": round(correct_abstentions / len(adversarial), 4)
                if adversarial
                else None,
                "false_abstentions": false_abstentions,
                "false_abstention_rate": round(false_abstentions / len(answerable), 4)
                if answerable
                else None,
            },
            "by_category": category_summary,
            "strategy_usage": _strategy_usage(completed),
        }

    # ------------------------------------------------------------ persistence
    @staticmethod
    async def _update_progress(run_id: str, completed: int, records: list[RunRecord]) -> None:
        async with session_scope() as session:
            run = await session.get(EvaluationRun, run_id)
            if run is not None:
                run.completed_count = completed
                run.failed_count = sum(1 for r in records if r.error)

    @staticmethod
    async def _persist(
        run_id: str, records: list[RunRecord], summary: dict[str, Any], label_summary: dict[str, Any]
    ) -> None:
        async with session_scope() as session:
            run = await session.get(EvaluationRun, run_id)
            if run is None:
                return

            retrieval = summary["retrieval"]
            generation = summary["generation"]
            system = summary["system"]

            run.status = "completed"
            run.question_count = summary["question_count"]
            run.completed_count = summary["completed_count"]
            run.failed_count = summary["failed_count"]
            run.recall_at_k = retrieval["recall_at_k"]
            run.precision_at_k = retrieval["precision_at_k"]
            run.mrr = retrieval["mrr"]
            run.ndcg_at_k = retrieval["ndcg_at_k"]
            run.context_relevance = retrieval["context_relevance"]
            run.answer_relevance = generation["answer_relevance"]
            run.faithfulness = generation["faithfulness"]
            run.groundedness = generation["groundedness"]
            run.citation_accuracy = generation["citation_accuracy"]
            run.avg_latency_ms = system["avg_latency_ms"]
            run.p95_latency_ms = system["p95_latency_ms"]
            run.avg_total_tokens = system["avg_total_tokens"]
            run.total_tokens = system["total_tokens"]
            run.estimated_cost_usd = system["estimated_cost_usd"]
            run.avg_retrieval_calls = system["avg_retrieval_calls"]
            run.corrective_retrievals = system["corrective_retrievals"]
            run.abstention_rate = system["abstention_rate"]
            run.config = {
                **(run.config or {}),
                "label_source": label_summary,
                "abstention": summary["abstention"],
                "by_category": summary["by_category"],
                "strategy_usage": summary["strategy_usage"],
                "labels_available": summary["labels_available"],
            }

            for record in records:
                session.add(
                    EvaluationResult(
                        run_id=run_id,
                        question_id=record.question.id,
                        question=record.question.question,
                        category=record.question.category,
                        answer=(record.answer or "")[:6000],
                        strategies_used=record.strategies_used,
                        retrieved_chunk_ids=record.retrieved_chunk_ids[:30],
                        relevant_chunk_ids=record.question.relevant_chunk_ids[:30],
                        recall_at_k=record.metrics.get("recall_at_k"),
                        precision_at_k=record.metrics.get("precision_at_k"),
                        reciprocal_rank=record.metrics.get("reciprocal_rank"),
                        ndcg_at_k=record.metrics.get("ndcg_at_k"),
                        context_relevance=getattr(record.judge, "context_relevance", None),
                        answer_relevance=getattr(record.judge, "answer_relevance", None),
                        faithfulness=getattr(record.judge, "faithfulness", None),
                        groundedness=getattr(record.judge, "groundedness", None),
                        citation_accuracy=getattr(record.judge, "citation_accuracy", None),
                        latency_ms=record.latency_ms,
                        total_tokens=record.total_tokens,
                        estimated_cost_usd=record.estimated_cost_usd,
                        retrieval_calls=record.retrieval_calls,
                        corrective_rounds=record.corrective_rounds,
                        abstained=record.abstained,
                        judge_detail=getattr(record.judge, "detail", {}) or {},
                        error_message=record.error,
                    )
                )


def _strategy_usage(records: list[RunRecord]) -> dict[str, int]:
    """Which underlying strategies actually ran (meaningful for adaptive/ragx)."""
    counts: dict[str, int] = {}
    for record in records:
        for strategy in record.strategies_used:
            counts[strategy] = counts.get(strategy, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


_runner: EvaluationRunner | None = None


def get_evaluation_runner() -> EvaluationRunner:
    global _runner
    if _runner is None:
        _runner = EvaluationRunner()
    return _runner


__all__ = ["EvaluationRunner", "RunRecord", "get_evaluation_runner", "FIXED_STRATEGIES", "ADAPTIVE_CONDITIONS"]
