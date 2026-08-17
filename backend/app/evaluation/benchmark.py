"""Benchmark dataset loading and pooled relevance labelling.

Datasets are JSON files in ``app/evaluation/datasets``. A dataset may ship
manual ``relevant_chunk_ids`` per question; when it does not, labels are built
at run time by **pooling**:

1. Every strategy under comparison contributes its top-N results for a question
   to a shared pool.
2. An LLM judge labels each pooled passage once for relevance to that question.
3. All strategies are scored against those identical labels.

This is the standard TREC pooling protocol. Its limitation -- a passage that no
strategy retrieved is never judged, so recall is measured relative to the pool --
is stated in the run's config and in the evaluation docs, because the honest
reading of a pooled Recall@K is "recall over the pooled candidate set".
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.evaluation.metrics.generation import judge_context_relevance
from app.retrieval.base import RetrievedChunk

log = get_logger("ragx.evaluation.benchmark")

DATASET_DIR = Path(__file__).parent / "datasets"


@dataclass
class BenchmarkQuestion:
    id: str
    question: str
    category: str = "general"
    difficulty: str = "moderate"
    expects_abstention: bool = False
    notes: str = ""
    relevant_chunk_ids: list[str] = field(default_factory=list)
    reference_answer: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "expects_abstention": self.expects_abstention,
            "notes": self.notes,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "reference_answer": self.reference_answer,
        }


@dataclass
class Benchmark:
    name: str
    version: str
    description: str
    questions: list[BenchmarkQuestion] = field(default_factory=list)
    categories: dict[str, str] = field(default_factory=dict)
    label_source: str = "pooled"

    @property
    def has_manual_labels(self) -> bool:
        return any(q.relevant_chunk_ids for q in self.questions)

    def filter(
        self, categories: list[str] | None = None, limit: int | None = None
    ) -> list[BenchmarkQuestion]:
        questions = self.questions
        if categories:
            wanted = {c.lower() for c in categories}
            questions = [q for q in questions if q.category.lower() in wanted]
        return questions[:limit] if limit else questions

    def as_info(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for question in self.questions:
            counts[question.category] = counts.get(question.category, 0) + 1
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "question_count": len(self.questions),
            "categories": counts,
            "has_relevance_labels": self.has_manual_labels,
            "questions": [q.as_dict() for q in self.questions],
        }


def load_benchmark(name: str = "ragx_benchmark") -> Benchmark:
    path = DATASET_DIR / f"{name}.json"
    if not path.exists():
        raise NotFoundError(f"Benchmark dataset '{name}' was not found in {DATASET_DIR}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        BenchmarkQuestion(
            id=str(raw["id"]),
            question=str(raw["question"]),
            category=str(raw.get("category", "general")),
            difficulty=str(raw.get("difficulty", "moderate")),
            expects_abstention=bool(raw.get("expects_abstention", False)),
            notes=str(raw.get("notes", "")),
            relevant_chunk_ids=[str(c) for c in (raw.get("relevant_chunk_ids") or [])],
            reference_answer=raw.get("reference_answer"),
        )
        for raw in payload.get("questions", [])
    ]
    return Benchmark(
        name=payload.get("name", name),
        version=str(payload.get("version", "1.0.0")),
        description=payload.get("description", ""),
        questions=questions,
        categories=payload.get("categories", {}),
        label_source=payload.get("label_source", "pooled"),
    )


def list_benchmarks() -> list[str]:
    return sorted(p.stem for p in DATASET_DIR.glob("*.json"))


def save_benchmark(benchmark: Benchmark, name: str | None = None) -> Path:
    """Persist a dataset -- used by ``scripts/build_benchmark.py``."""
    target = DATASET_DIR / f"{name or benchmark.name}.json"
    payload = {
        "name": benchmark.name,
        "version": benchmark.version,
        "description": benchmark.description,
        "label_source": benchmark.label_source,
        "categories": benchmark.categories,
        "questions": [q.as_dict() for q in benchmark.questions],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Pooled relevance labelling
# ---------------------------------------------------------------------------
class RelevancePool:
    """Accumulates candidates per question across strategies, then labels once."""

    def __init__(self, pool_depth: int = 10):
        self.pool_depth = pool_depth
        self._pools: dict[str, dict[str, RetrievedChunk]] = {}
        self._labels: dict[str, list[str]] = {}
        self._judged: set[str] = set()

    def add(self, question_id: str, chunks: list[RetrievedChunk]) -> None:
        pool = self._pools.setdefault(question_id, {})
        for chunk in chunks[: self.pool_depth]:
            pool.setdefault(chunk.chunk_id, chunk)

    def has_labels(self, question_id: str) -> bool:
        return question_id in self._judged

    def labels(self, question_id: str) -> list[str]:
        return self._labels.get(question_id, [])

    def pool_size(self, question_id: str) -> int:
        return len(self._pools.get(question_id, {}))

    async def judge(self, question_id: str, question: str) -> list[str]:
        """Label the pool for one question. Idempotent."""
        if question_id in self._judged:
            return self._labels.get(question_id, [])

        pool = list(self._pools.get(question_id, {}).values())
        self._judged.add(question_id)
        if not pool:
            self._labels[question_id] = []
            return []

        _, relevant_ids, _ = await judge_context_relevance(question, pool)
        self._labels[question_id] = relevant_ids or []
        log.info(
            "benchmark.pool_judged",
            question_id=question_id,
            pool_size=len(pool),
            relevant=len(self._labels[question_id]),
        )
        return self._labels[question_id]

    async def judge_all(self, questions: list[BenchmarkQuestion], concurrency: int = 3) -> None:
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(question: BenchmarkQuestion) -> None:
            async with semaphore:
                await self.judge(question.id, question.question)

        await asyncio.gather(*(_one(q) for q in questions), return_exceptions=True)

    def summary(self) -> dict[str, Any]:
        return {
            "method": "pooled_llm_judgement",
            "pool_depth": self.pool_depth,
            "questions_pooled": len(self._pools),
            "questions_judged": len(self._judged),
            "avg_pool_size": round(
                sum(len(p) for p in self._pools.values()) / len(self._pools), 2
            )
            if self._pools
            else 0.0,
            "avg_relevant_per_question": round(
                sum(len(v) for v in self._labels.values()) / len(self._labels), 2
            )
            if self._labels
            else 0.0,
            "caveat": (
                "Recall is measured relative to the pooled candidate set: a passage that no "
                "evaluated strategy retrieved was never judged."
            ),
        }


__all__ = [
    "Benchmark",
    "BenchmarkQuestion",
    "load_benchmark",
    "list_benchmarks",
    "save_benchmark",
    "RelevancePool",
    "DATASET_DIR",
]
