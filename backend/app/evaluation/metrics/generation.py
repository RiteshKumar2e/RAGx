"""Generation-quality metrics.

Four measures, each computed from something observable:

``faithfulness``
    LLM-judged. Fraction of the answer's factual content entailed by the
    retrieved context. A correct abstention scores 1.0.

``answer_relevance``
    LLM-judged. How directly and completely the answer addresses the question.

``context_relevance``
    LLM-judged. Fraction of retrieved passages that are useful for the question.
    Doubles as the source of pooled relevance labels for retrieval metrics.

``groundedness``
    Computed **mechanically** from the verification pipeline: the share of
    extracted claims with supporting evidence. Unlike the judged metrics it does
    not depend on a model's opinion of its own output.

``citation_accuracy``
    Also mechanical: the share of emitted ``[n]`` markers that resolve to a real
    evidence block, combined with citation coverage.

Judged metrics require a configured provider. Without one they return ``None``
and the dashboard reports them as unavailable rather than assuming a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import TraceRecorder, get_logger
from app.core.text import truncate_words
from app.llm.base import Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import (
    ANSWER_RELEVANCE_JUDGE_SYSTEM,
    ANSWER_RELEVANCE_JUDGE_USER,
    CONTEXT_RELEVANCE_JUDGE_SYSTEM,
    CONTEXT_RELEVANCE_JUDGE_USER,
    FAITHFULNESS_JUDGE_SYSTEM,
    FAITHFULNESS_JUDGE_USER,
)
from app.retrieval.base import RetrievedChunk

log = get_logger("ragx.evaluation.generation")


@dataclass
class JudgeScores:
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_relevance: float | None = None
    groundedness: float | None = None
    citation_accuracy: float | None = None
    relevant_chunk_ids: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    judged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "groundedness": self.groundedness,
            "citation_accuracy": self.citation_accuracy,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "judged": self.judged,
            "detail": self.detail,
        }


def _format_context(chunks: list[RetrievedChunk], words: int = 110) -> str:
    return "\n\n".join(
        f"[{c.chunk_id}] ({c.document_name}{', ' + c.location if c.location else ''})\n"
        f"{truncate_words(c.content, words)}"
        for c in chunks[:12]
    ) or "(no context retrieved)"


def _score(payload: Any, default: float | None = None) -> float | None:
    if not isinstance(payload, dict):
        return default
    try:
        return round(max(0.0, min(1.0, float(payload.get("score", 0.0)))), 4)
    except (TypeError, ValueError):
        return default


async def judge_faithfulness(
    question: str, answer: str, chunks: list[RetrievedChunk], trace: TraceRecorder | None = None
) -> tuple[float | None, dict[str, Any]]:
    gateway = get_gateway()
    if not gateway.any_configured or not answer:
        return None, {}
    try:
        payload, _ = await gateway.complete_json(
            [
                Message.system(FAITHFULNESS_JUDGE_SYSTEM),
                Message.user(
                    FAITHFULNESS_JUDGE_USER.format(
                        question=question, context=_format_context(chunks), answer=answer[:6000]
                    )
                ),
            ],
            Purpose.JUDGE,
            default={},
            temperature=0.0,
            max_output_tokens=700,
            trace=trace,
        )
    except Exception as exc:
        log.warning("judge.faithfulness_failed", error=str(exc)[:160])
        return None, {}
    return _score(payload), {
        "supported_claims": payload.get("supported_claims") if isinstance(payload, dict) else None,
        "total_claims": payload.get("total_claims") if isinstance(payload, dict) else None,
        "unsupported": (payload.get("unsupported") or [])[:5] if isinstance(payload, dict) else [],
        "reason": str(payload.get("reason", ""))[:300] if isinstance(payload, dict) else "",
    }


async def judge_answer_relevance(
    question: str, answer: str, trace: TraceRecorder | None = None
) -> tuple[float | None, dict[str, Any]]:
    gateway = get_gateway()
    if not gateway.any_configured or not answer:
        return None, {}
    try:
        payload, _ = await gateway.complete_json(
            [
                Message.system(ANSWER_RELEVANCE_JUDGE_SYSTEM),
                Message.user(ANSWER_RELEVANCE_JUDGE_USER.format(question=question, answer=answer[:6000])),
            ],
            Purpose.JUDGE,
            default={},
            temperature=0.0,
            max_output_tokens=300,
            trace=trace,
        )
    except Exception as exc:
        log.warning("judge.answer_relevance_failed", error=str(exc)[:160])
        return None, {}
    return _score(payload), {"reason": str(payload.get("reason", ""))[:300] if isinstance(payload, dict) else ""}


async def judge_context_relevance(
    question: str, chunks: list[RetrievedChunk], trace: TraceRecorder | None = None
) -> tuple[float | None, list[str], dict[str, Any]]:
    """Returns ``(score, relevant_chunk_ids, detail)``.

    The id list is what pooled relevance labelling is built from.
    """
    gateway = get_gateway()
    if not gateway.any_configured or not chunks:
        return None, [], {}
    try:
        payload, _ = await gateway.complete_json(
            [
                Message.system(CONTEXT_RELEVANCE_JUDGE_SYSTEM),
                Message.user(
                    CONTEXT_RELEVANCE_JUDGE_USER.format(question=question, passages=_format_context(chunks))
                ),
            ],
            Purpose.JUDGE,
            default={},
            temperature=0.0,
            max_output_tokens=600,
            trace=trace,
        )
    except Exception as exc:
        log.warning("judge.context_relevance_failed", error=str(exc)[:160])
        return None, [], {}

    valid = {c.chunk_id for c in chunks}
    relevant = [
        str(i) for i in (payload.get("relevant_ids") or []) if isinstance(payload, dict) and str(i) in valid
    ]
    return _score(payload), relevant, {
        "reason": str(payload.get("reason", ""))[:300] if isinstance(payload, dict) else ""
    }


def compute_groundedness(verification: dict[str, Any]) -> float | None:
    """Mechanical groundedness from the verification report."""
    if not verification or not verification.get("enabled", True):
        return None
    if verification.get("abstained"):
        # Abstaining when evidence is missing is maximally grounded behaviour.
        return 1.0
    total = verification.get("claims_total", 0)
    if not total:
        return None
    return round(verification.get("claims_supported", 0) / total, 4)


def compute_citation_accuracy(verification: dict[str, Any]) -> float | None:
    """Marker validity combined with coverage of factual sentences."""
    citations = (verification or {}).get("citations") or {}
    if verification.get("abstained"):
        return 1.0
    factual = citations.get("factual_sentences", 0)
    if not factual:
        return None
    validity = citations.get("citation_accuracy")
    coverage = citations.get("coverage", 0.0)
    if validity is None:
        return round(coverage, 4)
    return round(0.5 * validity + 0.5 * coverage, 4)


async def evaluate_generation(
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    verification: dict[str, Any],
    *,
    run_judges: bool = True,
    trace: TraceRecorder | None = None,
) -> JudgeScores:
    import asyncio  # noqa: PLC0415

    scores = JudgeScores(
        groundedness=compute_groundedness(verification),
        citation_accuracy=compute_citation_accuracy(verification),
    )

    if not run_judges or not get_gateway().any_configured:
        return scores

    faithfulness, relevance, context = await asyncio.gather(
        judge_faithfulness(question, answer, chunks, trace),
        judge_answer_relevance(question, answer, trace),
        judge_context_relevance(question, chunks, trace),
        return_exceptions=True,
    )

    if isinstance(faithfulness, tuple):
        scores.faithfulness, detail = faithfulness
        scores.detail["faithfulness"] = detail
    if isinstance(relevance, tuple):
        scores.answer_relevance, detail = relevance
        scores.detail["answer_relevance"] = detail
    if isinstance(context, tuple):
        scores.context_relevance, scores.relevant_chunk_ids, detail = context
        scores.detail["context_relevance"] = detail

    scores.judged = any(
        v is not None for v in (scores.faithfulness, scores.answer_relevance, scores.context_relevance)
    )
    return scores


__all__ = [
    "JudgeScores",
    "judge_faithfulness",
    "judge_answer_relevance",
    "judge_context_relevance",
    "compute_groundedness",
    "compute_citation_accuracy",
    "evaluate_generation",
]
