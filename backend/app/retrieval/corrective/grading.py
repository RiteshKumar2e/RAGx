"""Retrieval-quality evaluation for Corrective RAG.

Grading runs in two tiers so the cheap signal can veto the expensive one:

* **Heuristic grading** always runs. It reads score magnitude, score spread,
  lexical coverage of the query, coverage of the query's technical terms and
  result count. These catch the unambiguous failures -- nothing retrieved,
  everything scored near zero, no query term appears anywhere -- for free.
* **LLM grading** runs when the heuristic verdict is borderline. A judge sees the
  passages and returns a per-passage relevance score and a failure diagnosis,
  which is what tells the rewriter *how* to rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.logging import TraceRecorder, get_logger
from app.core.text import technical_terms, token_overlap, truncate_words
from app.llm.base import Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import RELEVANCE_GRADING_SYSTEM, RELEVANCE_GRADING_USER
from app.retrieval.base import RetrievedChunk

log = get_logger("ragx.corrective.grading")


class Diagnosis(str, Enum):
    GOOD = "good"
    EMPTY = "empty"
    OFF_TOPIC = "off_topic"
    TOO_GENERAL = "too_general"
    PARTIALLY_RELEVANT = "partially_relevant"
    WRONG_DOCUMENT = "wrong_document"

    @property
    def hint(self) -> str:
        return {
            "good": "Retrieval quality is acceptable.",
            "empty": "Nothing was retrieved. The query terms may not exist in the knowledge base.",
            "off_topic": "Retrieved passages are about a different subject than the query.",
            "too_general": "Retrieved passages are topically related but too general to answer the question.",
            "partially_relevant": "Some passages help; key specifics are missing.",
            "wrong_document": "Passages come from documents unlikely to contain the answer.",
        }[self.value]


@dataclass
class GradingResult:
    overall: float = 0.0
    diagnosis: Diagnosis = Diagnosis.GOOD
    per_chunk: dict[str, float] = field(default_factory=dict)
    method: str = "heuristic"
    reasons: dict[str, str] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def is_poor(self) -> bool:
        return self.diagnosis is not Diagnosis.GOOD

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "diagnosis": self.diagnosis.value,
            "diagnosis_hint": self.diagnosis.hint,
            "method": self.method,
            "signals": self.signals,
            "graded_chunks": len(self.per_chunk),
        }


def heuristic_grade(query: str, chunks: list[RetrievedChunk], floor: float) -> GradingResult:
    if not chunks:
        return GradingResult(
            overall=0.0,
            diagnosis=Diagnosis.EMPTY,
            method="heuristic",
            signals={"result_count": 0},
        )

    top_scores = [c.score for c in chunks[:5]]
    top_score = max(top_scores)
    mean_top = sum(top_scores) / len(top_scores)

    coverages = [token_overlap(query, c.content) for c in chunks[:5]]
    best_coverage = max(coverages)
    mean_coverage = sum(coverages) / len(coverages)

    terms = [t.lower() for t in technical_terms(query)]
    term_hit_rate = 0.0
    if terms:
        joined = " ".join(c.content.lower() for c in chunks[:6])
        term_hit_rate = sum(1 for t in terms if t in joined) / len(terms)

    documents = {c.document_id for c in chunks[:5]}

    # Composite quality: retrieval confidence plus evidence of actual overlap.
    quality = 0.45 * top_score + 0.2 * mean_top + 0.2 * best_coverage + 0.15 * (term_hit_rate or mean_coverage)

    signals = {
        "result_count": len(chunks),
        "top_score": round(top_score, 4),
        "mean_top_score": round(mean_top, 4),
        "best_query_coverage": round(best_coverage, 4),
        "technical_term_hit_rate": round(term_hit_rate, 4),
        "distinct_documents": len(documents),
    }

    if terms and term_hit_rate == 0.0 and best_coverage < 0.25:
        diagnosis = Diagnosis.OFF_TOPIC
    elif top_score < floor * 0.6:
        diagnosis = Diagnosis.OFF_TOPIC
    elif best_coverage < 0.2 and top_score < floor:
        diagnosis = Diagnosis.TOO_GENERAL
    elif quality < floor:
        diagnosis = Diagnosis.PARTIALLY_RELEVANT
    else:
        diagnosis = Diagnosis.GOOD

    return GradingResult(
        overall=round(min(1.0, quality), 4),
        diagnosis=diagnosis,
        per_chunk={c.chunk_id: round(c.score, 4) for c in chunks},
        method="heuristic",
        signals=signals,
    )


async def llm_grade(
    query: str, chunks: list[RetrievedChunk], trace: TraceRecorder | None = None, limit: int = 10
) -> GradingResult | None:
    gateway = get_gateway()
    if not chunks or not gateway.any_configured:
        return None

    candidates = chunks[:limit]
    passages = "\n\n".join(
        f"[{c.chunk_id}] ({c.document_name}{', ' + c.location if c.location else ''})\n"
        f"{truncate_words(c.content, 120)}"
        for c in candidates
    )
    try:
        payload, _ = await gateway.complete_json(
            [
                Message.system(RELEVANCE_GRADING_SYSTEM),
                Message.user(RELEVANCE_GRADING_USER.format(question=query, passages=passages)),
            ],
            Purpose.RELEVANCE_GRADING,
            default={},
            temperature=0.0,
            max_output_tokens=900,
            trace=trace,
        )
    except Exception as exc:
        log.warning("corrective.llm_grading_failed", error=str(exc)[:160])
        return None

    if not isinstance(payload, dict):
        return None

    per_chunk: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for item in payload.get("grades", []) or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("id", "")).strip()
        if not chunk_id:
            continue
        try:
            per_chunk[chunk_id] = max(0.0, min(1.0, float(item.get("score", 0.0))))
        except (TypeError, ValueError):
            continue
        reason = str(item.get("reason", ""))[:200]
        if reason:
            reasons[chunk_id] = reason

    if not per_chunk:
        return None

    try:
        overall = max(0.0, min(1.0, float(payload.get("overall", 0.0))))
    except (TypeError, ValueError):
        overall = sum(per_chunk.values()) / len(per_chunk)

    raw_diagnosis = str(payload.get("diagnosis", "good")).lower().strip()
    try:
        diagnosis = Diagnosis(raw_diagnosis)
    except ValueError:
        diagnosis = Diagnosis.GOOD if overall >= 0.6 else Diagnosis.PARTIALLY_RELEVANT

    return GradingResult(
        overall=round(overall, 4),
        diagnosis=diagnosis,
        per_chunk=per_chunk,
        method="llm",
        reasons=reasons,
        signals={
            "graded": len(per_chunk),
            "relevant_count": sum(1 for v in per_chunk.values() if v >= 0.5),
            "max_grade": round(max(per_chunk.values()), 4),
        },
    )


async def grade_retrieval(
    query: str,
    chunks: list[RetrievedChunk],
    floor: float,
    trace: TraceRecorder | None = None,
    allow_llm: bool = True,
) -> GradingResult:
    """Heuristic grading, escalated to the LLM only when the verdict is close."""
    heuristic = heuristic_grade(query, chunks, floor)

    borderline = abs(heuristic.overall - floor) < 0.22
    if not allow_llm or not chunks or not borderline:
        return heuristic

    llm = await llm_grade(query, chunks, trace=trace)
    if llm is None:
        return heuristic

    llm.signals.update({"heuristic_overall": heuristic.overall, **heuristic.signals})
    llm.method = "heuristic+llm"
    return llm


__all__ = ["Diagnosis", "GradingResult", "heuristic_grade", "llm_grade", "grade_retrieval"]
