"""Reranking.

Two rerankers, chosen by cost:

* :func:`heuristic_rerank` -- free and always applied. It blends the retrieval
  score with lexical coverage of the query's content terms, exact matches on
  technical identifiers, modality fit and a mild positional prior. This alone
  fixes the common failure where a semantically-adjacent chunk outranks the one
  that literally contains the requested identifier.
* :func:`llm_rerank` -- a cross-encoder-style judgement from the LLM, used when
  the router flags the query as high-stakes or when heuristic scores are flat
  (i.e. the retriever could not discriminate).

The LLM reranker degrades to the heuristic result on any failure.
"""

from __future__ import annotations

import statistics
from typing import Any

from app.core.logging import TraceRecorder, get_logger
from app.core.text import technical_terms, token_overlap, truncate_words
from app.llm.base import Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import RERANK_SYSTEM, RERANK_USER
from app.retrieval.base import RetrievedChunk

log = get_logger("ragx.rerank")


def heuristic_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    prefer_modalities: list[str] | None = None,
) -> list[RetrievedChunk]:
    if not chunks:
        return chunks

    query_terms = {t.lower() for t in technical_terms(query)}
    prefer = set(prefer_modalities or [])

    for chunk in chunks:
        base = chunk.score
        content_lower = chunk.content.lower()

        # Lexical coverage of the query's content words.
        coverage = token_overlap(query, chunk.content)

        # Exact technical-identifier matches are the strongest cheap signal.
        exact = 0.0
        if query_terms:
            hits = sum(1 for term in query_terms if term in content_lower)
            exact = hits / len(query_terms)

        # Agreement across strategies: found by dense *and* sparse is meaningful.
        consensus = min(0.12, 0.06 * max(0, len(chunk.sources) - 1))

        modality_bonus = 0.10 if prefer and chunk.modality in prefer else 0.0

        # Early chunks in a document (abstract, intro) answer definitional
        # questions disproportionately often.
        position_prior = 0.03 if chunk.ordinal <= 3 else 0.0

        # Very short chunks rarely carry a complete answer.
        length_penalty = -0.08 if chunk.token_count and chunk.token_count < 40 else 0.0

        adjusted = (
            base * 0.62
            + coverage * 0.16
            + exact * 0.14
            + consensus
            + modality_bonus
            + position_prior
            + length_penalty
        )
        chunk.metadata["rerank"] = {
            "base": round(base, 4),
            "coverage": round(coverage, 4),
            "exact_term_match": round(exact, 4),
            "consensus": round(consensus, 4),
            "modality_bonus": modality_bonus,
            "method": "heuristic",
        }
        chunk.score = round(max(0.0, min(1.0, adjusted)), 6)

    chunks.sort(key=lambda c: c.score, reverse=True)
    for index, chunk in enumerate(chunks, start=1):
        chunk.rank = index
    return chunks


def scores_are_flat(chunks: list[RetrievedChunk], threshold: float = 0.045) -> bool:
    """True when the retriever failed to separate its top candidates."""
    top = [c.score for c in chunks[:8]]
    if len(top) < 3:
        return False
    try:
        return statistics.pstdev(top) < threshold
    except statistics.StatisticsError:  # pragma: no cover
        return False


async def llm_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = 24,
    trace: TraceRecorder | None = None,
) -> tuple[list[RetrievedChunk], bool]:
    """Rerank with the LLM. Returns ``(chunks, llm_was_used)``."""
    gateway = get_gateway()
    if not chunks or not gateway.any_configured:
        return chunks, False

    candidates = chunks[:top_n]
    passages = "\n\n".join(
        f"[{c.chunk_id}] ({c.document_name}{', ' + c.location if c.location else ''})\n"
        f"{truncate_words(c.content, 110)}"
        for c in candidates
    )

    try:
        payload, _ = await gateway.complete_json(
            [
                Message.system(RERANK_SYSTEM),
                Message.user(RERANK_USER.format(question=query, passages=passages)),
            ],
            Purpose.RERANK,
            default={},
            temperature=0.0,
            max_output_tokens=1200,
            trace=trace,
        )
    except Exception as exc:
        log.warning("rerank.llm_failed", error=str(exc)[:160])
        return chunks, False

    ranking = payload.get("ranking") if isinstance(payload, dict) else None
    if not isinstance(ranking, list) or not ranking:
        return chunks, False

    llm_scores: dict[str, float] = {}
    for item in ranking:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("id", "")).strip()
        try:
            score = max(0.0, min(1.0, float(item.get("score", 0.0))))
        except (TypeError, ValueError):
            continue
        if chunk_id:
            llm_scores[chunk_id] = score

    if not llm_scores:
        return chunks, False

    for chunk in chunks:
        if chunk.chunk_id in llm_scores:
            llm_score = llm_scores[chunk.chunk_id]
            # Blend rather than replace: the retriever's score still carries
            # information the judge does not see (cross-strategy consensus).
            chunk.score = round(0.4 * chunk.score + 0.6 * llm_score, 6)
            chunk.metadata.setdefault("rerank", {})
            chunk.metadata["rerank"].update({"llm_score": round(llm_score, 4), "method": "llm"})
        else:
            # Not returned by the judge -> demoted, not dropped.
            chunk.score = round(chunk.score * 0.55, 6)

    chunks.sort(key=lambda c: c.score, reverse=True)
    for index, chunk in enumerate(chunks, start=1):
        chunk.rank = index
    return chunks, True


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    use_llm: bool = False,
    prefer_modalities: list[str] | None = None,
    top_n: int = 24,
    trace: TraceRecorder | None = None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Full reranking pass. Returns ``(chunks, diagnostics)``."""
    chunks = heuristic_rerank(query, chunks, prefer_modalities)
    diagnostics: dict[str, Any] = {"heuristic": True, "llm": False, "flat_scores": scores_are_flat(chunks)}

    if use_llm or diagnostics["flat_scores"]:
        chunks, used = await llm_rerank(query, chunks, top_n=top_n, trace=trace)
        diagnostics["llm"] = used
        diagnostics["llm_reason"] = (
            "router requested verification-grade ranking" if use_llm else "retrieval scores were not discriminative"
        )
    return chunks, diagnostics


__all__ = ["heuristic_rerank", "llm_rerank", "rerank", "scores_are_flat"]
