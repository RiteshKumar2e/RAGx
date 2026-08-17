"""Query rewriting and expansion.

Used by Corrective RAG when retrieval quality is poor, and by the Agentic loop
when a sub-question needs sharpening. Every rewrite path has a deterministic
fallback so the system still improves its query without an LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import TraceRecorder, get_logger
from app.core.text import STOPWORDS, technical_terms, tokenize, truncate_words
from app.llm.base import Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import QUERY_REWRITE_SYSTEM, QUERY_REWRITE_USER
from app.retrieval.base import RetrievedChunk

log = get_logger("ragx.rewrite")


@dataclass
class RewriteResult:
    rewrites: list[str] = field(default_factory=list)
    strategy: str = "heuristic"
    reasoning: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"rewrites": self.rewrites, "strategy": self.strategy, "reasoning": self.reasoning}


def heuristic_rewrites(query: str, max_rewrites: int = 3) -> RewriteResult:
    """Deterministic rewrites: keyword-only, generalised, and core-noun forms."""
    rewrites: list[str] = []

    terms = technical_terms(query)
    content_words = [t for t in tokenize(query, drop_stopwords=True)]

    if terms:
        rewrites.append(" ".join(dict.fromkeys(terms + content_words[:6])))
    elif content_words:
        rewrites.append(" ".join(content_words[:8]))

    # Generalised: drop interrogatives and qualifiers that rarely appear in text.
    qualifiers = {
        "what", "which", "how", "why", "when", "where", "who", "does", "did", "is",
        "are", "explain", "describe", "compare", "exactly", "specifically", "please",
    }
    generalized = [w for w in query.split() if w.lower().strip("?,.") not in qualifiers]
    if generalized and " ".join(generalized) != query:
        rewrites.append(" ".join(generalized).strip(" ?"))

    # Core-noun form: the longest content words carry the topic.
    if len(content_words) > 3:
        core = sorted(content_words, key=len, reverse=True)[:4]
        rewrites.append(" ".join(core))

    seen: set[str] = set()
    unique: list[str] = []
    for rewrite in rewrites:
        cleaned = " ".join(rewrite.split())
        key = cleaned.lower()
        if cleaned and key != query.lower().strip(" ?") and key not in seen:
            seen.add(key)
            unique.append(cleaned)

    return RewriteResult(
        rewrites=unique[:max_rewrites],
        strategy="heuristic",
        reasoning="Generated keyword-dense, generalised and core-term variants of the original query.",
    )


async def llm_rewrites(
    query: str,
    retrieved: list[RetrievedChunk],
    diagnosis: str,
    trace: TraceRecorder | None = None,
    max_rewrites: int = 3,
) -> RewriteResult:
    gateway = get_gateway()
    if not gateway.any_configured:
        return heuristic_rewrites(query, max_rewrites)

    snippet = (
        "\n".join(
            f"- ({c.document_name}) {truncate_words(c.content, 45)}" for c in retrieved[:5]
        )
        or "(nothing was retrieved)"
    )
    try:
        payload, _ = await gateway.complete_json(
            [
                Message.system(QUERY_REWRITE_SYSTEM),
                Message.user(
                    QUERY_REWRITE_USER.format(question=query, retrieved=snippet, diagnosis=diagnosis)
                ),
            ],
            Purpose.QUERY_REWRITE,
            default={},
            temperature=0.2,
            max_output_tokens=500,
            trace=trace,
        )
    except Exception as exc:
        log.warning("rewrite.llm_failed", error=str(exc)[:160])
        return heuristic_rewrites(query, max_rewrites)

    if not isinstance(payload, dict):
        return heuristic_rewrites(query, max_rewrites)

    rewrites = [
        " ".join(str(r).split())
        for r in (payload.get("rewrites") or [])
        if isinstance(r, str) and r.strip() and r.strip().lower() != query.strip().lower()
    ]
    if not rewrites:
        return heuristic_rewrites(query, max_rewrites)

    return RewriteResult(
        rewrites=rewrites[:max_rewrites],
        strategy=str(payload.get("strategy", "llm"))[:80] or "llm",
        reasoning=str(payload.get("reasoning", ""))[:400],
    )


def expand_with_entities(query: str, entities: list[str], limit: int = 4) -> str:
    """Append known entity surface forms that are not already in the query."""
    lowered = query.lower()
    additions = [
        entity for entity in entities
        if entity and entity.lower() not in lowered and entity.lower() not in STOPWORDS
    ][:limit]
    return f"{query} {' '.join(additions)}".strip() if additions else query


def condense_history(query: str, history: list[dict[str, str]], max_turns: int = 3) -> str:
    """Resolve follow-up questions against recent turns.

    A query like "and on the second dataset?" is unretrievable on its own; we
    prepend the salient terms of the recent conversation so the embedding and
    BM25 probes carry the missing subject. Purely lexical -- no LLM call.
    """
    if not history or len(tokenize(query)) > 6:
        return query

    recent = history[-max_turns * 2 :]
    terms: list[str] = []
    for turn in reversed(recent):
        if turn.get("role") != "user":
            continue
        for term in technical_terms(turn.get("content", "")):
            if term.lower() not in query.lower() and term not in terms:
                terms.append(term)
        if len(terms) >= 5:
            break
    return f"{query} {' '.join(terms[:5])}".strip() if terms else query


__all__ = [
    "RewriteResult",
    "heuristic_rewrites",
    "llm_rewrites",
    "expand_with_entities",
    "condense_history",
]
