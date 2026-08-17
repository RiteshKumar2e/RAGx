"""Claim extraction.

Splits a generated answer into atomic, checkable assertions and records which
``[n]`` citation markers each one carried. The LLM path handles pronoun
resolution and compound sentences; the deterministic path (sentence
segmentation + marker parsing + filtering of hedges and meta-commentary) runs
when no LLM is available, so verification never silently disappears.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import TraceRecorder, get_logger
from app.core.text import split_sentences, tokenize
from app.llm.base import Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import CLAIM_EXTRACTION_SYSTEM, CLAIM_EXTRACTION_USER

log = get_logger("ragx.verification.claims")

CITATION_MARKER = re.compile(r"\[(\d{1,2})\]")
ABSTENTION = "insufficient evidence found in the indexed knowledge base"

# Sentences that assert nothing checkable.
_META_PREFIXES = (
    "based on the", "according to the provided", "the evidence", "in summary",
    "to summarize", "overall,", "however,", "note that", "it is worth",
    "the retrieved", "the documents", "here is", "here's",
)
_HEDGES = ("might", "may be", "could be", "possibly", "perhaps", "it seems", "appears to")


@dataclass
class Claim:
    index: int
    text: str
    cited: list[int] = field(default_factory=list)
    claim_type: str = "factual"

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "text": self.text, "cited": self.cited, "type": self.claim_type}


def strip_markers(text: str) -> str:
    return CITATION_MARKER.sub("", text).replace("  ", " ").strip()


def is_abstention(answer: str) -> bool:
    return ABSTENTION in (answer or "").lower()


def heuristic_claims(answer: str, max_claims: int = 15) -> list[Claim]:
    claims: list[Claim] = []
    for sentence in split_sentences(answer or ""):
        cleaned = sentence.strip()
        if not cleaned or len(cleaned) < 25:
            continue
        lowered = cleaned.lower()
        if ABSTENTION in lowered:
            continue
        if lowered.startswith(_META_PREFIXES):
            continue
        # Markdown table rows, list bullets and headings are structure, not claims.
        if cleaned.startswith(("|", "#", "```")):
            continue
        if len(tokenize(cleaned)) < 4:
            continue

        cited = [int(m) for m in CITATION_MARKER.findall(cleaned)]
        claim_type = "factual"
        if re.search(r"\d", cleaned):
            claim_type = "numeric"
        if any(w in lowered for w in ("more than", "less than", "better", "worse", "outperform", "compared")):
            claim_type = "comparative"
        if any(w in lowered for w in ("because", "therefore", "causes", "leads to", "due to")):
            claim_type = "causal"
        if any(w in lowered for w in (" is a ", " is the ", " refers to", " means ")):
            claim_type = "definition"
        if any(h in lowered for h in _HEDGES):
            claim_type = "hedged"

        claims.append(
            Claim(index=len(claims), text=strip_markers(cleaned), cited=cited, claim_type=claim_type)
        )
        if len(claims) >= max_claims:
            break
    return claims


async def extract_claims(
    answer: str, trace: TraceRecorder | None = None, max_claims: int = 15
) -> tuple[list[Claim], str]:
    """Return ``(claims, method)``."""
    if not answer or is_abstention(answer):
        return [], "abstention"

    gateway = get_gateway()
    fallback = heuristic_claims(answer, max_claims)
    if not gateway.any_configured:
        return fallback, "heuristic"

    try:
        payload, _ = await gateway.complete_json(
            [
                Message.system(CLAIM_EXTRACTION_SYSTEM),
                Message.user(CLAIM_EXTRACTION_USER.format(answer=answer[:8000])),
            ],
            Purpose.CLAIM_EXTRACTION,
            default={},
            temperature=0.0,
            max_output_tokens=1100,
            trace=trace,
        )
    except Exception as exc:
        log.warning("claims.llm_failed", error=str(exc)[:160])
        return fallback, "heuristic"

    raw_claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(raw_claims, list) or not raw_claims:
        return fallback, "heuristic"

    claims: list[Claim] = []
    for raw in raw_claims[:max_claims]:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if len(text) < 15 or ABSTENTION in text.lower():
            continue
        cited = [int(c) for c in (raw.get("cited") or []) if isinstance(c, (int, float))]
        claims.append(
            Claim(
                index=len(claims),
                text=strip_markers(text),
                cited=cited,
                claim_type=str(raw.get("type", "factual"))[:24],
            )
        )
    if not claims:
        return fallback, "heuristic"
    return claims, "llm"


__all__ = ["Claim", "extract_claims", "heuristic_claims", "strip_markers", "is_abstention", "CITATION_MARKER", "ABSTENTION"]
