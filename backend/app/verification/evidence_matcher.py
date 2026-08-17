"""Evidence matching -- does the retrieved context actually support each claim?

Two independent judgements per claim:

* **Lexical support** (free). Content-word overlap between the claim and the
  best-matching evidence passage, with an extra requirement that any *number* in
  the claim appears verbatim in that passage. Numeric hallucination is the most
  common and most damaging failure mode in RAG answers, and a lexical check
  catches it reliably.
* **Entailment judgement** (LLM). A strict grader returns supported /
  partially_supported / unsupported / contradicted per claim, with the evidence
  ids it relied on.

The final support score combines both, with the numeric check able to veto: a
claim whose figures do not appear in any evidence is capped regardless of what
the judge said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import TraceRecorder, get_logger
from app.core.text import lexical_similarity, token_overlap, truncate_words
from app.llm.base import Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import EVIDENCE_MATCHING_SYSTEM, EVIDENCE_MATCHING_USER
from app.retrieval.base import RetrievedChunk
from app.verification.claims import Claim

log = get_logger("ragx.verification.matcher")

NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")

VERDICTS = {"supported", "partially_supported", "unsupported", "contradicted"}


@dataclass
class ClaimVerdict:
    claim: Claim
    verdict: str = "unsupported"
    support_score: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""
    lexical_score: float = 0.0
    numeric_ok: bool = True
    method: str = "lexical"

    @property
    def is_supported(self) -> bool:
        return self.verdict in {"supported", "partially_supported"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.text,
            "claim_index": self.claim.index,
            "claim_type": self.claim.claim_type,
            "cited": self.claim.cited,
            "verdict": self.verdict,
            "support_score": round(self.support_score, 4),
            "evidence_ids": self.evidence_ids,
            "reason": self.reason,
            "lexical_score": round(self.lexical_score, 4),
            "numeric_consistent": self.numeric_ok,
            "method": self.method,
        }


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(text or "")}


def lexical_match(claim: Claim, evidence: list[RetrievedChunk]) -> tuple[float, list[str], bool]:
    """Return ``(best_score, supporting_ids, numbers_are_consistent)``."""
    if not evidence:
        return 0.0, [], False

    scored: list[tuple[float, RetrievedChunk]] = []
    for chunk in evidence:
        overlap = token_overlap(claim.text, chunk.content)
        similarity = lexical_similarity(claim.text, chunk.content)
        scored.append((0.6 * overlap + 0.4 * similarity, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score = scored[0][0]
    supporting = [chunk.chunk_id for score, chunk in scored if score >= max(0.3, best_score * 0.75)][:3]

    claim_numbers = _numbers(claim.text)
    if not claim_numbers:
        numeric_ok = True
    else:
        joined = " ".join(c.content for c in evidence)
        evidence_numbers = _numbers(joined)
        numeric_ok = claim_numbers.issubset(evidence_numbers)

    return round(best_score, 4), supporting, numeric_ok


async def llm_match(
    claims: list[Claim], evidence: list[RetrievedChunk], trace: TraceRecorder | None = None
) -> dict[int, dict[str, Any]] | None:
    gateway = get_gateway()
    if not claims or not evidence or not gateway.any_configured:
        return None

    claims_text = "\n".join(f"{c.index}. {c.text}" for c in claims)
    evidence_text = "\n\n".join(
        f"[{c.chunk_id}] ({c.document_name}{', ' + c.location if c.location else ''})\n"
        f"{truncate_words(c.content, 130)}"
        for c in evidence[:12]
    )

    try:
        payload, _ = await gateway.complete_json(
            [
                Message.system(EVIDENCE_MATCHING_SYSTEM),
                Message.user(
                    EVIDENCE_MATCHING_USER.format(claims=claims_text, evidence=evidence_text)
                ),
            ],
            Purpose.EVIDENCE_MATCHING,
            default={},
            temperature=0.0,
            max_output_tokens=1400,
            trace=trace,
        )
    except Exception as exc:
        log.warning("verification.llm_matching_failed", error=str(exc)[:160])
        return None

    verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
    if not isinstance(verdicts, list) or not verdicts:
        return None

    out: dict[int, dict[str, Any]] = {}
    valid_ids = {c.chunk_id for c in evidence}
    for raw in verdicts:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("claim_index", -1))
        except (TypeError, ValueError):
            continue
        verdict = str(raw.get("verdict", "unsupported")).lower().strip()
        if verdict not in VERDICTS:
            verdict = "unsupported"
        try:
            score = max(0.0, min(1.0, float(raw.get("support_score", 0.0))))
        except (TypeError, ValueError):
            score = 1.0 if verdict == "supported" else 0.0
        # Only accept evidence ids that were actually shown to the judge.
        ids = [str(i) for i in (raw.get("evidence_ids") or []) if str(i) in valid_ids][:4]
        out[index] = {
            "verdict": verdict,
            "support_score": score,
            "evidence_ids": ids,
            "reason": str(raw.get("reason", ""))[:280],
        }
    return out or None


async def match_claims(
    claims: list[Claim], evidence: list[RetrievedChunk], trace: TraceRecorder | None = None
) -> list[ClaimVerdict]:
    if not claims:
        return []

    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        lexical_score, supporting, numeric_ok = lexical_match(claim, evidence)
        verdicts.append(
            ClaimVerdict(
                claim=claim,
                lexical_score=lexical_score,
                evidence_ids=supporting,
                numeric_ok=numeric_ok,
                support_score=lexical_score,
                verdict=(
                    "supported"
                    if lexical_score >= 0.55 and numeric_ok
                    else "partially_supported"
                    if lexical_score >= 0.3
                    else "unsupported"
                ),
                reason="Lexical overlap with the best-matching evidence passage.",
                method="lexical",
            )
        )

    llm_verdicts = await llm_match(claims, evidence, trace)
    if llm_verdicts:
        for verdict in verdicts:
            judged = llm_verdicts.get(verdict.claim.index)
            if judged is None:
                continue
            verdict.verdict = judged["verdict"]
            verdict.reason = judged["reason"] or verdict.reason
            verdict.method = "lexical+llm"
            if judged["evidence_ids"]:
                verdict.evidence_ids = judged["evidence_ids"]
            # Weight the entailment judgement higher, but keep the lexical
            # signal so a confidently-wrong judge cannot fully override it.
            verdict.support_score = round(0.7 * judged["support_score"] + 0.3 * verdict.lexical_score, 4)

    # Numeric veto: unverifiable figures cap the score no matter the verdict.
    for verdict in verdicts:
        if not verdict.numeric_ok:
            verdict.support_score = round(min(verdict.support_score, 0.35), 4)
            if verdict.verdict == "supported":
                verdict.verdict = "partially_supported"
            verdict.reason = (
                (verdict.reason + " ") if verdict.reason else ""
            ) + "A number in this claim does not appear in the retrieved evidence."

    return verdicts


__all__ = ["ClaimVerdict", "match_claims", "lexical_match", "llm_match"]
