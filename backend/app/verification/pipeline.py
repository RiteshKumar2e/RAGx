"""The Evidence Verification pipeline.

    answer + evidence
      -> relevance evaluation
      -> claim extraction
      -> evidence matching
      -> source validation
      -> confidence scoring
      -> citation validation
      -> verified answer (or an explicit abstention)

Two properties matter:

* It runs **after** generation and can *change the outcome*: if the evidence
  does not support the answer, the pipeline replaces it with the abstention
  message rather than shipping an unsupported answer with a low score attached.
* It is **fully inspectable**: every claim, verdict, citation and confidence
  component is returned so the "Why this answer?" panel shows real data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import TraceRecorder, get_logger
from app.retrieval.base import RetrievedChunk
from app.verification.citations import CitationReport, analyze_citations, strip_invalid_markers
from app.verification.claims import ABSTENTION, Claim, extract_claims, is_abstention
from app.verification.confidence import ConfidenceReport, compute_confidence
from app.verification.evidence_matcher import ClaimVerdict, match_claims

log = get_logger("ragx.verification")

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient evidence found in the indexed knowledge base."


@dataclass
class VerificationReport:
    answer: str = ""
    original_answer: str = ""
    answer_modified: bool = False
    abstained: bool = False
    claims: list[Claim] = field(default_factory=list)
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    citations: CitationReport = field(default_factory=CitationReport)
    confidence: ConfidenceReport = field(default_factory=ConfidenceReport)
    claim_extraction_method: str = "none"
    latency_ms: float = 0.0
    enabled: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def claims_supported(self) -> int:
        return sum(1 for v in self.verdicts if v.is_supported)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "abstained": self.abstained,
            "answer_modified": self.answer_modified,
            "claims_total": len(self.claims),
            "claims_supported": self.claims_supported,
            "claims_unsupported": sum(1 for v in self.verdicts if v.verdict == "unsupported"),
            "claims_contradicted": sum(1 for v in self.verdicts if v.verdict == "contradicted"),
            "claim_extraction_method": self.claim_extraction_method,
            "claim_verdicts": [v.as_dict() for v in self.verdicts],
            "citations": self.citations.as_dict(),
            "confidence": self.confidence.as_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "notes": self.notes,
        }


class VerificationPipeline:
    async def verify(
        self,
        question: str,
        answer: str,
        evidence: list[RetrievedChunk],
        *,
        trace: TraceRecorder | None = None,
        corrective_rounds: int = 0,
        corrective_resolved: bool = True,
        retrieval_quality: float | None = None,
    ) -> VerificationReport:
        settings = get_settings()
        started = time.perf_counter()
        report = VerificationReport(answer=answer, original_answer=answer, enabled=settings.verification_enabled)

        if not settings.verification_enabled:
            report.citations = analyze_citations(answer, evidence)
            report.notes.append("Verification is disabled in configuration.")
            report.latency_ms = (time.perf_counter() - started) * 1000
            return report

        # -- Stage 1: relevance / sufficiency gate ---------------------------
        # If retrieval never found anything usable, no amount of claim checking
        # will help; abstain before spending LLM calls.
        if not evidence:
            report.abstained = True
            report.answer = (
                f"{INSUFFICIENT_EVIDENCE_MESSAGE}\n\n"
                "No passages in the indexed documents matched this question. "
                "Try rephrasing it, or upload a document that covers this topic."
            )
            report.answer_modified = answer.strip() != report.answer.strip()
            report.confidence = compute_confidence([], report.citations, [], abstained=True)
            report.latency_ms = (time.perf_counter() - started) * 1000
            return report

        if is_abstention(answer):
            report.abstained = True
            report.citations = analyze_citations(answer, evidence)
            report.confidence = compute_confidence([], report.citations, evidence, abstained=True)
            report.notes.append("The model correctly declined to answer from the available evidence.")
            report.latency_ms = (time.perf_counter() - started) * 1000
            return report

        # -- Stage 2: claim extraction ---------------------------------------
        claims, method = await extract_claims(answer, trace=trace)
        report.claims = claims
        report.claim_extraction_method = method

        # -- Stage 3: evidence matching --------------------------------------
        if claims:
            report.verdicts = await match_claims(claims, evidence, trace=trace)

        # -- Stage 4: citation validation ------------------------------------
        report.citations = analyze_citations(answer, evidence)
        if report.citations.invalid_markers:
            # Remove dead markers so the UI cannot link to nothing; the fact
            # that they were emitted is still reported.
            valid = {c.marker for c in report.citations.citations}
            report.answer = strip_invalid_markers(answer, valid)
            report.answer_modified = True
            report.notes.append(
                f"Removed {len(report.citations.invalid_markers)} citation marker(s) that "
                "referenced non-existent evidence."
            )

        # -- Stage 5: confidence ---------------------------------------------
        report.confidence = compute_confidence(
            report.verdicts,
            report.citations,
            evidence,
            corrective_rounds=corrective_rounds,
            corrective_resolved=corrective_resolved,
        )

        # -- Stage 6: abstention enforcement ---------------------------------
        # This is the rule that prevents confident fabrication: when the
        # evidence genuinely does not support the answer, the answer is replaced.
        threshold = settings.insufficient_evidence_threshold
        support_ratio = (
            report.claims_supported / len(report.verdicts) if report.verdicts else 0.0
        )
        quality = retrieval_quality if retrieval_quality is not None else report.confidence.components.get(
            "retrieval_quality", 0.0
        )
        should_abstain = (
            report.confidence.score < threshold
            and (support_ratio < settings.min_claim_support_score or quality < threshold)
        )

        if should_abstain:
            report.abstained = True
            # The withheld answer's text is deliberately NOT echoed here. Quoting
            # an unsupported claim back to the user -- even to explain why it was
            # rejected -- reintroduces the fabricated content the abstention
            # exists to suppress. Only counts and categories are reported; the
            # full verdict list remains available in the verification payload
            # for the "Why this answer?" panel.
            unsupported = [v for v in report.verdicts if not v.is_supported]
            detail_bits: list[str] = []
            if unsupported:
                detail_bits.append(
                    f"{len(unsupported)} of {len(report.verdicts)} statements in the drafted "
                    "answer could not be matched to any retrieved passage"
                )
            if any(not v.numeric_ok for v in report.verdicts):
                detail_bits.append("figures in the draft did not appear in the evidence")
            if quality < threshold:
                detail_bits.append(f"retrieval relevance was low ({quality:.2f})")

            report.answer = (
                f"{INSUFFICIENT_EVIDENCE_MESSAGE}\n\n"
                "The retrieved passages do not support a reliable answer"
                + (f" — {'; '.join(detail_bits)}." if detail_bits else ".")
                + "\n\nTry narrowing the question, or upload a document that covers this topic."
            )
            report.answer_modified = True
            report.notes.append(
                f"The drafted answer was withheld: confidence {report.confidence.score:.2f} "
                f"< threshold {threshold:.2f} with {support_ratio:.0%} claim support. "
                "Its text is not shown, to avoid repeating unsupported content."
            )
            report.confidence = compute_confidence([], report.citations, evidence, abstained=True)
            log.info(
                "verification.abstained",
                claims=len(report.claims),
                supported=report.claims_supported,
                quality=round(quality, 3),
            )

        report.latency_ms = (time.perf_counter() - started) * 1000
        log.info(
            "verification.completed",
            claims=len(report.claims),
            supported=report.claims_supported,
            confidence=report.confidence.score,
            label=report.confidence.label,
            citation_coverage=report.citations.coverage,
            abstained=report.abstained,
            latency_ms=round(report.latency_ms, 1),
        )
        return report


_pipeline: VerificationPipeline | None = None


def get_verification_pipeline() -> VerificationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VerificationPipeline()
    return _pipeline


__all__ = [
    "VerificationPipeline",
    "VerificationReport",
    "get_verification_pipeline",
    "INSUFFICIENT_EVIDENCE_MESSAGE",
    "ABSTENTION",
]
