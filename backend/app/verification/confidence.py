"""Confidence scoring.

The score is a weighted combination of independent signals, each of which can
be inspected in the "Why this answer?" panel. Nothing here is a magic number
pulled from the model -- every component is computed from something observable:

============================  ======  ===================================================
Component                     Weight  What it measures
============================  ======  ===================================================
claim support                 0.34    fraction of claims the evidence entails
retrieval quality             0.22    top and mean relevance of the evidence used
citation coverage             0.18    fraction of factual sentences carrying a citation
citation accuracy             0.10    fraction of emitted markers pointing at real blocks
evidence agreement            0.08    corroboration across distinct documents
strategy consensus            0.08    how many strategies independently surfaced the top evidence
============================  ======  ===================================================

Penalties are then applied for contradicted claims, unresolved corrective
rounds and numeric inconsistency. The result is bucketed into
high / medium / low for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retrieval.base import RetrievedChunk
from app.verification.citations import CitationReport
from app.verification.evidence_matcher import ClaimVerdict

WEIGHTS = {
    "claim_support": 0.34,
    "retrieval_quality": 0.22,
    "citation_coverage": 0.18,
    "citation_accuracy": 0.10,
    "evidence_agreement": 0.08,
    "strategy_consensus": 0.08,
}


@dataclass
class ConfidenceReport:
    score: float = 0.0
    label: str = "low"
    components: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "label": self.label,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "weights": WEIGHTS,
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
            "rationale": self.rationale,
        }


def label_for(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def compute_confidence(
    verdicts: list[ClaimVerdict],
    citations: CitationReport,
    evidence: list[RetrievedChunk],
    *,
    corrective_rounds: int = 0,
    corrective_resolved: bool = True,
    abstained: bool = False,
) -> ConfidenceReport:
    report = ConfidenceReport()

    if abstained:
        # Correctly declining to answer is a high-confidence *decision*; it is
        # reported separately so it is never mistaken for a confident answer.
        report.score = 0.0
        report.label = "abstained"
        report.rationale.append(
            "The system found insufficient evidence and declined to answer, which is the "
            "intended behaviour rather than a low-confidence answer."
        )
        report.components = {"claim_support": 0.0, "retrieval_quality": 0.0}
        return report

    # -- claim support -----------------------------------------------------
    if verdicts:
        supported = sum(v.support_score for v in verdicts) / len(verdicts)
        contradicted = sum(1 for v in verdicts if v.verdict == "contradicted")
        unsupported = sum(1 for v in verdicts if v.verdict == "unsupported")
    else:
        supported, contradicted, unsupported = 0.0, 0, 0
    report.components["claim_support"] = supported

    # -- retrieval quality --------------------------------------------------
    if evidence:
        scores = [c.score for c in evidence]
        top = max(scores)
        mean_top = sum(scores[:5]) / min(5, len(scores))
        retrieval_quality = 0.6 * top + 0.4 * mean_top
    else:
        retrieval_quality = 0.0
    report.components["retrieval_quality"] = retrieval_quality

    # -- citations ----------------------------------------------------------
    report.components["citation_coverage"] = citations.coverage
    report.components["citation_accuracy"] = citations.accuracy if citations.valid_markers or citations.invalid_markers else 0.0

    # -- corroboration across documents ------------------------------------
    used_documents = {c.document_id for c in evidence[:8]}
    agreement = min(1.0, (len(used_documents) - 1) / 2) if len(used_documents) > 1 else 0.35
    report.components["evidence_agreement"] = agreement

    # -- strategy consensus -------------------------------------------------
    if evidence:
        top_sources = [len([s for s in c.sources if not s.startswith("neighbor")]) for c in evidence[:5]]
        consensus = min(1.0, (sum(top_sources) / len(top_sources) - 1) / 2) if top_sources else 0.0
        consensus = max(0.0, consensus)
    else:
        consensus = 0.0
    report.components["strategy_consensus"] = consensus

    base = sum(WEIGHTS[k] * report.components.get(k, 0.0) for k in WEIGHTS)

    # -- penalties ----------------------------------------------------------
    if contradicted:
        penalty = min(0.35, 0.18 * contradicted)
        report.penalties["contradicted_claims"] = penalty
        report.rationale.append(
            f"{contradicted} claim(s) conflict with the retrieved evidence."
        )
    if unsupported and verdicts:
        ratio = unsupported / len(verdicts)
        if ratio > 0.3:
            penalty = min(0.2, ratio * 0.25)
            report.penalties["unsupported_claims"] = penalty
            report.rationale.append(
                f"{unsupported} of {len(verdicts)} claims are not supported by the evidence."
            )
    if citations.invalid_markers:
        penalty = min(0.12, 0.04 * len(citations.invalid_markers))
        report.penalties["hallucinated_citations"] = penalty
        report.rationale.append(
            f"{len(citations.invalid_markers)} citation marker(s) referenced non-existent evidence."
        )
    if corrective_rounds and not corrective_resolved:
        report.penalties["unresolved_correction"] = 0.12
        report.rationale.append(
            "Corrective retrieval ran but did not raise retrieval quality above the floor."
        )
    numeric_failures = sum(1 for v in verdicts if not v.numeric_ok)
    if numeric_failures:
        penalty = min(0.18, 0.06 * numeric_failures)
        report.penalties["numeric_mismatch"] = penalty
        report.rationale.append(
            f"{numeric_failures} claim(s) contain figures that do not appear in the evidence."
        )

    score = max(0.0, min(1.0, base - sum(report.penalties.values())))
    report.score = round(score, 4)
    report.label = label_for(score)

    # -- positive rationale -------------------------------------------------
    if verdicts and supported >= 0.7:
        report.rationale.insert(
            0, f"{sum(1 for v in verdicts if v.is_supported)} of {len(verdicts)} claims are backed by cited evidence."
        )
    if citations.coverage >= 0.8 and citations.factual_sentences:
        report.rationale.insert(0, f"{citations.coverage:.0%} of factual sentences carry a citation.")
    if len(used_documents) > 1:
        report.rationale.append(f"Evidence was corroborated across {len(used_documents)} documents.")
    if not report.rationale:
        report.rationale.append("Confidence reflects retrieval scores and citation coverage.")

    return report


__all__ = ["ConfidenceReport", "compute_confidence", "label_for", "WEIGHTS"]
