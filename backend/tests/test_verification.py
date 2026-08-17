"""Evidence verification: claims, citations, confidence and abstention."""

from __future__ import annotations

import pytest

from app.retrieval.base import RetrievedChunk
from app.verification.citations import analyze_citations, strip_invalid_markers
from app.verification.claims import heuristic_claims, is_abstention, strip_markers
from app.verification.confidence import compute_confidence, label_for
from app.verification.evidence_matcher import lexical_match, match_claims
from app.verification.pipeline import INSUFFICIENT_EVIDENCE_MESSAGE, VerificationPipeline


def chunk(chunk_id: str, content: str, score: float = 0.9, document_id: str = "doc1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        content=content,
        score=score,
        page_number=7,
        section="Results",
        sources=["hybrid"],
        strategy_scores={"hybrid": score},
    )


EVIDENCE = [
    chunk("c1", "DefectNet reaches 78.4 mAP on the NEU-DET benchmark, outperforming YOLOv5s at 74.1 mAP."),
    chunk("c2", "Training used the Adam optimizer with a learning rate of 0.001.", 0.8, "doc2"),
]


# -------------------------------------------------------------------- claims
def test_claims_extracted_with_markers() -> None:
    answer = (
        "DefectNet reaches 78.4 mAP on the NEU-DET benchmark [1]. "
        "Training used the Adam optimizer [2]."
    )
    claims = heuristic_claims(answer)
    assert len(claims) == 2
    assert claims[0].cited == [1]
    assert "[1]" not in claims[0].text


def test_abstention_recognised() -> None:
    assert is_abstention(INSUFFICIENT_EVIDENCE_MESSAGE)
    assert not is_abstention("DefectNet reaches 78.4 mAP.")


def test_markers_stripped() -> None:
    assert strip_markers("A fact [1][2].") == "A fact ."


# ----------------------------------------------------------------- citations
def test_valid_citations_marked_used() -> None:
    answer = "DefectNet reaches 78.4 mAP on the NEU-DET benchmark [1]."
    report = analyze_citations(answer, EVIDENCE)
    assert report.valid_markers == [1]
    assert report.invalid_markers == []
    assert report.citations[0].used_in_answer is True
    assert report.coverage == 1.0


def test_hallucinated_citation_detected() -> None:
    answer = "DefectNet reaches 78.4 mAP on the NEU-DET benchmark [9]."
    report = analyze_citations(answer, EVIDENCE)
    assert report.invalid_markers == [9]
    assert report.accuracy == 0.0


def test_uncited_sentence_lowers_coverage() -> None:
    answer = (
        "DefectNet reaches 78.4 mAP on the NEU-DET benchmark [1]. "
        "It is also widely deployed in production factories worldwide."
    )
    report = analyze_citations(answer, EVIDENCE)
    assert report.factual_sentences == 2
    assert report.cited_sentences == 1
    assert report.coverage == 0.5


def test_invalid_markers_can_be_stripped() -> None:
    assert strip_invalid_markers("Fact [1] and fiction [9].", {1}) == "Fact [1] and fiction ."


# ------------------------------------------------------------ evidence match
def test_numeric_hallucination_detected() -> None:
    from app.verification.claims import Claim

    good = Claim(index=0, text="DefectNet reaches 78.4 mAP on NEU-DET.")
    bad = Claim(index=1, text="DefectNet reaches 91.2 mAP on NEU-DET.")

    _, _, good_ok = lexical_match(good, EVIDENCE)
    _, _, bad_ok = lexical_match(bad, EVIDENCE)
    assert good_ok is True
    assert bad_ok is False


@pytest.mark.anyio
async def test_numeric_mismatch_caps_support_score() -> None:
    from app.verification.claims import Claim

    verdicts = await match_claims(
        [Claim(index=0, text="DefectNet reaches 91.2 mAP on the NEU-DET benchmark.")], EVIDENCE
    )
    assert verdicts[0].numeric_ok is False
    assert verdicts[0].support_score <= 0.35


# ---------------------------------------------------------------- confidence
def test_confidence_labels() -> None:
    assert label_for(0.9) == "high"
    assert label_for(0.5) == "medium"
    assert label_for(0.1) == "low"


def test_hallucinated_citation_penalises_confidence() -> None:
    clean = analyze_citations("DefectNet reaches 78.4 mAP on the NEU-DET benchmark [1].", EVIDENCE)
    dirty = analyze_citations("DefectNet reaches 78.4 mAP on the NEU-DET benchmark [9].", EVIDENCE)
    clean_score = compute_confidence([], clean, EVIDENCE).score
    dirty_report = compute_confidence([], dirty, EVIDENCE)
    assert dirty_report.score < clean_score
    assert "hallucinated_citations" in dirty_report.penalties


def test_abstention_reported_separately() -> None:
    report = compute_confidence([], analyze_citations("", EVIDENCE), EVIDENCE, abstained=True)
    assert report.label == "abstained"
    assert report.rationale


# ------------------------------------------------------------------ pipeline
@pytest.mark.anyio
async def test_no_evidence_forces_abstention() -> None:
    report = await VerificationPipeline().verify(
        "What is the revenue?", "The revenue was $4.2 million in 2019.", []
    )
    assert report.abstained is True
    assert INSUFFICIENT_EVIDENCE_MESSAGE in report.answer
    assert "4.2 million" not in report.answer


@pytest.mark.anyio
async def test_unsupported_answer_is_replaced_not_shipped() -> None:
    """The central safety property: an unsupported answer never reaches the user."""
    weak = [chunk("c9", "This document discusses unrelated manufacturing logistics.", 0.12)]
    report = await VerificationPipeline().verify(
        "What mAP does DefectNet achieve?",
        "DefectNet achieves 95.7 mAP and is the state of the art [1].",
        weak,
    )
    assert report.abstained is True
    assert "95.7" not in report.answer


@pytest.mark.anyio
async def test_correct_abstention_is_preserved() -> None:
    report = await VerificationPipeline().verify(
        "What is the revenue?", INSUFFICIENT_EVIDENCE_MESSAGE, EVIDENCE
    )
    assert report.abstained is True
    assert report.confidence.label == "abstained"
