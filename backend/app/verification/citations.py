"""Citation extraction and validation.

The generator is instructed to cite evidence blocks as ``[n]``. This module:

* parses those markers out of the answer,
* validates each against the evidence list that was actually supplied
  (a marker pointing at a block that does not exist is a *hallucinated
  citation* and is reported as one),
* builds the citation records the frontend renders and links to source
  evidence,
* and computes citation coverage -- the share of factual sentences that carry at
  least one valid citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.text import split_sentences, truncate_words
from app.retrieval.base import RetrievedChunk
from app.verification.claims import ABSTENTION, CITATION_MARKER

SENTENCE_NEEDS_CITATION = re.compile(r"[A-Za-z]{3,}")


@dataclass
class Citation:
    marker: int
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_path: list[str] = field(default_factory=list)
    figure: str | None = None
    table: str | None = None
    modality: str = "text"
    asset_key: str | None = None
    relevance: float = 0.0
    excerpt: str = ""
    used_in_answer: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page": self.page,
            "page_end": self.page_end,
            "section": self.section,
            "section_path": self.section_path,
            "figure": self.figure,
            "table": self.table,
            "modality": self.modality,
            "asset_key": self.asset_key,
            "relevance": round(self.relevance, 4),
            "excerpt": self.excerpt,
            "used_in_answer": self.used_in_answer,
        }


@dataclass
class CitationReport:
    citations: list[Citation] = field(default_factory=list)
    valid_markers: list[int] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    coverage: float = 0.0
    cited_sentences: int = 0
    factual_sentences: int = 0
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Share of emitted markers that point at real evidence."""
        total = len(self.valid_markers) + len(self.invalid_markers)
        return round(len(self.valid_markers) / total, 4) if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "citations": [c.as_dict() for c in self.citations],
            "citation_count": len(self.citations),
            "used_count": sum(1 for c in self.citations if c.used_in_answer),
            "valid_markers": self.valid_markers,
            "invalid_markers": self.invalid_markers,
            "hallucinated_citations": len(self.invalid_markers),
            "citation_accuracy": self.accuracy,
            "coverage": round(self.coverage, 4),
            "cited_sentences": self.cited_sentences,
            "factual_sentences": self.factual_sentences,
            "uncited_sentences": self.uncited_sentences[:5],
        }


def build_citations(evidence: list[RetrievedChunk], excerpt_words: int = 60) -> list[Citation]:
    """One citation record per evidence block, numbered from 1."""
    return [
        Citation(
            marker=index,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page=chunk.page_number,
            page_end=chunk.page_end,
            section=chunk.section,
            section_path=chunk.section_path,
            figure=chunk.figure_label,
            table=chunk.table_label,
            modality=chunk.modality,
            asset_key=chunk.asset_key,
            relevance=chunk.score,
            excerpt=truncate_words(chunk.content, excerpt_words),
        )
        for index, chunk in enumerate(evidence, start=1)
    ]


def analyze_citations(answer: str, evidence: list[RetrievedChunk]) -> CitationReport:
    citations = build_citations(evidence)
    by_marker = {c.marker: c for c in citations}

    if not answer or ABSTENTION in answer.lower():
        # An abstention is correct behaviour, not an uncited answer.
        return CitationReport(
            citations=citations,
            coverage=1.0 if ABSTENTION in (answer or "").lower() else 0.0,
            factual_sentences=0,
            cited_sentences=0,
        )

    valid: list[int] = []
    invalid: list[int] = []
    factual_sentences = 0
    cited_sentences = 0
    uncited: list[str] = []

    for sentence in split_sentences(answer):
        stripped = sentence.strip()
        if not stripped or stripped.startswith(("|", "#", "```", "-", "*")):
            continue
        if not SENTENCE_NEEDS_CITATION.search(stripped) or len(stripped) < 25:
            continue

        factual_sentences += 1
        markers = [int(m) for m in CITATION_MARKER.findall(stripped)]
        sentence_has_valid = False
        for marker in markers:
            if marker in by_marker:
                valid.append(marker)
                by_marker[marker].used_in_answer = True
                sentence_has_valid = True
            else:
                invalid.append(marker)
        if sentence_has_valid:
            cited_sentences += 1
        else:
            uncited.append(truncate_words(stripped, 25))

    coverage = cited_sentences / factual_sentences if factual_sentences else 0.0

    return CitationReport(
        citations=citations,
        valid_markers=sorted(set(valid)),
        invalid_markers=sorted(set(invalid)),
        coverage=round(coverage, 4),
        cited_sentences=cited_sentences,
        factual_sentences=factual_sentences,
        uncited_sentences=uncited,
    )


def strip_invalid_markers(answer: str, valid_markers: set[int]) -> str:
    """Remove markers that point at nothing so the UI never renders a dead link."""

    def _replace(match: re.Match) -> str:
        return match.group(0) if int(match.group(1)) in valid_markers else ""

    return CITATION_MARKER.sub(_replace, answer or "")


__all__ = ["Citation", "CitationReport", "build_citations", "analyze_citations", "strip_invalid_markers"]
