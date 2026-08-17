"""Query Analyzer.

Produces the structured characterisation the Adaptive Router routes on. Two
sources are combined:

* **Heuristic analysis** (free, deterministic, always runs). Lexical and
  syntactic signals: interrogative form, comparison and relationship markers,
  multi-hop connectives, visual/tabular vocabulary, technical-identifier
  density, quoted phrases, query length.
* **LLM analysis** (one fast call via Groq, or Gemini as fallback). The model
  sees the query, the recent conversation and the knowledge-base document
  titles, and returns intent, complexity, sub-questions and named entities.

They are then **merged rather than one overriding the other**: the heuristic
provides the floor for signals it can prove (a query containing "Figure 3"
requires visual retrieval regardless of what the model says), while the LLM
supplies the judgements that need semantics (intent, decomposition, ambiguity).
The merge rule is recorded on the result so the routing explanation is honest
about where each signal came from.

When no LLM is configured the heuristic result is used alone and the system
still routes -- less precisely, and it says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.cache import analysis_cache, cache_key
from app.core.logging import TraceRecorder, get_logger
from app.core.text import (
    extract_quoted_phrases,
    technical_terms,
    tokenize,
)
from app.llm.base import Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import QUERY_ANALYSIS_SYSTEM, QUERY_ANALYSIS_USER

log = get_logger("ragx.adaptive.analyzer")


class Intent(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"
    MULTI_HOP = "multi_hop"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    VISUAL = "visual"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

    @property
    def rank(self) -> int:
        return {"simple": 0, "moderate": 1, "complex": 2}[self.value]


# -- lexical signal banks ---------------------------------------------------
# Visual detection is split by strength because a single ambiguous word is a
# common false-positive source: "quarterly revenue figures" is not a request for
# a figure, and "the graph of results" may just mean a table. Strong markers fire
# on their own; weak markers need corroboration from another visual signal.
STRONG_VISUAL_MARKERS = {
    "chart", "charts", "plot", "plots", "diagram", "diagrams", "screenshot",
    "heatmap", "scatter", "histogram", "bar graph", "line graph", "pie chart",
    "architecture diagram", "flowchart", "visualization", "visualisation",
}
WEAK_VISUAL_MARKERS = {
    "figure", "figures", "fig", "image", "images", "picture", "pictures", "graph",
    "visual", "axis", "axes", "curve", "curves", "legend", "illustrated", "depicted",
    "shown in", "pictured",
}
# "Figure 3", "fig. 2b" -- an explicit reference is always a visual requirement.
FIGURE_REFERENCE = re.compile(r"\b(?:fig(?:ure)?s?\.?\s*\d+[a-z]?)\b", re.IGNORECASE)
# Phrases where a visual word is being used non-visually.
VISUAL_FALSE_POSITIVES = re.compile(
    r"\b(?:revenue|sales|financial|profit|cost|budget|headline|key|the|these|those|those)\s+figures\b"
    r"|\bfigures?\s+(?:for|from)\s+\d{4}\b"
    r"|\bgraph\s+(?:database|store|traversal|rag|neural network)\b",
    re.IGNORECASE,
)

TABULAR_MARKERS = {
    "table", "tables", "row", "rows", "column", "columns", "cell", "spreadsheet",
    "csv", "tabulated", "dataset values", "numbers in",
}
RELATIONSHIP_MARKERS = {
    "relationship", "related", "relate", "relates", "connection", "connected", "connect",
    "between", "link", "linked", "depends", "dependency", "interaction", "influence",
    "affects", "impact", "association", "associated", "versus", "vs", "built on",
    "based on", "derived from", "uses", "used by", "cites", "extends",
}
COMPARISON_MARKERS = {
    "compare", "comparison", "versus", "vs", "difference", "differences", "differ",
    "better", "worse", "outperform", "outperforms", "faster", "slower", "trade-off",
    "tradeoff", "advantages", "disadvantages", "pros", "cons", "instead of",
}
MULTI_HOP_MARKERS = {
    "and then", "which in turn", "that also", "both", "all of", "each of", "across",
    "combined", "together with", "as well as", "in addition to", "followed by",
    "why does", "how does", "what causes", "leads to", "results in", "consequently",
}
SUMMARY_MARKERS = {
    "summarize", "summary", "summarise", "overview", "outline", "key points",
    "main findings", "tl;dr", "brief", "recap", "gist",
}
PROCEDURAL_MARKERS = {
    "how to", "how do i", "steps", "procedure", "process", "install", "configure",
    "setup", "set up", "implement", "workflow", "pipeline",
}
DEFINITION_MARKERS = {"what is", "what are", "define", "definition", "meaning of", "stands for"}
ANALYSIS_MARKERS = {
    "why", "analyze", "analyse", "evaluate", "assess", "critique", "implication",
    "implications", "significance", "reasoning", "justify", "explain why", "limitations",
}
CROSS_DOC_MARKERS = {
    "papers", "documents", "sources", "studies", "across", "each paper", "all papers",
    "every document", "both papers", "literature", "compared across",
}
VERIFICATION_MARKERS = {
    "exact", "exactly", "precise", "precisely", "cite", "citation", "source", "prove",
    "evidence", "verify", "confirm", "accurate", "correct", "actual", "official",
    "statistic", "percentage", "number",
}
# Nouns that mean the query wants a *value*, not a definition. "What is the mAP
# score?" is a lookup; "What is a feature pyramid network?" is a definition.
VALUE_MARKERS = {
    "score", "scores", "value", "values", "result", "results", "rate", "accuracy",
    "precision", "recall", "latency", "throughput", "size", "count", "how many",
    "how much", "number of", "percentage", "metric", "performance",
}

# A separate interrogative after a conjunction ("…, and what data…") signals a
# second, distinct question -- the clearest lexical marker of multi-hop intent.
INTERROGATIVE = re.compile(r"\b(what|which|how|why|when|where|who|whose)\b", re.IGNORECASE)
CHAINED_INTERROGATIVE = re.compile(
    r"[,;]?\s*\b(?:and|then|also|plus|as well as)\s+(what|which|how|why|when|where|who)\b",
    re.IGNORECASE,
)

NUMERIC = re.compile(r"\b\d+(?:\.\d+)?%?\b")
TABLE_REFERENCE = re.compile(r"\btable\s*\d+", re.IGNORECASE)


def detect_visual(text: str) -> tuple[bool, str]:
    """Decide whether a query genuinely needs visual retrieval.

    Returns ``(is_visual, why)``. The rules, in order:

    1. An explicit reference ("Figure 3") always wins.
    2. A phrase where a visual word is used non-visually ("revenue figures",
       "graph database") is excluded outright.
    3. A strong marker ("chart", "diagram") fires on its own.
    4. A weak marker ("figure", "image") needs a second visual signal --
       another weak marker, or a verb of showing/depicting.
    """
    lowered = text.lower()

    if FIGURE_REFERENCE.search(text):
        return True, "explicit figure reference"

    if VISUAL_FALSE_POSITIVES.search(text):
        return False, "visual word used in a non-visual sense"

    strong = [m for m in STRONG_VISUAL_MARKERS if m in lowered]
    if strong:
        return True, f"strong visual marker: {strong[0]}"

    weak = [m for m in WEAK_VISUAL_MARKERS if m in lowered]
    if len(weak) >= 2:
        return True, f"multiple visual markers: {', '.join(weak[:3])}"
    if weak and re.search(r"\b(show|shows|shown|display|displays|depict|depicts|illustrate|illustrates|look|see)\b", lowered):
        return True, f"visual marker '{weak[0]}' with a verb of showing"

    return False, "no visual requirement detected"


@dataclass
class QueryAnalysis:
    """The structured description the router acts on."""

    query: str = ""
    intent: Intent = Intent.FACTUAL_LOOKUP
    complexity: Complexity = Complexity.SIMPLE
    semantic_requirement: float = 0.5
    keyword_requirement: float = 0.5
    multi_hop: bool = False
    requires_visual: bool = False
    requires_tabular: bool = False
    relationship_query: bool = False
    cross_document: bool = False
    expected_documents: int = 1
    requires_verification: bool = False
    entities: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    ambiguity: float = 0.0
    reasoning: str = ""
    source: str = "heuristic"  # heuristic | llm | heuristic+llm
    signals: dict[str, Any] = field(default_factory=dict)
    llm_available: bool = False

    @property
    def modalities(self) -> list[str] | None:
        wanted: list[str] = []
        if self.requires_visual:
            wanted += ["figure", "image", "ocr"]
        if self.requires_tabular:
            wanted.append("table")
        return wanted or None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.value,
            "complexity": self.complexity.value,
            "semantic_requirement": round(self.semantic_requirement, 3),
            "keyword_requirement": round(self.keyword_requirement, 3),
            "multi_hop": self.multi_hop,
            "requires_visual": self.requires_visual,
            "requires_tabular": self.requires_tabular,
            "relationship_query": self.relationship_query,
            "cross_document": self.cross_document,
            "expected_documents": self.expected_documents,
            "requires_verification": self.requires_verification,
            "entities": self.entities,
            "key_terms": self.key_terms,
            "sub_questions": self.sub_questions,
            "ambiguity": round(self.ambiguity, 3),
            "reasoning": self.reasoning,
            "source": self.source,
            "llm_available": self.llm_available,
            "signals": self.signals,
        }


class QueryAnalyzer:
    def __init__(self) -> None:
        self.gateway = get_gateway()

    # -------------------------------------------------------------- heuristic
    def heuristic(self, query: str, history: list[dict[str, str]] | None = None) -> QueryAnalysis:
        lowered = query.lower()
        tokens = tokenize(query, drop_stopwords=True)
        all_tokens = tokenize(query, drop_stopwords=False)
        terms = technical_terms(query)
        quoted = extract_quoted_phrases(query)
        word_count = len(all_tokens)

        def has(markers: set[str]) -> bool:
            return any(m in lowered for m in markers)

        def count(markers: set[str]) -> int:
            return sum(1 for m in markers if m in lowered)

        visual, visual_reason = detect_visual(query)
        tabular = has(TABULAR_MARKERS) or bool(TABLE_REFERENCE.search(query))
        relationship = has(RELATIONSHIP_MARKERS)
        comparison = has(COMPARISON_MARKERS)
        multi_hop_markers = count(MULTI_HOP_MARKERS)
        summary = has(SUMMARY_MARKERS)
        procedural = has(PROCEDURAL_MARKERS)
        definition = has(DEFINITION_MARKERS)
        analysis_marker = has(ANALYSIS_MARKERS)
        cross_doc = has(CROSS_DOC_MARKERS)
        verification = has(VERIFICATION_MARKERS) or bool(NUMERIC.search(query))

        question_marks = query.count("?")
        conjunctions = len(re.findall(r"\b(and|or|as well as|plus|also)\b", lowered))
        interrogatives = len(INTERROGATIVE.findall(query))
        chained_interrogatives = len(CHAINED_INTERROGATIVE.findall(query))
        value_seeking = has(VALUE_MARKERS)

        # -- multi-hop ------------------------------------------------------
        # A chained interrogative ("…, and what data was it trained on?") is the
        # strongest lexical evidence of a second, dependent question.
        multi_hop = (
            chained_interrogatives >= 1
            or interrogatives >= 3
            or multi_hop_markers >= 2
            or question_marks > 1
            or (comparison and conjunctions >= 1 and word_count > 10)
            or (relationship and cross_doc)
            or (conjunctions >= 2 and word_count > 14)
        )

        # -- intent ---------------------------------------------------------
        if visual:
            intent = Intent.VISUAL
        elif multi_hop and (chained_interrogatives or interrogatives >= 3):
            intent = Intent.MULTI_HOP
        elif summary:
            intent = Intent.SUMMARIZATION
        elif comparison:
            intent = Intent.COMPARISON
        elif relationship:
            intent = Intent.RELATIONSHIP
        elif procedural:
            intent = Intent.PROCEDURAL
        elif definition and not (value_seeking or terms):
            # "What is X?" is only a definition when X is not a metric or a
            # named identifier -- otherwise it is a value lookup.
            intent = Intent.DEFINITION
        elif analysis_marker:
            intent = Intent.ANALYSIS
        elif question_marks > 1 or multi_hop_markers >= 2:
            intent = Intent.MULTI_HOP
        else:
            intent = Intent.FACTUAL_LOOKUP

        # -- complexity -----------------------------------------------------
        complexity_score = 0
        complexity_score += 2 if multi_hop else 0
        complexity_score += 1 if interrogatives >= 3 else 0
        complexity_score += 1 if comparison else 0
        complexity_score += 1 if relationship else 0
        complexity_score += 1 if cross_doc else 0
        complexity_score += 1 if word_count > 18 else 0
        complexity_score += 1 if analysis_marker else 0
        complexity_score += 1 if summary and word_count > 10 else 0
        complexity_score -= 1 if word_count <= 6 and not multi_hop else 0

        if complexity_score >= 4:
            complexity = Complexity.COMPLEX
        elif complexity_score >= 2:
            complexity = Complexity.MODERATE
        else:
            complexity = Complexity.SIMPLE

        # -- retrieval requirements -----------------------------------------
        # Keyword pressure rises with the density of identifiers/acronyms and
        # with any quoted exact phrase.
        identifier_density = len(terms) / max(1, len(tokens)) if tokens else 0.0
        keyword_requirement = min(
            1.0,
            0.25
            + identifier_density * 1.5
            + (0.25 if quoted else 0.0)
            + (0.15 if NUMERIC.search(query) else 0.0),
        )

        # Semantic pressure rises with abstract/analytical phrasing and length,
        # and falls when the query is mostly identifiers.
        semantic_requirement = min(
            1.0,
            0.35
            + (0.2 if analysis_marker else 0.0)
            + (0.15 if summary else 0.0)
            + (0.15 if definition else 0.0)
            + (0.1 if word_count > 14 else 0.0)
            - identifier_density * 0.5,
        )
        semantic_requirement = max(0.1, semantic_requirement)

        expected_documents = 1
        if cross_doc or comparison:
            expected_documents = 3
        elif multi_hop:
            expected_documents = 2

        ambiguity = 0.0
        if word_count <= 3:
            ambiguity += 0.4
        if not terms and not quoted:
            ambiguity += 0.2
        if re.match(r"^(and|but|what about|how about|also)\b", lowered):
            ambiguity += 0.35  # follow-up fragment
        ambiguity = min(1.0, ambiguity)

        signals = {
            "word_count": word_count,
            "technical_terms": terms[:10],
            "quoted_phrases": quoted[:5],
            "identifier_density": round(identifier_density, 3),
            "question_marks": question_marks,
            "conjunctions": conjunctions,
            "interrogatives": interrogatives,
            "chained_interrogatives": chained_interrogatives,
            "visual_reason": visual_reason,
            "markers": {
                "visual": visual,
                "tabular": tabular,
                "relationship": relationship,
                "comparison": comparison,
                "multi_hop_markers": multi_hop_markers,
                "summary": summary,
                "procedural": procedural,
                "definition": definition,
                "analysis": analysis_marker,
                "cross_document": cross_doc,
                "verification": verification,
            },
            "complexity_score": complexity_score,
        }

        return QueryAnalysis(
            query=query,
            intent=intent,
            complexity=complexity,
            semantic_requirement=round(semantic_requirement, 3),
            keyword_requirement=round(keyword_requirement, 3),
            multi_hop=multi_hop,
            requires_visual=visual,
            requires_tabular=tabular,
            relationship_query=relationship or comparison,
            cross_document=cross_doc,
            expected_documents=expected_documents,
            requires_verification=verification or complexity is Complexity.COMPLEX,
            entities=terms[:10],
            key_terms=(quoted + terms)[:10],
            sub_questions=[],
            ambiguity=round(ambiguity, 3),
            reasoning=self._heuristic_reason(intent, complexity, signals),
            source="heuristic",
            signals=signals,
        )

    @staticmethod
    def _heuristic_reason(intent: Intent, complexity: Complexity, signals: dict) -> str:
        markers = [k for k, v in signals["markers"].items() if v is True]
        marker_text = ", ".join(markers) if markers else "no strong lexical markers"
        return (
            f"Lexical analysis: {intent.value} intent, {complexity.value} complexity "
            f"({marker_text}; {signals['word_count']} content words, "
            f"{len(signals['technical_terms'])} technical terms)."
        )

    # -------------------------------------------------------------------- LLM
    async def llm_analysis(
        self,
        query: str,
        history: list[dict[str, str]] | None,
        document_titles: list[str],
        trace: TraceRecorder | None = None,
    ) -> dict[str, Any] | None:
        if not self.gateway.any_configured:
            return None

        context_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')[:220]}"
            for turn in (history or [])[-4:]
        ) or "(no prior turns)"
        documents_text = "\n".join(f"- {t}" for t in document_titles[:25]) or "(knowledge base is empty)"

        try:
            payload, _ = await self.gateway.complete_json(
                [
                    Message.system(QUERY_ANALYSIS_SYSTEM),
                    Message.user(
                        QUERY_ANALYSIS_USER.format(
                            context=context_text, documents=documents_text, question=query
                        )
                    ),
                ],
                Purpose.QUERY_ANALYSIS,
                default=None,
                temperature=0.0,
                max_output_tokens=700,
                trace=trace,
            )
        except Exception as exc:
            log.warning("analyzer.llm_failed", error=str(exc)[:160])
            return None
        return payload if isinstance(payload, dict) else None

    # ------------------------------------------------------------------ merge
    def merge(self, heuristic: QueryAnalysis, payload: dict[str, Any]) -> QueryAnalysis:
        """Combine both analyses.

        Booleans use OR for signals the heuristic can *prove* from the query text
        (a literal "Figure 3" means visual retrieval is required no matter what
        the model returns). Continuous requirements take the max, because
        under-retrieving is more damaging than retrieving a little too broadly.
        Intent, sub-questions and entities come from the LLM, which has the
        semantics the heuristic lacks.
        """
        merged = QueryAnalysis(**{**heuristic.__dict__})
        merged.source = "heuristic+llm"
        merged.llm_available = True

        try:
            merged.intent = Intent(str(payload.get("intent", heuristic.intent.value)).lower())
        except ValueError:
            pass
        try:
            llm_complexity = Complexity(str(payload.get("complexity", heuristic.complexity.value)).lower())
            # Take the higher of the two: under-routing a complex query is the
            # expensive mistake.
            merged.complexity = (
                llm_complexity if llm_complexity.rank >= heuristic.complexity.rank else heuristic.complexity
            )
        except ValueError:
            pass

        merged.semantic_requirement = max(
            heuristic.semantic_requirement, _float(payload.get("semantic_requirement"), heuristic.semantic_requirement)
        )
        merged.keyword_requirement = max(
            heuristic.keyword_requirement, _float(payload.get("keyword_requirement"), heuristic.keyword_requirement)
        )
        merged.multi_hop = heuristic.multi_hop or bool(payload.get("multi_hop"))
        merged.requires_visual = heuristic.requires_visual or bool(payload.get("requires_visual"))
        merged.requires_tabular = heuristic.requires_tabular or bool(payload.get("requires_tabular"))
        merged.relationship_query = heuristic.relationship_query or bool(payload.get("relationship_query"))
        merged.cross_document = heuristic.cross_document or bool(payload.get("cross_document"))
        merged.requires_verification = heuristic.requires_verification or bool(
            payload.get("requires_verification")
        )
        merged.expected_documents = max(
            heuristic.expected_documents, _int(payload.get("expected_documents"), 1)
        )
        merged.ambiguity = max(heuristic.ambiguity, _float(payload.get("ambiguity"), 0.0))

        merged.entities = _merge_strings(heuristic.entities, payload.get("entities"), 12)
        merged.key_terms = _merge_strings(heuristic.key_terms, payload.get("key_terms"), 12)
        merged.sub_questions = [
            str(s).strip() for s in (payload.get("sub_questions") or []) if str(s).strip()
        ][:6]

        llm_reason = str(payload.get("reasoning", "")).strip()
        merged.reasoning = llm_reason or heuristic.reasoning
        merged.signals = {
            **heuristic.signals,
            "llm": {
                "intent": payload.get("intent"),
                "complexity": payload.get("complexity"),
                "multi_hop": payload.get("multi_hop"),
                "sub_questions": len(merged.sub_questions),
            },
            "heuristic_intent": heuristic.intent.value,
            "heuristic_complexity": heuristic.complexity.value,
        }
        return merged

    # ------------------------------------------------------------------ main
    async def analyze(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        document_titles: list[str] | None = None,
        trace: TraceRecorder | None = None,
        use_llm: bool = True,
    ) -> QueryAnalysis:
        heuristic = self.heuristic(query, history)

        if not use_llm or not self.gateway.any_configured:
            heuristic.llm_available = self.gateway.any_configured
            if not self.gateway.any_configured:
                heuristic.signals["note"] = (
                    "No cloud LLM is configured; routing used lexical analysis only."
                )
            return heuristic

        key = cache_key("query_analysis", query, [t.get("content", "")[:80] for t in (history or [])[-2:]])
        cached = await analysis_cache.get(key)
        if cached is not None:
            merged = self.merge(heuristic, cached)
            merged.signals["cache_hit"] = True
            return merged

        payload = await self.llm_analysis(query, history, document_titles or [], trace)
        if payload is None:
            heuristic.signals["note"] = "LLM query analysis was unavailable; used lexical analysis only."
            heuristic.llm_available = False
            return heuristic

        await analysis_cache.set(key, payload)
        return self.merge(heuristic, payload)


# -- coercion helpers -------------------------------------------------------
def _float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _merge_strings(base: list[str], extra: Any, limit: int) -> list[str]:
    out = list(base)
    for item in extra or []:
        text = str(item).strip()
        if text and text.lower() not in {o.lower() for o in out}:
            out.append(text)
    return out[:limit]


__all__ = ["QueryAnalyzer", "QueryAnalysis", "Intent", "Complexity"]
