"""Query, answer, evidence and explainability schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

STRATEGY_NAMES = Literal[
    "naive", "hybrid", "hyde", "multimodal", "corrective", "graph", "adaptive", "agentic"
]


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    conversation_id: str | None = None
    document_ids: list[str] | None = None
    strategies: list[STRATEGY_NAMES] | None = Field(
        default=None,
        description="Pin specific strategies and bypass the adaptive router. "
        "Used by the evaluation harness; leave unset for adaptive routing.",
    )
    top_k: int | None = Field(default=None, ge=1, le=30)
    rerank: bool | None = None
    include_evidence: bool = True
    include_trace: bool = True
    verify: bool = True
    stream: bool = False
    provider: Literal["gemini", "groq"] | None = None

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        cleaned = " ".join(v.split())
        if not cleaned:
            raise ValueError("The question cannot be empty.")
        return cleaned


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    conversation_id: str | None = None


class QueryAnalysisSchema(BaseModel):
    query: str = ""
    intent: str
    complexity: str
    semantic_requirement: float = 0.0
    keyword_requirement: float = 0.0
    multi_hop: bool = False
    requires_visual: bool = False
    requires_tabular: bool = False
    relationship_query: bool = False
    cross_document: bool = False
    expected_documents: int = 1
    requires_verification: bool = False
    entities: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    ambiguity: float = 0.0
    reasoning: str = ""
    source: str = "heuristic"
    llm_available: bool = False
    signals: dict[str, Any] = Field(default_factory=dict)


class RoutingRule(BaseModel):
    rule: str
    reason: str


class RoutingDecisionSchema(BaseModel):
    primary: str
    parallel: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    strategy_labels: list[str] = Field(default_factory=list)
    use_corrective: bool = False
    use_agentic: bool = False
    mode: str = "single"
    reason: str = ""
    rules_fired: list[RoutingRule] = Field(default_factory=list)
    estimated_llm_calls: int = 0
    config: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    analysis: QueryAnalysisSchema
    routing: RoutingDecisionSchema
    latency_ms: float = 0.0


class EvidenceItem(BaseModel):
    marker: int
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_path: list[str] = Field(default_factory=list)
    figure: str | None = None
    table: str | None = None
    modality: str = "text"
    asset_key: str | None = None
    relevance: float = 0.0
    content: str = ""
    excerpt: str = ""
    location: str = ""
    sources: list[str] = Field(default_factory=list)
    strategy_scores: dict[str, float] = Field(default_factory=dict)
    graph_path: str | None = None
    used_in_answer: bool = False


class ClaimVerdictSchema(BaseModel):
    claim: str
    claim_index: int = 0
    claim_type: str = "factual"
    cited: list[int] = Field(default_factory=list)
    verdict: str
    support_score: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    lexical_score: float = 0.0
    numeric_consistent: bool = True
    method: str = "lexical"


class ConfidenceSchema(BaseModel):
    score: float = 0.0
    label: str = "low"
    components: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    penalties: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)


class VerificationSchema(BaseModel):
    enabled: bool = True
    abstained: bool = False
    answer_modified: bool = False
    claims_total: int = 0
    claims_supported: int = 0
    claims_unsupported: int = 0
    claims_contradicted: int = 0
    claim_extraction_method: str = "none"
    claim_verdicts: list[ClaimVerdictSchema] = Field(default_factory=list)
    citations: dict[str, Any] = Field(default_factory=dict)
    confidence: ConfidenceSchema = Field(default_factory=ConfidenceSchema)
    latency_ms: float = 0.0
    notes: list[str] = Field(default_factory=list)


class GenerationInfo(BaseModel):
    provider: str | None = None
    model: str | None = None
    fallback_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    multimodal: bool = False
    images_sent: int = 0


class RetrievalSummary(BaseModel):
    strategies_used: list[str] = Field(default_factory=list)
    chunks_retrieved: int = 0
    documents_used: list[str] = Field(default_factory=list)
    document_names: list[str] = Field(default_factory=list)
    retrieval_calls: int = 0
    corrective_triggered: bool = False
    corrective_rounds: int = 0
    agentic_used: bool = False
    reranked: bool = False
    top_score: float = 0.0
    mean_score: float = 0.0
    latency_ms: float = 0.0
    notes: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class WhyThisAnswer(BaseModel):
    """The full explainability payload rendered by the frontend panel."""

    analysis: QueryAnalysisSchema
    routing: RoutingDecisionSchema
    retrieval: RetrievalSummary
    generation: GenerationInfo
    verification: VerificationSchema
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query_id: str
    trace_id: str
    conversation_id: str | None = None
    question: str
    answer: str
    abstained: bool = False
    confidence: float = 0.0
    confidence_label: str = "low"
    strategies: list[str] = Field(default_factory=list)
    strategy_labels: list[str] = Field(default_factory=list)
    routing_reason: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    why: WhyThisAnswer | None = None
    trace: dict[str, Any] | None = None
    total_latency_ms: float = 0.0
    created_at: datetime | None = None


class QueryHistoryItem(BaseModel):
    id: str
    question: str
    answer_preview: str = ""
    intent: str | None = None
    complexity: str | None = None
    strategies: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_label: str | None = None
    chunks_retrieved: int = 0
    corrective_triggered: bool = False
    agentic_used: bool = False
    insufficient_evidence: bool = False
    llm_provider: str | None = None
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    status: str = "completed"
    created_at: datetime


class QueryHistoryResponse(BaseModel):
    items: list[QueryHistoryItem]
    total: int
    page: int = 1
    page_size: int = 20


class ConversationSummary(BaseModel):
    id: str
    title: str
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class EvidenceDetailResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    document_title: str | None = None
    content: str
    modality: str
    page: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_path: list[str] = Field(default_factory=list)
    figure: str | None = None
    table: str | None = None
    ordinal: int = 0
    token_count: int = 0
    has_image: bool = False
    image_url: str | None = None
    neighbors: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "QueryRequest",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "QueryAnalysisSchema",
    "RoutingDecisionSchema",
    "RoutingRule",
    "EvidenceItem",
    "ClaimVerdictSchema",
    "ConfidenceSchema",
    "VerificationSchema",
    "GenerationInfo",
    "RetrievalSummary",
    "WhyThisAnswer",
    "QueryResponse",
    "QueryHistoryItem",
    "QueryHistoryResponse",
    "ConversationSummary",
    "EvidenceDetailResponse",
]
