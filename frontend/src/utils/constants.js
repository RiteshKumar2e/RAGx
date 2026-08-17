/**
 * Shared display metadata.
 *
 * Strategy colours are also the chart palette, so a strategy keeps the same
 * colour across strategy chips, the routing diagram and the evaluation charts.
 */

export const STRATEGIES = {
  naive: {
    label: 'Naive RAG',
    short: 'Naive',
    color: '#64748b',
    description: 'Single-shot dense vector search over chunk embeddings.',
    bestFor: 'Direct single-fact lookups.',
    cost: 'lowest',
  },
  hybrid: {
    label: 'Hybrid RAG',
    short: 'Hybrid',
    color: '#3567f0',
    description: 'Dense vector search fused with BM25 keyword search via reciprocal rank fusion.',
    bestFor: 'Exact terminology, model names, identifiers and version strings.',
    cost: 'low',
  },
  hyde: {
    label: 'HyDE',
    short: 'HyDE',
    color: '#8b5cf6',
    description: 'Generates a hypothetical answer passage, embeds it, then retrieves real evidence near it.',
    bestFor: 'Conceptual questions whose wording differs from the source text.',
    cost: 'medium',
  },
  multimodal: {
    label: 'Multimodal RAG',
    short: 'Multimodal',
    color: '#0ea5e9',
    description: 'Retrieves figures, charts, tables and OCR content, and loads the images themselves.',
    bestFor: 'Questions about a figure, chart, diagram or table.',
    cost: 'medium',
  },
  corrective: {
    label: 'Corrective RAG',
    short: 'Corrective',
    color: '#f59e0b',
    description: 'Grades retrieval quality and, when it is poor, rewrites the query and retrieves again.',
    bestFor: 'Anything where a wrong answer would be costly.',
    cost: 'medium',
  },
  graph: {
    label: 'Graph RAG',
    short: 'Graph',
    color: '#10b981',
    description: 'Traverses the entity knowledge graph and returns the chunks behind each relation.',
    bestFor: 'Relationship and multi-hop questions no single passage answers.',
    cost: 'medium',
  },
  adaptive: {
    label: 'Adaptive RAG',
    short: 'Adaptive',
    color: '#ec4899',
    description: 'Analyses the query and selects the minimum set of strategies that can answer it.',
    bestFor: 'The default entry point for every query.',
    cost: 'varies',
  },
  agentic: {
    label: 'Agentic RAG',
    short: 'Agentic',
    color: '#ef4444',
    description: 'Plans multi-step retrieval, calls other strategies as tools, and reflects on sufficiency.',
    bestFor: 'Complex research questions that must be decomposed.',
    cost: 'highest',
  },
  ragx: {
    label: 'RAGX (full)',
    short: 'RAGX',
    color: '#111a27',
    description: 'Adaptive routing plus corrective retrieval and evidence verification.',
    bestFor: 'The complete system.',
    cost: 'varies',
  },
};

export const STRATEGY_ORDER = [
  'naive',
  'hybrid',
  'hyde',
  'multimodal',
  'corrective',
  'graph',
  'adaptive',
  'agentic',
  'ragx',
];

export const INTENT_LABELS = {
  factual_lookup: 'Factual lookup',
  definition: 'Definition',
  comparison: 'Comparison',
  relationship: 'Relationship',
  multi_hop: 'Multi-hop',
  summarization: 'Summarisation',
  analysis: 'Analysis',
  visual: 'Visual',
  procedural: 'Procedural',
  exploratory: 'Exploratory',
};

export const COMPLEXITY_STYLES = {
  simple: { label: 'Simple', className: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20' },
  moderate: { label: 'Moderate', className: 'bg-amber-50 text-amber-700 ring-amber-600/20' },
  complex: { label: 'Complex', className: 'bg-rose-50 text-rose-700 ring-rose-600/20' },
};

export const CONFIDENCE_STYLES = {
  high: { label: 'High confidence', className: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20', bar: 'bg-emerald-500' },
  medium: { label: 'Medium confidence', className: 'bg-amber-50 text-amber-700 ring-amber-600/20', bar: 'bg-amber-500' },
  low: { label: 'Low confidence', className: 'bg-rose-50 text-rose-700 ring-rose-600/20', bar: 'bg-rose-500' },
  abstained: { label: 'Abstained', className: 'bg-ink-100 text-ink-700 ring-ink-600/20', bar: 'bg-ink-400' },
};

export const DOCUMENT_STATUS = {
  uploaded: { label: 'Queued', className: 'bg-ink-100 text-ink-700', active: true },
  parsing: { label: 'Extracting', className: 'bg-brand-50 text-brand-700', active: true },
  chunking: { label: 'Chunking', className: 'bg-brand-50 text-brand-700', active: true },
  embedding: { label: 'Embedding', className: 'bg-brand-50 text-brand-700', active: true },
  graph_indexing: { label: 'Graph indexing', className: 'bg-violet-50 text-violet-700', active: true },
  ready: { label: 'Ready', className: 'bg-emerald-50 text-emerald-700', active: false },
  failed: { label: 'Failed', className: 'bg-rose-50 text-rose-700', active: false },
};

/** The ingestion pipeline stages shown as a progress list in the Knowledge Base. */
export const PROCESSING_STEPS = [
  { key: 'upload', label: 'Uploading' },
  { key: 'parse', label: 'Extracting text, tables and figures' },
  { key: 'chunk', label: 'Structure-aware chunking' },
  { key: 'embed', label: 'Embedding and vector indexing' },
  { key: 'graph', label: 'Entity and relation extraction' },
  { key: 'ready', label: 'Ready' },
];

export const MODALITY_STYLES = {
  text: { label: 'Text', className: 'bg-ink-100 text-ink-700' },
  table: { label: 'Table', className: 'bg-sky-50 text-sky-700' },
  figure: { label: 'Figure', className: 'bg-violet-50 text-violet-700' },
  image: { label: 'Image', className: 'bg-violet-50 text-violet-700' },
  ocr: { label: 'OCR', className: 'bg-amber-50 text-amber-700' },
  code: { label: 'Code', className: 'bg-emerald-50 text-emerald-700' },
};

export const ENTITY_TYPE_COLORS = {
  METHOD: '#3567f0',
  MODEL: '#8b5cf6',
  DATASET: '#10b981',
  METRIC: '#f59e0b',
  TASK: '#0ea5e9',
  ORGANIZATION: '#ec4899',
  PERSON: '#ef4444',
  TOOL: '#14b8a6',
  CONCEPT: '#64748b',
  ARCHITECTURE: '#6366f1',
  FRAMEWORK: '#a855f7',
};

/** Chart palette — colour-blind-safe ordering, distinct in both hue and value. */
export const CHART_COLORS = [
  '#3567f0',
  '#10b981',
  '#f59e0b',
  '#8b5cf6',
  '#ef4444',
  '#0ea5e9',
  '#ec4899',
  '#64748b',
];

/** Metrics shown in the evaluation comparison, with direction and formatting. */
export const EVAL_METRICS = [
  { key: 'recall_at_k', label: 'Recall@K', group: 'Retrieval', higherIsBetter: true, format: 'ratio' },
  { key: 'precision_at_k', label: 'Precision@K', group: 'Retrieval', higherIsBetter: true, format: 'ratio' },
  { key: 'mrr', label: 'MRR', group: 'Retrieval', higherIsBetter: true, format: 'ratio' },
  { key: 'ndcg_at_k', label: 'nDCG@K', group: 'Retrieval', higherIsBetter: true, format: 'ratio' },
  { key: 'context_relevance', label: 'Context relevance', group: 'Retrieval', higherIsBetter: true, format: 'ratio' },
  { key: 'faithfulness', label: 'Faithfulness', group: 'Generation', higherIsBetter: true, format: 'ratio' },
  { key: 'answer_relevance', label: 'Answer relevance', group: 'Generation', higherIsBetter: true, format: 'ratio' },
  { key: 'groundedness', label: 'Groundedness', group: 'Generation', higherIsBetter: true, format: 'ratio' },
  { key: 'citation_accuracy', label: 'Citation accuracy', group: 'Generation', higherIsBetter: true, format: 'ratio' },
  { key: 'avg_latency_ms', label: 'Avg latency', group: 'System', higherIsBetter: false, format: 'ms' },
  { key: 'p95_latency_ms', label: 'P95 latency', group: 'System', higherIsBetter: false, format: 'ms' },
  { key: 'avg_total_tokens', label: 'Avg tokens', group: 'System', higherIsBetter: false, format: 'number' },
  { key: 'estimated_cost_usd', label: 'Est. cost', group: 'System', higherIsBetter: false, format: 'usd' },
  { key: 'avg_retrieval_calls', label: 'Retrieval calls', group: 'System', higherIsBetter: false, format: 'decimal' },
  { key: 'abstention_rate', label: 'Abstention rate', group: 'System', higherIsBetter: null, format: 'percent' },
];

export const BENCHMARK_CATEGORIES = {
  simple: 'Simple lookups',
  keyword: 'Keyword / exact terms',
  semantic: 'Semantic / conceptual',
  multi_hop: 'Multi-hop',
  relationship: 'Relationship',
  cross_document: 'Cross-document',
  multimodal: 'Multimodal',
  poor_retrieval: 'Adversarial (expect abstention)',
  complex_research: 'Complex research',
};
