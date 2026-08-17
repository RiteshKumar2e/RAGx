import { STRATEGIES } from './constants';

/** Metadata for a strategy key, with a safe fallback for unknown names. */
export function strategyMeta(name) {
  return (
    STRATEGIES[name] || {
      label: name,
      short: name,
      color: '#64748b',
      description: '',
      bestFor: '',
      cost: 'unknown',
    }
  );
}

export function strategyLabel(name) {
  return strategyMeta(name).label;
}

export function strategyColor(name) {
  return strategyMeta(name).color;
}

/**
 * Order strategies for display: the pipeline stage order, not alphabetical.
 * Agentic leads (it plans), then retrievers, then corrective (it repairs).
 */
const DISPLAY_ORDER = [
  'agentic',
  'adaptive',
  'graph',
  'multimodal',
  'hybrid',
  'hyde',
  'naive',
  'corrective',
];

export function orderStrategies(names = []) {
  return [...new Set(names)].sort((a, b) => {
    const ai = DISPLAY_ORDER.indexOf(a);
    const bi = DISPLAY_ORDER.indexOf(b);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });
}

/**
 * Turn the router's `rules_fired` into the short bullet list the
 * "Why these strategies?" popover shows.
 */
export function routingHighlights(analysis, routing) {
  if (!analysis) return [];
  const rows = [
    { label: 'Query complexity', value: analysis.complexity },
    { label: 'Intent', value: analysis.intent?.replace(/_/g, ' ') },
    { label: 'Multi-hop', value: analysis.multi_hop ? 'Yes' : 'No' },
    { label: 'Keyword search', value: analysis.keyword_requirement >= 0.5 ? 'Required' : 'Not required' },
    { label: 'Semantic search', value: analysis.semantic_requirement >= 0.6 ? 'Required' : 'Not required' },
  ];
  if (analysis.requires_visual) rows.push({ label: 'Visual evidence', value: 'Required' });
  if (analysis.requires_tabular) rows.push({ label: 'Tabular evidence', value: 'Required' });
  if (analysis.relationship_query) rows.push({ label: 'Relationship traversal', value: 'Required' });
  if (analysis.cross_document) rows.push({ label: 'Cross-document', value: 'Yes' });
  rows.push({
    label: 'Evidence verification',
    value: routing?.use_corrective || analysis.requires_verification ? 'Required' : 'Standard',
  });
  return rows;
}

/** Human-readable relative cost of a routing decision. */
export function routingCostLabel(routing) {
  if (!routing) return 'unknown';
  if (routing.use_agentic) return 'high';
  const count = routing.strategies?.length || 1;
  if (count >= 3) return 'medium-high';
  if (count === 2) return 'medium';
  return 'low';
}
