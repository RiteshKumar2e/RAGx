import { get } from './api';

/** Knowledge-graph traversal and export for the graph explorer. */
export const graphService = {
  /** Nodes and edges ordered by degree, shaped for React Flow. */
  export({ limit = 250, documentId } = {}) {
    return get('/graph', { params: { limit, document_id: documentId || undefined } });
  },

  stats() {
    return get('/graph/stats');
  },

  searchEntities(query, limit = 20) {
    return get('/graph/search', { params: { q: query, limit } });
  },

  /** Sub-graph around one entity, expanded to `depth` hops. */
  neighborhood(entity, { depth = 2, limit = 60 } = {}) {
    return get('/graph/neighborhood', { params: { entity, depth, limit } });
  },

  /** The traversal that answers multi-hop relationship questions. */
  paths(source, target, maxDepth = 4) {
    return get('/graph/paths', { params: { source, target, max_depth: maxDepth } });
  },

  documentsForEntity(entity) {
    return get(`/graph/entity/${encodeURIComponent(entity)}/documents`);
  },
};

export default graphService;
