import { del, get, post, resourceUrl } from './api';

/** Query execution, routing inspection, history and evidence drill-down. */
export const queryService = {
  /** Run the full RAGX pipeline. Leave `strategies` unset for adaptive routing. */
  ask({
    question,
    conversationId,
    documentIds,
    strategies,
    topK,
    rerank,
    verify = true,
    includeTrace = true,
  }) {
    return post('/query', {
      question,
      conversation_id: conversationId || undefined,
      document_ids: documentIds?.length ? documentIds : undefined,
      strategies: strategies?.length ? strategies : undefined,
      top_k: topK || undefined,
      rerank: typeof rerank === 'boolean' ? rerank : undefined,
      verify,
      include_evidence: true,
      include_trace: includeTrace,
    });
  },

  /** Analysis + routing decision only — no retrieval, no generation. */
  analyze(question, conversationId) {
    return post('/query/analyze', { question, conversation_id: conversationId || undefined });
  },

  /**
   * Streamed answer over Server-Sent Events.
   *
   * Uses `fetch` rather than EventSource because the request is a POST with a
   * JSON body. Handlers: onStatus, onToken, onDone, onError.
   *
   * The backend verifies the answer *after* generation, so the `done` payload is
   * authoritative and may replace streamed text with an abstention.
   */
  async stream(payload, { onStatus, onToken, onDone, onError, signal } = {}) {
    const base = import.meta.env.VITE_API_BASE_URL || '/api/v1';
    const headers = { 'Content-Type': 'application/json' };
    if (import.meta.env.VITE_RAGX_API_KEY) {
      headers['X-Ragx-Key'] = import.meta.env.VITE_RAGX_API_KEY;
    }

    let response;
    try {
      response = await fetch(`${base}/query/stream`, {
        method: 'POST',
        headers,
        signal,
        body: JSON.stringify({
          question: payload.question,
          conversation_id: payload.conversationId || undefined,
          document_ids: payload.documentIds?.length ? payload.documentIds : undefined,
          strategies: payload.strategies?.length ? payload.strategies : undefined,
          top_k: payload.topK || undefined,
          verify: payload.verify !== false,
          include_evidence: true,
          include_trace: true,
          stream: true,
        }),
      });
    } catch {
      onError?.({ message: 'Cannot reach the RAGX backend. Check that it is running.' });
      return;
    }

    if (!response.ok || !response.body) {
      onError?.({ message: `The query failed (HTTP ${response.status}).` });
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const dispatch = (rawEvent) => {
      const lines = rawEvent.split('\n');
      const name = lines.find((l) => l.startsWith('event:'))?.slice(6).trim();
      const data = lines
        .filter((l) => l.startsWith('data:'))
        .map((l) => l.slice(5).trim())
        .join('');
      if (!data) return;

      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch {
        return;
      }

      if (name === 'status') onStatus?.(parsed);
      else if (name === 'token') onToken?.(parsed.text || '');
      else if (name === 'done') onDone?.(parsed);
      else if (name === 'error') onError?.(parsed);
    };

    try {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() ?? '';
        events.forEach(dispatch);
      }
      if (buffer.trim()) dispatch(buffer);
    } catch (error) {
      if (error?.name !== 'AbortError') {
        onError?.({ message: 'The answer stream was interrupted.' });
      }
    }
  },

  history({ page = 1, pageSize = 20, conversationId } = {}) {
    return get('/query/history', {
      params: { page, page_size: pageSize, conversation_id: conversationId || undefined },
    });
  },

  byId(queryId) {
    return get(`/query/${queryId}`);
  },

  conversations(limit = 30) {
    return get('/query/conversations', { params: { limit } });
  },

  deleteConversation(conversationId) {
    return del(`/query/conversations/${conversationId}`);
  },

  /** Full passage behind a citation, plus neighbouring chunks. */
  evidence(chunkId) {
    return get(`/evidence/${chunkId}`);
  },

  evidenceImageUrl(chunkId) {
    return resourceUrl(`/evidence/${chunkId}/image`);
  },
};

export default queryService;
