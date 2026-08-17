# RAGX — API Reference

Base URL: `http://localhost:8000/api/v1`
Interactive docs: [`/docs`](http://localhost:8000/docs) (Swagger) · [`/redoc`](http://localhost:8000/redoc)

---

## Conventions

**Authentication.** Optional. When `RAGX_API_KEY` is set on the backend,
mutating endpoints (upload, delete, reindex, evaluation runs, settings) require:

```
X-Ragx-Key: <your-key>
```

This gates RAGX's own API. It is **not** a model-provider key and grants no
access to any LLM.

**Errors.** Every failure returns the same envelope. Stack traces are logged
server-side and never returned.

```json
{
  "error": {
    "code": "not_found",
    "message": "Document 'abc123' was not found.",
    "detail": null
  }
}
```

| Status | Codes |
|---|---|
| 401 | `unauthorized` |
| 404 | `not_found` |
| 413 | `file_too_large` |
| 415 | `unsupported_file` |
| 422 | `validation_error`, `ingestion_failed` |
| 502 | `provider_error` |
| 503 | `provider_not_configured`, `storage_error` |
| 500 | `internal_error` |

**Response headers.** `X-Request-ID` and `X-Process-Time-Ms` on every response.

---

## Query

### `POST /query`

Run the full RAGX pipeline.

```jsonc
{
  "question": "What mAP does DefectNet reach on NEU-DET?",
  "conversation_id": null,          // continue a thread
  "document_ids": null,             // restrict retrieval to these documents
  "strategies": null,               // null ⇒ adaptive routing (recommended)
  "top_k": null,                    // null ⇒ router decides
  "rerank": null,
  "verify": true,                   // false disables verification (grounding guarantees drop)
  "include_evidence": true,
  "include_trace": true,
  "provider": null                  // "gemini" | "groq" — override provider selection
}
```

`strategies` accepts any of `naive`, `hybrid`, `hyde`, `multimodal`,
`corrective`, `graph`, `adaptive`, `agentic`. Setting it **bypasses the router** —
this is what the evaluation harness does to benchmark a single strategy.

**Response**

```jsonc
{
  "query_id": "…", "trace_id": "…", "conversation_id": "…",
  "question": "…",
  "answer": "DefectNet reaches 78.4 mAP on NEU-DET [1], outperforming YOLOv5s [2].",
  "abstained": false,
  "confidence": 0.81,
  "confidence_label": "high",       // high | medium | low | abstained
  "strategies": ["hybrid"],
  "strategy_labels": ["Hybrid RAG"],
  "routing_reason": "Selected Hybrid RAG because the query requires exact terminology matching.",

  "evidence": [{
    "marker": 1,
    "chunk_id": "…", "document_id": "…", "document_name": "paper.pdf",
    "page": 7, "page_end": 7, "section": "Results",
    "figure": null, "table": "Table 2", "modality": "table",
    "relevance": 0.93,
    "content": "…", "excerpt": "…", "location": "p.7 · Results · Table 2",
    "sources": ["dense", "bm25", "hybrid"],
    "strategy_scores": { "dense": 0.88, "bm25": 1.0 },
    "graph_path": null,
    "used_in_answer": true
  }],

  "citations": [ /* one per evidence block */ ],
  "why": { /* see below */ },
  "trace": { /* see below */ },
  "total_latency_ms": 2841.7,
  "created_at": "2026-08-17T05:46:49Z"
}
```

#### The `why` object

Everything the "Why this answer?" panel renders. All of it was recorded during
the run.

```jsonc
{
  "analysis": {
    "intent": "factual_lookup", "complexity": "simple",
    "semantic_requirement": 0.35, "keyword_requirement": 0.72,
    "multi_hop": false, "requires_visual": false, "requires_tabular": false,
    "relationship_query": false, "cross_document": false,
    "expected_documents": 1, "requires_verification": true,
    "entities": ["DefectNet", "NEU-DET"], "key_terms": [...],
    "sub_questions": [], "ambiguity": 0.0,
    "reasoning": "…", "source": "heuristic+llm", "signals": { … }
  },
  "routing": {
    "primary": "hybrid", "parallel": [],
    "strategies": ["hybrid"], "strategy_labels": ["Hybrid RAG"],
    "use_corrective": false, "use_agentic": false, "mode": "single",
    "reason": "…",
    "rules_fired": [{ "rule": "keyword_dominant", "reason": "…" }],
    "estimated_llm_calls": 5,
    "config": { "top_k": 8, "candidate_pool": 40, "dense_weight": 0.36, "sparse_weight": 0.64, … }
  },
  "retrieval": {
    "strategies_used": ["hybrid"], "chunks_retrieved": 8,
    "documents_used": ["…"], "retrieval_calls": 2,
    "corrective_triggered": false, "corrective_rounds": 0,
    "agentic_used": false, "reranked": true,
    "top_score": 0.93, "mean_score": 0.51, "latency_ms": 412.8,
    "notes": [], "diagnostics": { "per_strategy": { … }, "corrective": { … }, "agentic": { … } }
  },
  "generation": {
    "provider": "gemini", "model": "gemini-2.0-flash", "fallback_used": false,
    "prompt_tokens": 1842, "completion_tokens": 96, "total_tokens": 1938,
    "estimated_cost_usd": 0.000222, "latency_ms": 1204.3,
    "multimodal": false, "images_sent": 0
  },
  "verification": {
    "enabled": true, "abstained": false, "answer_modified": false,
    "claims_total": 3, "claims_supported": 3,
    "claims_unsupported": 0, "claims_contradicted": 0,
    "claim_extraction_method": "llm",
    "claim_verdicts": [{
      "claim": "…", "verdict": "supported", "support_score": 0.91,
      "evidence_ids": ["…"], "numeric_consistent": true, "method": "lexical+llm"
    }],
    "citations": {
      "coverage": 1.0, "citation_accuracy": 1.0,
      "hallucinated_citations": 0, "cited_sentences": 3, "factual_sentences": 3
    },
    "confidence": {
      "score": 0.81, "label": "high",
      "components": { "claim_support": 0.91, "retrieval_quality": 0.78, … },
      "weights": { … }, "penalties": {}, "rationale": ["…"]
    }
  },
  "stage_latency_ms": { "query_analysis": 310.2, "retrieval": 412.8, … }
}
```

#### The `trace` object

```jsonc
{
  "trace_id": "…", "total_latency_ms": 2841.7,
  "retrieval_calls": 2, "corrective_rounds": 0, "corrective_events": [],
  "llm_calls": [{
    "provider": "groq", "model": "llama-3.1-8b-instant",
    "purpose": "query_analysis", "latency_ms": 289.1,
    "prompt_tokens": 412, "completion_tokens": 118, "total_tokens": 530,
    "cost_usd": 0.000335, "fallback_used": false, "error": null
  }],
  "prompt_tokens": 2254, "completion_tokens": 214, "total_tokens": 2468,
  "estimated_cost_usd": 0.000557,
  "stage_latency_ms": { … },
  "events": [{ "name": "retrieval:hybrid", "category": "retrieval", "duration_ms": 412.8, "detail": { … } }]
}
```

---

### `POST /query/analyze`

Query analysis and the routing decision **without retrieving or generating**.
Costs one fast LLM call. The cheapest way to inspect routing behaviour.

```bash
curl -X POST localhost:8000/api/v1/query/analyze \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does DefectNet relate to MobileNetV2?"}'
```

```jsonc
{ "analysis": { … }, "routing": { … }, "latency_ms": 284.1 }
```

---

### `POST /query/stream`

Same payload as `POST /query`; responds with Server-Sent Events.

| Event | Payload |
|---|---|
| `status` | `{stage, message, strategies?, routing_reason?}` |
| `token` | `{text}` — incremental answer text |
| `done` | the complete `QueryResponse` |
| `error` | `{message, code}` |

> Verification runs on the *completed* answer, so the `done` payload is
> authoritative and may replace streamed text with an abstention. Clients must
> render the `done` answer as final.

---

### Query history

| Endpoint | Purpose |
|---|---|
| `GET /query/history?page=1&page_size=20&conversation_id=` | Paginated history |
| `GET /query/{query_id}` | Re-read a stored answer with its full `why` |
| `GET /query/conversations?limit=30` | List threads |
| `DELETE /query/conversations/{id}` | Delete a thread and its queries |

---

## Evidence

### `GET /evidence/{chunk_id}`

The full passage behind a citation, plus neighbouring chunks for context.

```jsonc
{
  "chunk_id": "…", "document_id": "…", "document_name": "paper.pdf",
  "content": "…full passage…", "modality": "text",
  "page": 7, "section": "Results", "section_path": ["Results", "Ablation"],
  "figure": null, "table": "Table 2",
  "ordinal": 14, "token_count": 312,
  "has_image": false, "image_url": null,
  "neighbors": [{ "chunk_id": "…", "ordinal": 13, "content": "…", "page": 7 }]
}
```

### `GET /evidence/{chunk_id}/image`

The figure or table image behind a citation. Returns image bytes, or 404 when the
chunk has no associated image.

---

## Documents

### `POST /documents/upload`

`multipart/form-data`, field `files` (repeatable).

Each file is validated by extension allow-list, declared MIME type, magic bytes
and size. Files that fail are reported in `rejected`; valid files still proceed.

```jsonc
{
  "uploaded": [{
    "document_id": "…", "filename": "paper.pdf", "status": "uploaded",
    "message": "…", "duplicate_of": null      // set when the checksum already exists
  }],
  "rejected": [{ "filename": "notes.exe", "reason": "'.exe' is not an accepted file type." }]
}
```

Processing runs in the background — poll `GET /documents/{id}` for status.

### `POST /documents/ingest-url`

```jsonc
{ "url": "https://example.com/paper", "title": "optional" }
```

Only `text/html` and plain-text responses are accepted, subject to the same size
limit as uploads.

### `GET /documents`

Query: `page`, `page_size`, `status`, `file_type`, `search`.

```jsonc
{
  "items": [{
    "id": "…", "filename": "paper.pdf", "title": "…", "file_type": ".pdf",
    "size_bytes": 284913, "status": "ready",
    "page_count": 12, "chunk_count": 47, "table_count": 3,
    "figure_count": 5, "entity_count": 28, "token_count": 18420,
    "processing_steps": {
      "upload":  {"status": "completed", "duration_ms": 0.0,   "detail": "284,913 bytes"},
      "parse":   {"status": "completed", "duration_ms": 1284.2,"detail": "112 blocks · 12 pages · 3 tables · 5 figures"},
      "chunk":   {"status": "completed", "duration_ms": 42.1,  "detail": "47 chunks"},
      "embed":   {"status": "completed", "duration_ms": 2104.7,"detail": "47 vectors · gemini · dim 768"},
      "graph":   {"status": "completed", "duration_ms": 8421.3,"detail": "28 entities · 34 relations · networkx"},
      "ready":   {"status": "completed", "duration_ms": 11852.3,"detail": "47 chunks indexed"}
    },
    "created_at": "…", "indexed_at": "…"
  }],
  "total": 12, "page": 1, "page_size": 20,
  "status_counts": { "ready": 11, "failed": 1 }
}
```

Statuses: `uploaded` → `parsing` → `chunking` → `embedding` → `graph_indexing`
→ `ready`, or `failed`.

### Other document endpoints

| Endpoint | Purpose |
|---|---|
| `GET /documents/{id}?chunk_limit=50` | Detail: chunks, entities, outline, metadata, modality breakdown |
| `GET /documents/{id}/file` | Download the original file |
| `GET /documents/stats` | Knowledge-base statistics |
| `POST /documents/{id}/reindex` | Re-run the ingestion pipeline |
| `DELETE /documents/{id}` | Remove from **all** indexes (vector, BM25, graph, object storage) |
| `POST /documents/rebuild-indexes` | Rebuild BM25 from the database |

---

## Graph

| Endpoint | Purpose |
|---|---|
| `GET /graph?limit=250&document_id=` | Nodes + edges, ordered by degree (React Flow shaped) |
| `GET /graph/stats` | Entity/relation counts, type distributions, hub entities |
| `GET /graph/search?q=&limit=20` | Entity search |
| `GET /graph/neighborhood?entity=&depth=2&limit=60` | Sub-graph around one entity |
| `GET /graph/paths?source=&target=&max_depth=4` | **Relationship paths between two entities** |
| `GET /graph/entity/{entity}/documents` | Documents an entity appears in |

`GET /graph/paths` exposes the traversal Graph RAG runs internally:

```jsonc
{
  "source": "DefectNet", "target": "NEU-DET", "found": true,
  "paths": [{
    "entities": ["DefectNet", "NEU-DET"],
    "relations": [{ "type": "EVALUATED_ON", "confidence": 0.92, "chunk_id": "…", "context": "…" }],
    "score": 0.92,
    "description": "DefectNet -[EVALUATED_ON]-> NEU-DET",
    "chunk_ids": ["…"]
  }]
}
```

---

## Evaluation

### `POST /evaluation/run`

```jsonc
{
  "strategies": ["naive", "hybrid", "adaptive", "ragx"],
  "dataset": "ragx_benchmark",
  "categories": null,          // null ⇒ all
  "limit": 10,                 // questions per run
  "k": 8,
  "judge_generation": true,    // LLM judges + pooled relevance labels
  "name": "routing-ablation-v1",
  "notes": null
}
```

Returns immediately; runs execute in the background.

```jsonc
{
  "run_ids": ["…"], "strategies": ["…"], "question_count": 10,
  "message": "Started 4 evaluation run(s) over 10 questions.",
  "warnings": ["This dataset ships no manual relevance labels, so Recall@K … pooled LLM judgements."]
}
```

> Returns **422** when no documents are indexed — every condition would score
> zero and the comparison would be meaningless.

### `GET /evaluation/comparison`

Latest completed run per strategy, side by side.

```jsonc
{
  "runs": [{ "strategy": "ragx", "recall_at_k": 0.82, "faithfulness": 0.94, "avg_latency_ms": 2841.7, … }],
  "metrics": ["recall_at_k", "precision_at_k", … ],
  "best_by_metric": { "recall_at_k": "ragx", "avg_latency_ms": "naive" },
  "has_data": true,
  "message": ""
}
```

When nothing has been run: `has_data: false`, `runs: []`, and an explanatory
message. **Never placeholder numbers.** A metric that could not be computed is
`null`, which the UI renders as `—` — visibly different from `0`.

### Other evaluation endpoints

| Endpoint | Purpose |
|---|---|
| `GET /evaluation/benchmark?dataset=` | Questions grouped by category |
| `GET /evaluation/datasets` | Available datasets |
| `GET /evaluation/runs?limit=50&strategy=` | Run history with progress |
| `GET /evaluation/runs/{id}` | Run detail with per-question results |
| `GET /evaluation/runs/{id}/results` | Per-question results only |
| `DELETE /evaluation/runs/{id}` | Delete a run |

---

## System

### `GET /health?probe=false`

`probe=true` makes a live request to each configured provider (costs a few
tokens); the default is a free configuration-only check.

```jsonc
{
  "status": "healthy",             // healthy | degraded | unhealthy
  "version": "1.0.0", "environment": "development",
  "components": [
    { "name": "database",       "status": "healthy", "healthy": true,  "detail": {…} },
    { "name": "vector_store",   "status": "healthy", "healthy": true,  "detail": {…} },
    { "name": "graph_store",    "status": "healthy", "healthy": true,  "detail": {…} },
    { "name": "bm25_index",     "status": "healthy", "healthy": true,  "detail": {…} },
    { "name": "object_storage", "status": "healthy", "healthy": true,  "detail": {…} },
    { "name": "llm_providers",  "status": "configured", "healthy": true, "detail": {…} }
  ],
  "warnings": ["Running on SQLite. This is fine for local development; …"]
}
```

### `GET /llm/status?probe=false`

```jsonc
{
  "providers": [
    { "provider": "gemini", "configured": true, "model": "gemini-2.0-flash",
      "embedding_model": "text-embedding-004", "multimodal": true, "kind": "cloud" },
    { "provider": "groq", "configured": true, "model": "llama-3.3-70b-versatile",
      "fast_model": "llama-3.1-8b-instant", "multimodal": false, "kind": "cloud" }
  ],
  "primary": "gemini", "fallback": "groq",
  "any_configured": true,
  "local_llms_supported": false
}
```

**API keys are never returned** by this or any endpoint — only whether one is
configured.

### Other system endpoints

| Endpoint | Purpose |
|---|---|
| `GET /analytics?days=30` | Dashboard: stats, timeseries, distributions, recent queries |
| `GET /settings` | Safe runtime configuration and component status |
| `PATCH /settings` | Update retrieval/verification tuning (**credentials cannot be set here**) |
| `GET /strategies` | Strategy catalogue |
| `POST /cache/clear` | Clear embedding, analysis and answer caches |
| `POST /settings/reload` | Re-read configuration from the environment |

`PATCH /settings` accepts only: `default_top_k`, `candidate_pool_size`,
`rerank_enabled`, `min_relevance_score`, `corrective_relevance_floor`,
`corrective_max_rounds`, `agentic_max_steps`, `verification_enabled`,
`insufficient_evidence_threshold`, `primary_llm_provider`,
`fallback_llm_provider`. Changes apply to the running process and are not
persisted across restarts — put permanent values in `.env`.

---

## Rate limits and timeouts

No server-side rate limiting is built in — put a reverse proxy in front for
production. Relevant timeouts:

| Setting | Default | Applies to |
|---|---|---|
| `LLM_TIMEOUT_SECONDS` | 90 | Each provider call |
| `RETRIEVAL_TIMEOUT_SECONDS` | 60 | Retrieval operations |
| `VITE_API_TIMEOUT_MS` | 120000 | Frontend client |

Agentic queries can legitimately take 60–120 s — they make many calls by design.
