# RAGX — Architecture

Design decisions and their rationale. For a project overview see the
[README](../README.md); for the benchmark protocol see
[EVALUATION.md](EVALUATION.md).

---

## Contents

- [Design principles](#design-principles)
- [Request lifecycle](#request-lifecycle)
- [Layer map](#layer-map)
- [Ingestion pipeline](#ingestion-pipeline)
- [Storage architecture](#storage-architecture)
- [Retrieval strategies](#retrieval-strategies)
- [The Adaptive Router](#the-adaptive-router)
- [Fusion and reranking](#fusion-and-reranking)
- [Evidence verification](#evidence-verification)
- [LLM Gateway](#llm-gateway)
- [Observability](#observability)
- [Security](#security)
- [Performance](#performance)
- [Extending the system](#extending-the-system)

---

## Design principles

**1. One interface for every strategy.** Everything implements
`retrieve(query, context, config) -> RetrievalResult`. This is what makes the
router possible: it treats strategies as interchangeable parts. It also lets
composite strategies *be* strategies — `CorrectiveRAG` wraps a base strategy,
`AgenticRAG` calls several as tools, and `AdaptiveRAG` orchestrates them. None
of them is a special case in the execution path.

**2. Every decision is recorded, not inferred.** The router writes the name and
reason of each rule that fired. The trace records per-strategy chunk ids and
scores, corrective events, and per-call LLM metering. The UI renders these
directly — nothing in the "Why this answer?" panel is reconstructed at display
time.

**3. Degradation is explicit and honest.** Missing infrastructure or missing API
keys never produce a silent wrong answer. Qdrant falls back to embedded mode,
Neo4j to NetworkX, Turso to a local SQLite file, Gemini embeddings to a labelled
development embedder. Each fallback is a real implementation of the same
interface, and each one that changes result quality raises a visible warning.

**4. Refusing is a valid outcome.** The verification layer can *replace* an
answer. A system that cannot decline is a system that must fabricate.

**5. Cost is a first-class concern.** The router's purpose is as much to avoid
expensive work as to select capable work. Every strategy carries a `uses_llm`
flag, every decision carries an LLM-call estimate, and every query records
actual tokens and cost.

---

## Request lifecycle

A query through `POST /api/v1/query`:

```
1.  Conversation context      history + indexed document titles loaded
2.  Follow-up resolution      lexical: carry salient terms from recent turns
                              ("and on the second dataset?" is unretrievable alone)
3.  Query analysis            heuristic (free) merged with LLM (1 fast call)
4.  Routing                   8 ordered rules → RoutingDecision + config
5.  Retrieval                 selected strategies run CONCURRENTLY
6.  Fusion                    RRF → dedupe → document diversity → rerank
7.  Generation                evidence packed to a token budget; images attached
                              if present; routed to a vision-capable provider
8.  Verification              claims → matching → confidence → citations
                              MAY REPLACE THE ANSWER
9.  Persistence               QueryRecord + one RetrievalLog per strategy
10. Response                  answer, evidence, citations, why, trace
```

Step 8 running *after* step 7 is what makes the grounding guarantee real: the
system judges what it actually produced, not what it intended to produce.

---

## Layer map

```
backend/app/
├── api/v1/            REST routes — thin; all logic lives in services
├── core/              config, logging/tracing, errors, cache, security, text
├── models/            SQLAlchemy ORM
├── schemas/           Pydantic request/response contracts
├── db/                async engine, session, schema bootstrap
├── storage/           object storage (local | S3) behind one interface
├── llm/               ← the ONLY place a provider SDK is imported
│   ├── base.py        Message / LLMRequest / LLMResponse / LLMProvider
│   ├── gemini/        Gemini provider
│   ├── groq/          Groq provider
│   ├── gateway.py     purpose routing, retry, fallback, metering
│   ├── embeddings.py  Gemini embedder + labelled dev embedder
│   └── prompts.py     every prompt, versioned in one place
├── indexing/          vector_store (Qdrant) · bm25_index · graph_store
├── ingestion/         parsers/ · ocr · chunking · entities · pipeline
├── retrieval/
│   ├── base.py        the strategy interface
│   ├── registry.py    lazy construction, no import cycles
│   ├── loader.py      batched chunk hydration
│   ├── fusion.py      RRF, weighted fusion, dedupe, diversity
│   ├── rerank.py      heuristic + LLM reranking
│   ├── query_rewrite.py
│   └── naive|hybrid|hyde|multimodal|corrective|graph|adaptive|agentic/
├── verification/      claims · evidence_matcher · citations · confidence · pipeline
├── evaluation/        metrics/ · benchmark · runner · datasets/
└── services/          orchestration used by the API layer
```

**Dependency direction.** `api → services → retrieval/verification → indexing/llm
→ storage/db`. Nothing lower imports something higher. `app/llm` is the seam that
isolates provider SDKs; swapping Gemini for another provider touches one
directory.

---

## Ingestion pipeline

```
upload → parse → OCR/tables/figures → chunk → persist
       → embed → vector + BM25 index → entity extraction → graph → ready
```

Each stage writes its status and duration to `Document.processing_steps`, which
is what the Knowledge Base checklist renders live.

### Parsing

| Format | Extraction |
|---|---|
| **PDF** | PyMuPDF for layout-aware text; font size + weight detect headings and build the section path. pdfplumber extracts tables → Markdown. Embedded images above a size threshold become figure blocks, paired with the nearest `Figure N` caption. |
| **DOCX** | python-docx; heading styles build the hierarchy, tables → Markdown |
| **TXT / MD** | Markdown headings and numbered headings build the section path; code fences preserved as `code` blocks |
| **CSV** | Schema summary (columns, dtypes, `describe()`) plus row windows as Markdown tables — keeps numeric questions answerable and citable by row range |
| **Images** | OCR (Tesseract) and/or a Gemini vision description; raw bytes retained as multimodal evidence |
| **HTML** | BeautifulSoup; nav/script/footer stripped |

**Thin-text pages** (scanned papers) are rendered at 200 dpi and sent through
OCR — capped at 25 pages so a large scan cannot stall ingestion.

**Figures** get a text description attached (Tesseract first, Gemini vision when
OCR yields little). That description is what gets embedded, which is how a chart
containing no text layer becomes retrievable at all.

### Structure-aware chunking

Four rules, in priority order:

1. **Never cross a section boundary** — a chunk belongs to exactly one section
   path, so its citation (`Methodology > Training setup`) is truthful.
2. **Never split an atomic unit** — tables, figures, images and code blocks
   become their own chunks. Splitting a Markdown table mid-row destroys it. An
   oversized table splits on row boundaries *with the header repeated*, so each
   part remains a valid table.
3. **Split long prose at sentence boundaries**, with sentence-level overlap, so a
   fact spanning a boundary is retrievable from either side.
4. **Prefix the heading breadcrumb** to the embedded text (not the stored
   content). This gives the embedder the context a human reader gets from page
   layout, and helps when a section's body omits its own topic.

Chunks below `CHUNK_MIN_TOKENS` are merged forward so the index is not polluted
with fragments.

### Entity extraction

The most expensive stage, so it is bounded. Chunks are ranked by *density of
technical terms per token*, with method/results sections up-weighted — an
abstract yields far more graph structure than a page of boilerplate. Only the
top `ENTITY_EXTRACTION_MAX_CHUNKS` go to the LLM.

A deterministic rule-based pass (acronyms, model identifiers, CamelCase) always
runs, so the graph has real content even with no API key.

**Hallucination guard:** a relation is discarded unless *both* endpoints appear
in the same extraction's entity list. This is what keeps invented edges out of
the graph.

---

## Storage architecture

| Layer | Production | Fallback | Holds |
|---|---|---|---|
| Vector | Qdrant server | Qdrant **embedded** (same client, local path) | dense embeddings + citation payload |
| Graph | Neo4j (Cypher) | NetworkX (in-process, JSON-persisted) | entities, typed relations |
| Relational | Turso (hosted libSQL) | local SQLite file | documents, chunks, queries, evaluations |
| Objects | S3 / MinIO | local filesystem | original files, extracted figures |

**Why payloads live in Qdrant.** Each vector carries `document_id`, page,
section, figure/table label and a short preview. A single-hop retrieval is
therefore citable without a database round-trip, and modality filters push down
into the index.

**Why the graph has two backends.** Neo4j is the right production answer, but
requiring it to demonstrate Graph RAG would make the project undemonstrable
without Docker. The NetworkX store implements the same traversal semantics —
typed edges, bounded-depth BFS with confidence decay per hop, shortest paths
between entities — in-process. Both return `GraphPath` objects carrying the
chunk each edge was extracted from, so a graph hit is ordinary citable evidence.

**Why large files never touch the database.** `Document.object_key` points at
object storage. SQL holds metadata and extracted text only.

---

## Retrieval strategies

### 1 · Naive RAG

`query → embedding → Qdrant top-K`. No rewriting, no fusion, no grading.
Deliberately the simplest possible pipeline: the fast path for genuinely simple
questions, and the control condition in evaluation.

### 2 · Hybrid RAG

Dense and BM25 run **concurrently**, fused with Reciprocal Rank Fusion.

Dense retrieval generalises across paraphrase but is unreliable on rare exact
tokens — model names, dataset identifiers, version strings often sit in a sparse
region of embedding space. BM25 handles exactly those and fails on paraphrase.
Running both covers each one's blind spot.

The blend is tunable, not fixed: the router raises `sparse_weight` when analysis
reports high keyword pressure. RRF is the default fusion because it reads only
*positions*, so it needs no score normalisation between cosine similarity and
BM25's unbounded scale.

### 3 · HyDE

`query → LLM writes a hypothetical answer passage → embed that → retrieve`.

The premise: a question and its answer are lexically different objects, so a
question embedding sits some distance from the passages that answer it. A
hypothetical *answer* lives in the same neighbourhood as the real one.

Both probes run — the hypothesis (weight 0.65) and the original query (0.35) —
so a hypothesis that drifts off-topic cannot destroy retrieval. The hypothetical
text is discarded after retrieval and never enters the answer context.

Costs one LLM call, so the router gates it on complexity.

### 4 · Multimodal RAG

Two things distinguish it from text retrieval:

- **Modality-aware retrieval.** One probe restricted to visual/tabular chunks,
  one unrestricted, fused with the visual side up-weighted. Restricting alone
  would lose the prose explaining the figure; not restricting buries the figure.
- **Image payload loading.** Retrieved figure chunks carry an object-storage key;
  the strategy fetches the bytes and attaches them, so the generator hands
  *actual pixels* to Gemini rather than only a caption.

### 5 · Corrective RAG

```
retrieve → grade → if poor: diagnose → rewrite → re-retrieve → merge → re-grade
```

A **wrapper**, which is why the router can compose "Hybrid + Corrective" or
"Graph + Corrective" without new code.

Grading is two-tiered so the cheap signal can veto the expensive one. Heuristic
grading (score magnitude and spread, query-term coverage, technical-term hit
rate, result count) always runs and catches unambiguous failures for free. LLM
grading escalates only when the heuristic verdict is *borderline* — and it is
what supplies the failure diagnosis.

The repair action follows the diagnosis rather than being applied blindly:

| Diagnosis | Action |
|---|---|
| `empty` | rewrite + broaden (drop filters, double the pool) |
| `off_topic` | rewrite the query |
| `too_general` | rewrite + widen pool and top-K |
| `wrong_document` | drop the document filter |
| `partially_relevant` | widen and rerank |

If every round still fails, `insufficient_evidence` is set and the generator
abstains. Failing loudly is the point of this strategy.

### 6 · Graph RAG

Vector search answers *"what does the corpus say about X"*. It cannot answer
*"how does X relate to Y"* when no single passage states the relation — the
connection only exists across passages.

1. Resolve query entities against the graph (analysis entities first, then
   surface-form technical terms, then a free-text lookup).
2. Traverse: for two named entities, find the *path between them*; otherwise
   expand the neighbourhood to bounded depth with confidence decaying per hop.
3. Return the chunks each traversed edge was extracted from.

Dense results are blended in (weight 0.5) so a graph miss degrades to normal
retrieval rather than an empty answer. Every fallback is reported in the trace.

### 7 · Adaptive RAG

Deliberately thin. All the intelligence is in the analyzer and router; all the
retrieval work is in the strategies it delegates to. Its job is orchestration:
run the selected strategies concurrently, fuse while preserving per-strategy
provenance, and carry the routing explanation through to the response.

A single strategy failing is caught and reported, not propagated — the query
still returns whatever the others found.

### 8 · Agentic RAG

```
plan → act (concurrent waves) → observe (grade) → reflect → optional follow-up
```

The agent's **action space is the other strategies**: `dense_search`,
`hybrid_search`, `graph_search`, `multimodal_search`, `hyde_search`. It selects
strategies rather than reimplementing retrieval.

Steps with satisfied dependencies run concurrently. A dependent step receives
the earlier step's finding injected into its query — which is what makes a hop
genuinely multi-hop rather than two independent lookups.

Reflection asks whether the evidence is sufficient; only a *specific, nameable*
gap earns one more step. Bounded by `AGENTIC_MAX_STEPS`, with a guard against
dependency cycles, so the loop always terminates.

---

## The Adaptive Router

### Analysis merge

| Signal type | Merge rule | Why |
|---|---|---|
| Provable booleans (`requires_visual`) | OR | A literal `Figure 3` requires visual retrieval regardless of model opinion |
| Continuous requirements | max | Under-retrieving is worse than retrieving slightly too broadly |
| Complexity | max of the two | Under-routing a complex query is the expensive mistake |
| Intent, sub-questions, entities | LLM | Needs semantics the heuristic lacks |

### Visual detection

Three tiers, because naive keyword matching produces expensive false positives:

1. **Explicit reference** (`Figure 3`, `fig. 2b`) → always visual.
2. **Guarded phrase** (`revenue figures`, `graph database`, `figures for 2019`)
   → never visual.
3. **Strong marker** (`chart`, `diagram`, `plot`) fires alone; **weak marker**
   (`figure`, `image`, `graph`) needs a second signal — another weak marker or a
   verb of showing.

### Multi-hop detection

The strongest lexical signal is a **chained interrogative**: `"…, and what data
was that model trained on?"` — a second, dependent question. Also triggered by
≥3 interrogatives, multiple question marks, or comparison plus conjunction in a
long query.

### Cost guards

- Simple queries: reranking **disabled**, candidate pool reduced, top-K reduced.
- Single-document corpus: graph traversal depth capped.
- Agentic escalation requires **≥3 of 5** complexity signals.
- Corrective is the recovery path, so the router need not over-provision up
  front — if retrieval actually fails, correction repairs it.

---

## Fusion and reranking

**Reciprocal Rank Fusion** — `score = Σ wₛ / (k + rankₛ)`. Position-based, so no
score normalisation is needed between scoring universes. Used whenever lists come
from different retrievers.

**Weighted fusion** — used when both lists are already on a comparable 0–1 scale
and the router wants an explicit dense/sparse bias.

**Deduplication** — near-duplicate evidence (TF-cosine ≥ 0.92 within the same
document) is merged, keeping the higher-scoring copy and unioning its sources.
Overlapping chunks otherwise waste context budget and inflate apparent agreement.

**Document diversity** — caps per-document contribution for cross-document
queries, applied only when enough distinct documents are actually present.
Overflow is appended rather than discarded.

**Reranking** — two tiers:
- *Heuristic* (free, always): blends retrieval score with query-term coverage,
  exact technical-identifier matches, cross-strategy consensus, modality fit,
  position prior and a short-chunk penalty. This alone fixes the common case
  where a semantically-adjacent chunk outranks the one literally containing the
  requested identifier.
- *LLM* (conditional): runs when the router flags verification-critical, or when
  heuristic scores are **flat** (σ < 0.045 across the top 8 — the retriever could
  not discriminate). Blended 60/40 with the retrieval score so a confidently
  wrong judge cannot fully override cross-strategy consensus. Passages the judge
  omits are demoted, not dropped.

---

## Evidence verification

### Claim extraction
LLM path resolves pronouns and splits compound sentences; deterministic path
(sentence segmentation + marker parsing + hedge/meta filtering) runs without an
LLM, so verification never silently disappears.

### Evidence matching
Two independent judgements per claim:
- **Lexical support** — content-word overlap with the best-matching passage.
- **Entailment** — a strict LLM grader returning
  supported / partially_supported / unsupported / contradicted, restricted to
  evidence ids it was actually shown.

Combined 70/30 in favour of entailment, with a **numeric veto**: every number in
a claim must appear verbatim in the evidence, or the support score is capped at
0.35 regardless of the verdict.

### Citation validation
Markers are parsed, validated against the evidence the model actually saw (not
the full retrieval result — otherwise markers would point at evidence that never
entered the prompt), and invalid markers are stripped from the answer while being
*reported* as hallucinated citations.

### Abstention enforcement
Triggered when confidence < `INSUFFICIENT_EVIDENCE_THRESHOLD` **and** either
claim support < `MIN_CLAIM_SUPPORT_SCORE` or retrieval quality < threshold.

The withheld text is **not** echoed. Quoting a fabricated claim to explain its
rejection reintroduces the content the abstention exists to suppress; only
counts and categories are reported, with full verdicts available in the
verification payload.

---

## LLM Gateway

```
Purpose ──▶ provider selection ──▶ retry (backoff) ──▶ fallback ──▶ metering
```

| Purpose group | Preferred | Rationale |
|---|---|---|
| query analysis, grading, rerank, claim extraction, evidence matching | Groq fast model | Latency dominates; these are short structured calls |
| synthesis, planning, judging | configured primary (Gemini) | Depth matters more than latency |
| anything with images | vision-capable only | No meaningful text-only fallback exists for a visual question |

JSON responses are parsed tolerantly — models occasionally wrap JSON in prose or
code fences even under explicit instruction; `parse_json_response` recovers the
payload instead of failing the request.

Streaming falls back to a single non-streaming call on the secondary provider if
the primary stream fails *before emitting anything*, so the caller's contract
(an async iterator of strings) always holds.

---

## Observability

`TraceRecorder` accumulates, per query:

- stage spans with durations
- per-strategy chunk ids, scores, effective query, round index
- corrective events (round, diagnosis, action taken)
- per-LLM-call provider, model, purpose, latency, tokens, cost, fallback flag
- aggregate tokens, cost, retrieval calls, corrective rounds

**Redaction is structural.** A processor redacts any log field whose key matches
a sensitive name, text snippets are hard-truncated, and full document bodies are
never logged. Identifiers, scores and counts are logged; content is not.

Every trace persists to `QueryRecord.trace`, so an answer can be re-inspected
long after the request.

---

## Security

| Concern | Control |
|---|---|
| Credentials | Environment only. No endpoint returns them; asserted by a sentinel-value test across all read endpoints. |
| Frontend exposure | Browser never holds a provider key. All model traffic proxied through FastAPI. |
| Upload validation | Extension allow-list **+** declared MIME check **+** magic-byte sniffing **+** size ceiling. A file is accepted only when all three agree; text formats additionally reject NUL bytes. |
| Path traversal | Filenames sanitised to a basename with unsafe characters stripped; object keys resolved and confirmed to stay within the storage root. |
| Error leakage | Unhandled exceptions log the trace server-side and return an opaque message. No stack trace reaches a client. |
| CORS | Explicit origin allow-list, `allow_credentials=False`, restricted methods and headers. |
| API gate | Optional `RAGX_API_KEY` on mutating routes, compared with `secrets.compare_digest`. |
| Prompt injection | Retrieved content is framed as evidence with explicit grounding rules, and — more importantly — the verification layer checks the *output* against the evidence rather than trusting the model to have behaved. |

---

## Performance

- **Parallel retrieval** — selected strategies run under `asyncio.gather`;
  Hybrid runs dense and BM25 concurrently; Multimodal runs three probes
  concurrently; agentic steps execute in dependency waves.
- **Batched hydration** — one SQL round-trip per retrieval, never per-hit.
- **Caching** — TTL+LRU caches for query embeddings, query analysis and HyDE
  hypotheses. All three are pure functions of their input and each costs an API
  call.
- **Adaptive cost control** — the router's primary performance mechanism. A
  simple query runs one dense search with no rerank and no LLM beyond analysis
  and generation.
- **Batched embedding** — configurable batch size; documents embed in batches,
  not per-chunk.
- **Frontend** — route-level code splitting with Recharts and React Flow in
  separate chunks, so the first paint is not blocked by visualisation libraries.

---

## Extending the system

### Add a retrieval strategy

1. Implement `RetrievalStrategy` in `app/retrieval/<name>/strategy.py`.
2. Add the enum member to `StrategyName` (its label comes free).
3. Register it in `app/retrieval/registry.py::_build`.
4. Add a routing rule in `AdaptiveRouter.route` — with a stated reason.
5. Add display metadata to `frontend/src/utils/constants.js::STRATEGIES`.

Nothing else changes: fusion, reranking, verification, evaluation and the UI all
operate on the shared interface.

### Add an LLM provider

Implement `LLMProvider` in `app/llm/<provider>/`, register it in the gateway's
`_providers` map, and add its selection preference. No business logic changes —
that is the point of the gateway.

### Add a document format

Implement `DocumentParser` in `app/ingestion/parsers/`, add it to `_PARSERS`, and
add the extension to `ALLOWED_EXTENSIONS`. Emit `ContentBlock`s with section
paths and the pipeline handles the rest.
