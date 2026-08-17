<div align="center">

# RAGX

### Adaptive Multi-Strategy Research Intelligence System

**One query. Multiple retrieval strategies. Verified answers.**

*An intelligent research infrastructure that decides **how to retrieve** before deciding **what to answer**.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector-DC244C)](https://qdrant.tech/)
[![Neo4j](https://img.shields.io/badge/Neo4j-graph-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Table of contents

1. [Problem](#1-problem)
2. [Solution](#2-solution)
3. [Architecture](#3-architecture)
4. [RAG strategies](#4-rag-strategies)
5. [The Adaptive Router](#5-the-adaptive-router)
6. [Evidence verification](#6-evidence-verification)
7. [LLM architecture](#7-llm-architecture)
8. [Tech stack](#8-tech-stack)
9. [Installation](#9-installation)
10. [Usage](#10-usage)
11. [Evaluation](#11-evaluation)
12. [Results](#12-results)
13. [Limitations](#13-limitations)
14. [Future work](#14-future-work)

---

## 1. Problem

Most RAG systems are a single fixed pipeline: embed the query, fetch the top-K
nearest chunks, stuff them into a prompt. That pipeline has a specific,
predictable set of failure modes:

| Question | Why the fixed pipeline fails |
|---|---|
| *"What mAP does **MobileNetV2** reach on **NEU-DET**?"* | Rare exact tokens sit in a sparse region of embedding space. Dense search retrieves passages *about* the topic while missing the one containing the identifier. |
| *"Why does this approach work better than earlier attempts?"* | The question and its answer are lexically different objects. The question embedding sits far from the passage that answers it. |
| *"How does this method relate to the work it builds on?"* | The relationship is never stated in any single passage — it only exists *across* passages. No amount of top-K tuning surfaces it. |
| *"What does Figure 3 show?"* | The answer is pixels. A text-only index cannot represent it. |
| *"What were the 2019 revenue figures?"* | Nothing relevant exists in the corpus. The system retrieves the *least irrelevant* chunks and the model confidently writes fiction. |

A single pipeline cannot be right for all five. Tuning it for one makes another
worse. And critically: **the system has no idea which case it is in**, so it
applies the same expensive-or-inadequate treatment to every query.

## 2. Solution

RAGX inverts the order of operations. Before retrieving anything, it
characterises the query; then it selects the smallest set of retrieval
strategies that can answer *that* query; then it verifies the answer against
the evidence it actually found.

```
Query
  → Query Understanding      (intent, complexity, modality, multi-hop, keyword vs semantic pressure)
  → Adaptive RAG Router      (select / compose strategies, and record why)
  → Multi-Strategy Retrieval (run selected strategies in parallel)
  → Retrieval Fusion         (RRF, deduplication, document diversity, reranking)
  → Evidence Verification    (claim extraction → evidence matching → confidence)
  → Cloud LLM Reasoning      (Gemini / Groq via an internal gateway)
  → Citation Validation      (every marker resolves to real evidence, or is removed)
  → Grounded Answer          (or an explicit abstention)
  → Evaluation & Analytics
```

Three properties distinguish it from a chatbot with retrieval bolted on:

**It does not run everything.** A simple lookup routes to a single dense search
with reranking disabled. The expensive path — graph traversal, hypothesis
generation, agentic planning — is reserved for queries that demonstrably need
it. The router's job is as much about *not spending* as about spending well.

**It refuses to fabricate.** After generation, the answer is decomposed into
atomic claims and matched back to the retrieved passages. Numeric claims are
checked verbatim against the evidence. If the evidence does not support the
answer, the answer is **replaced** with `Insufficient evidence found in the
indexed knowledge base.` — not shipped with a low confidence score attached.

**Every decision is inspectable.** The "Why this answer?" panel shows the query
analysis, the exact routing rules that fired and their stated reasons,
per-strategy retrieval scores, corrective-retrieval events, token and cost
accounting, and a per-claim verdict table.

<div align="center">

*Different queries, different pipelines — the same system:*

| Query | Routed to |
|---|---|
| "What optimizer was used?" | `Naive RAG` |
| "What mAP on NEU-DET?" | `Hybrid RAG` *(BM25 up-weighted)* |
| "Why does this approach work better?" | `HyDE` |
| "What does Figure 3 show?" | `Multimodal RAG` + `Corrective RAG` |
| "How does A relate to B?" | `Graph RAG` + `Hybrid RAG` |
| "What are the limitations, what causes them, how to fix?" | `Agentic RAG` + `Graph RAG` + `Corrective RAG` |

</div>

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  React 18 + Vite  ·  Dashboard · Research Assistant · Knowledge Base     │
│                      Knowledge Graph · Evaluation · Settings             │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │  REST (Axios service layer)
┌────────────────────────────────▼─────────────────────────────────────────┐
│  FastAPI  ·  Pydantic validation · structured errors · request tracing   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────┐      ┌──────────────────────────────────────┐    │
│  │  Query Analyzer    │─────▶│         Adaptive Router              │    │
│  │  heuristic + LLM   │      │  8 ordered rules · records each one  │    │
│  └────────────────────┘      └───────────────┬──────────────────────┘    │
│                                              │                           │
│         ┌────────────────────────────────────┼───────────────────┐       │
│         ▼            ▼            ▼          ▼          ▼        ▼       │
│      Naive       Hybrid        HyDE     Multimodal   Graph    Agentic    │
│         └────────────┴─────┬──────┴──────────┴──────────┘       │        │
│                            ▼                                    │        │
│                    Fusion (RRF) + Rerank ◀───── Corrective ◀─────┘        │
│                            │                    (grade → rewrite → retry)│
│                            ▼                                             │
│              ┌──────────────────────────────┐                            │
│              │  Evidence Verification       │                            │
│              │  claims → matching →         │                            │
│              │  confidence → citations      │                            │
│              └──────────────┬───────────────┘                            │
│                             ▼                                            │
│              ┌──────────────────────────────┐                            │
│              │  LLM Gateway                 │  purpose-based routing,     │
│              │  Gemini ⇄ Groq               │  retry, fallback, metering  │
│              └──────────────────────────────┘                            │
└──────────┬──────────────┬──────────────┬──────────────┬──────────────────┘
           ▼              ▼              ▼              ▼
      ┌─────────┐   ┌──────────┐   ┌───────────┐  ┌──────────────┐
      │ Qdrant  │   │  Neo4j   │   │PostgreSQL │  │Object storage│
      │ vectors │   │  graph   │   │ metadata  │  │ files/figures│
      └─────────┘   └──────────┘   └───────────┘  └──────────────┘
```

**Storage separation.** Four layers, each doing what it is good at. Original
files and extracted figures live in object storage — never as blobs in
PostgreSQL. Every layer has a working fallback so the project runs with zero
infrastructure (see [Installation](#9-installation)).

**Modular retrieval.** Every strategy implements one interface:

```python
async def retrieve(query, context, config) -> RetrievalResult
```

This is what lets the router treat strategies as interchangeable parts, and lets
composite strategies (Corrective, Adaptive, Agentic) *be* strategies that call
other strategies. Adding a ninth strategy requires no change to the router's
execution path.

Full detail: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

---

## 4. RAG strategies

| # | Strategy | Mechanism | Selected when |
|---|---|---|---|
| 1 | **Naive RAG** | Query → embedding → Qdrant top-K | Direct single-fact lookup. Also the evaluation control. |
| 2 | **Hybrid RAG** | Dense + BM25, fused with Reciprocal Rank Fusion | Exact terminology, model names, identifiers, version strings |
| 3 | **HyDE** | LLM writes a hypothetical answer passage → embed *that* → retrieve | Conceptual queries whose wording won't match the source text |
| 4 | **Multimodal RAG** | Modality-filtered retrieval + loads figure/table images for the vision model | Questions about figures, charts, diagrams, tables |
| 5 | **Corrective RAG** | Retrieve → grade → diagnose → rewrite → re-retrieve | Verification-critical, ambiguous, or complex queries |
| 6 | **Graph RAG** | Resolve entities → traverse typed relations → return source chunks | Relationship and multi-hop questions |
| 7 | **Adaptive RAG** | Analyse → route → execute in parallel → fuse | **The default entry point** |
| 8 | **Agentic RAG** | Plan → act (tools) → observe → reflect → follow up | Decomposable research questions |

Details on each — including why the mechanism addresses the failure mode it
targets — are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#retrieval-strategies).

**Composition, not selection.** Corrective RAG *wraps* another strategy;
Agentic RAG *calls* the others as tools. So "Hybrid + Corrective" and
"Graph + Agentic + Corrective" are real compositions, not labels.

---

## 5. The Adaptive Router

The router is the component the project exists to demonstrate. It reads a
`QueryAnalysis` and produces a `RoutingDecision`: the primary strategy, any
parallel strategies, whether to wrap in Corrective, whether to escalate to
Agentic, and the retrieval configuration to use.

### Query analysis

Two sources, deliberately merged rather than one overriding the other:

- **Heuristic** (free, deterministic, always runs). Interrogative structure,
  chained sub-questions, comparison/relationship markers, technical-identifier
  density, quoted phrases, visual/tabular vocabulary, query length.
- **LLM** (one fast call, Groq-preferred). Intent, complexity, decomposition
  into sub-questions, named entities, ambiguity.

The merge rule is asymmetric on purpose: booleans the heuristic can *prove* from
the query text use OR (a literal `Figure 3` requires visual retrieval regardless
of what the model says), while continuous requirements take the max — because
under-retrieving is more damaging than retrieving slightly too broadly.

Visual detection is deliberately three-tiered, because a naive keyword match
produces false positives that are expensive:

```
"What does Figure 3 show?"                 → visual   (explicit reference)
"Explain the architecture diagram."        → visual   (strong marker)
"What were the quarterly revenue figures?" → NOT visual  (guarded phrase)
"How is the graph database configured?"    → NOT visual  (guarded phrase)
```

### Routing rules

Eight rules, most-specific first. Each firing rule records its name and a
human-readable reason, which is what the UI displays verbatim:

| Rule | Fires when | Selects |
|---|---|---|
| `visual_or_tabular_requirement` | figures/tables referenced | Multimodal (+ Hybrid if exact terms) |
| `relationship_or_multi_hop` | entity relations or fact chaining | Graph (+ Hybrid if exact terms) |
| `keyword_dominant` | keyword pressure > semantic pressure | Hybrid, BM25 up-weighted |
| `high_semantic_difficulty` | conceptual + non-simple | HyDE |
| `cross_document_comparison` | comparison across documents | Hybrid, wider pool, per-document cap |
| `simple_lookup_default` | none of the above | Naive |
| `cost_guard_simple_query` | complexity is simple | reranking off, pool reduced |
| `corrective_wrapping` | verification needed / complex / ambiguous | wrap in Corrective |
| `agentic_escalation` | ≥3 of 5 complexity signals | Agentic |

**The anti-requirement is tested.** `test_router_never_selects_every_strategy`
asserts that no query fans out to all eight strategies, and
`test_simple_query_skips_expensive_work` asserts that a simple lookup never
touches HyDE, Graph, Multimodal or Agentic and runs with reranking disabled.

Try it without spending a retrieval call:

```bash
curl -X POST http://localhost:8000/api/v1/query/analyze \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does DefectNet relate to MobileNetV2?"}'
```

---

## 6. Evidence verification

The layer that makes RAGX's answers trustworthy rather than merely fluent.

```
Retrieved context
  → Relevance evaluation      (abstain immediately if nothing was retrieved)
  → Claim extraction          (answer → atomic, checkable assertions)
  → Evidence matching         (lexical support + LLM entailment, per claim)
  → Source validation         (every [n] marker resolves to a real block)
  → Confidence scoring        (6 weighted components, explicit penalties)
  → Citation validation       (invalid markers stripped, and reported)
```

**Numeric veto.** Numeric hallucination is the most damaging and most common RAG
failure. Every number in a claim must appear verbatim in the retrieved evidence;
if it does not, the claim's support score is capped regardless of what the LLM
judge said.

**Abstention is enforced, not suggested.** When confidence falls below the
threshold *and* claim support is weak, the drafted answer is discarded and
replaced. The withheld text is deliberately **not** echoed back — quoting a
fabricated claim to explain why it was rejected would reintroduce the very
content the abstention exists to suppress. (This was caught by
`test_unsupported_answer_is_replaced_not_shipped`, which asserts the fabricated
figure appears nowhere in the response.)

**Confidence is composed, not guessed:**

| Component | Weight | Measures |
|---|---|---|
| Claim support | 0.34 | fraction of claims the evidence entails |
| Retrieval quality | 0.22 | top and mean relevance of evidence used |
| Citation coverage | 0.18 | factual sentences carrying a citation |
| Citation accuracy | 0.10 | markers pointing at real blocks |
| Evidence agreement | 0.08 | corroboration across distinct documents |
| Strategy consensus | 0.08 | independent strategies surfacing the same evidence |

Penalties are then subtracted for contradicted claims, unsupported claims,
hallucinated citations, unresolved corrective rounds and numeric mismatches.
Every component and penalty is shown in the UI.

---

## 7. LLM architecture

**Cloud providers only.** Local model runtimes (Ollama, llama.cpp, local Llama
weights) are out of scope by design and are not referenced anywhere in the
codebase or UI.

```
Business logic  →  LLM Gateway  →  ┬→ Gemini  (reasoning, synthesis, vision, embeddings)
                                   └→ Groq    (fast internal steps, fallback generation)
```

No module outside `app/llm/` imports a provider SDK. The gateway owns:

- **Purpose-based routing.** Latency-sensitive internal calls (query analysis,
  relevance grading, claim extraction, reranking) prefer Groq's fast model;
  long-form reasoning prefers the configured primary; anything carrying images is
  forced to a vision-capable provider.
- **Retry then fallback.** Transient failures retry with exponential backoff on
  the same provider, then fail over to the secondary.
- **Metering.** Provider, model, latency, prompt/completion tokens and estimated
  cost recorded for every single call, attached to the query's trace.

**Key handling.** API keys are read from the backend environment only. No
endpoint returns them — `test_no_endpoint_echoes_a_credential` plants sentinel
values into every credential setting and asserts none appear in any response
body. The frontend never holds a provider key; all model traffic is proxied
through FastAPI.

---

## 8. Tech stack

**Backend** — Python 3.10+ · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) ·
Qdrant · Neo4j · PostgreSQL · rank-bm25 · NetworkX · PyMuPDF · pdfplumber ·
python-docx · pandas · Pillow · Tesseract (optional) · structlog

**Frontend** — React 18 · Vite 6 · React Router 6 · Tailwind CSS 3 · Axios ·
Recharts · React Flow · Lucide · react-markdown

**LLM** — Gemini API · Groq API

---

## 9. Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- *(optional)* Docker, for Qdrant / Neo4j / PostgreSQL / MinIO

### Zero-infrastructure quick start

RAGX runs with **no databases installed**. Qdrant has an embedded mode, the
graph falls back to an in-process NetworkX store, PostgreSQL falls back to
SQLite, and object storage falls back to the local filesystem. Every fallback is
a real implementation of the same interface — not a stub.

```bash
git clone <your-repo-url> RAGx && cd RAGx
```

**Backend**

```bash
cd backend
cp .env.example .env                     # then add GEMINI_API_KEY / GROQ_API_KEY

python -m venv .venv                     # or: python -m virtualenv .venv
source .venv/Scripts/activate            # Windows: .venv\Scripts\activate
                                         # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> Run the backend from the `backend/` directory — the default data paths in
> `.env` are relative to it. The config also reads a `.env` at the repository
> root if you prefer to keep one there instead.

→ API at <http://localhost:8000> · interactive docs at <http://localhost:8000/docs>

**Frontend** (second terminal)

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

→ App at <http://localhost:5173>

The dev server proxies `/api` to the backend, so the browser makes a same-origin
request and CORS never enters the picture.

### With production backends

```bash
docker compose up -d          # Qdrant + Neo4j + PostgreSQL + MinIO
```

Then set in `backend/.env`:

```bash
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=ragx_dev_password
DATABASE_URL=postgresql+asyncpg://ragx:ragx_dev_password@localhost:5432/ragx
STORAGE_BACKEND=s3
S3_ENDPOINT_URL=http://localhost:9000
S3_BUCKET=ragx
AWS_ACCESS_KEY_ID=ragx
AWS_SECRET_ACCESS_KEY=ragx_dev_password
```

Restart the backend — the same code now uses the server backends. Check
**Settings → Storage** to confirm each layer reports healthy.

### Environment variables

Every backend variable is documented inline in
**[`backend/.env.example`](backend/.env.example)**; frontend variables are in
**[`frontend/.env.example`](frontend/.env.example)**.
The ones that matter most:

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Primary LLM + embeddings | *(none)* |
| `GROQ_API_KEY` | Fast internal steps, fallback | *(none)* |
| `EMBEDDING_PROVIDER` | `gemini` (production) or `hashing` (offline dev) | `gemini` |
| `QDRANT_URL` | Empty ⇒ embedded Qdrant | *(empty)* |
| `NEO4J_URI` | Empty ⇒ embedded NetworkX graph | *(empty)* |
| `DATABASE_URL` | Empty ⇒ local SQLite | *(empty)* |
| `RAGX_API_KEY` | Gate on mutating endpoints | *(empty)* |
| `CORRECTIVE_RELEVANCE_FLOOR` | Below this, retrieval is repaired | `0.45` |
| `INSUFFICIENT_EVIDENCE_THRESHOLD` | Below this, the answer is withheld | `0.35` |

> **Without an LLM key:** ingestion, chunking, embedding, vector/BM25 indexing,
> retrieval, routing and the evidence panel all work. Answer generation, LLM
> query analysis, entity extraction and verification do not — and the UI says so
> in a banner rather than failing silently.

### Tests

```bash
cd backend && pytest              # 90 tests
cd frontend && npm run lint && npm run build
```

The suite runs against an isolated temporary data directory with no API keys,
exercising the real pipeline end to end: upload → parse → chunk → embed →
index → route → retrieve → verify.

---

## 10. Usage

### Demo workflow

1. **Upload** — *Knowledge Base* → drop in a PDF or paper. Watch the pipeline
   checklist advance live: Uploading → Extracting → Chunking → Embedding →
   Graph indexing → Ready.

2. **Inspect** — click the document. Chunks carry page, section and
   figure/table labels; entities and the recovered outline are listed.

3. **Compare routing** — *Research Assistant*. Ask both:
   - *"What optimizer was used?"* → routes to **Naive RAG**
   - *"What are the limitations, what causes each, and how could they be fixed?"*
     → escalates to **Agentic + Graph + Corrective**

   Open **"Why these strategies?"** on each to see the rules that fired.

4. **Verify a claim** — click any `[n]` marker in an answer. The evidence panel
   scrolls to that source; open it to read the full passage in context, with its
   neighbours.

5. **Test abstention** — ask something the corpus cannot answer, e.g.
   *"What were the quarterly revenue figures in 2019?"* RAGX withholds the
   answer instead of inventing one.

6. **Explore the graph** — *Knowledge Graph*. Select an entity, then trace a
   path to another — the same traversal Graph RAG runs internally.

7. **Benchmark** — *Evaluation* → **Run experiment**. Compare `naive`,
   `hybrid`, `adaptive` and `ragx` on the same questions.

### API

Interactive docs: `/docs` (Swagger) and `/redoc`. Full reference:
**[docs/API.md](docs/API.md)**.

```bash
# What would the router do? (no retrieval, no generation)
curl -X POST localhost:8000/api/v1/query/analyze \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does DefectNet relate to MobileNetV2?"}'

# Full pipeline, adaptive routing
curl -X POST localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What mAP does DefectNet reach on NEU-DET?"}'

# Pin a strategy (bypasses the router — what the evaluation harness does)
curl -X POST localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"...","strategies":["graph"],"top_k":8}'
```

---

## 11. Evaluation

**Hypothesis.** *Adaptive selection and composition of RAG strategies can
improve answer faithfulness and retrieval effectiveness while reducing
unnecessary retrieval and inference cost, compared with a fixed RAG pipeline.*

### Conditions

| Condition | What it isolates |
|---|---|
| `naive` … `agentic` | A single fixed strategy, router bypassed — the controls |
| `adaptive` | Routing **with verification disabled** — isolates routing alone |
| `ragx` | Routing **+ corrective + verification** — the full system |

The `adaptive` → `ragx` delta is the measured contribution of the verification
layer; the `naive` → `adaptive` delta is the contribution of routing.

### Metrics

**Retrieval** — Recall@K · Precision@K · MRR · nDCG@K · Context relevance
**Generation** — Faithfulness · Answer relevance · Groundedness · Citation accuracy
**System** — Latency (mean/p95) · Tokens · Estimated cost · Retrieval calls · Corrective rounds · Abstention rate

Groundedness and citation accuracy are computed **mechanically** from the
verification pipeline, not judged by a model grading its own output.

### Relevance labels

The benchmark ships no manual labels, so retrieval metrics use **pooled LLM
relevance judgements** (the standard TREC pooling protocol): every evaluated
strategy contributes its top-N results to a shared pool, the pool is judged once
per question, and all strategies are scored against those identical labels.

The known limitation is stated in the UI and stored on every run: *a passage no
strategy retrieved was never judged, so recall is measured relative to the
pooled candidate set.*

### Benchmark dataset

28 questions across the nine query classes the router discriminates between —
including adversarial questions where **the correct behaviour is abstention**.
The questions are corpus-agnostic (phrased against the *structure* of a
technical corpus), so the suite runs against whatever you index.

Methodology in full: **[docs/EVALUATION.md](docs/EVALUATION.md)**

---

## 12. Results

> **No benchmark numbers are published here.**
>
> This is deliberate. Results depend entirely on the corpus you index, the
> embedding model configured, and which provider answers. Publishing figures
> from one private corpus as if they characterised the system would be
> fabrication.
>
> The Evaluation dashboard is empty until you run an experiment, and returns
> `has_data: false` rather than placeholder values. Metrics that could not be
> computed render as `—`, never as `0`.
>
> **To produce results:** index your corpus → *Evaluation* → **Run experiment** →
> select `naive`, `hybrid`, `adaptive`, `ragx` → enable LLM judges. Results are
> written to PostgreSQL and rendered as comparison charts and tables.

### What *is* verified in this repository

These are properties asserted by the test suite (90 tests, all passing), not
performance claims:

- ✅ A simple query routes to Naive alone, with reranking disabled
- ✅ No query fans out to all eight strategies
- ✅ Keyword-heavy queries up-weight BM25 above dense retrieval
- ✅ Visual queries route to Multimodal; `"revenue figures"` and
      `"graph database"` do **not**
- ✅ Chained interrogatives (`"…, and what data was it trained on?"`) are
      detected as multi-hop
- ✅ Every routing decision records at least one rule with a stated reason
- ✅ Numeric claims absent from the evidence are capped in support score
- ✅ Unsupported answers are replaced with abstention, and the fabricated text
      does not appear anywhere in the response
- ✅ Hallucinated citation markers are detected, stripped and reported
- ✅ Deleting a document removes it from the vector index, BM25 index, graph
      and object storage
- ✅ No endpoint echoes any credential (sentinel-value test)

---

## 13. Limitations

**Retrieval metrics depend on pooled judgements.** Without manual labels,
Recall@K is recall over the pooled candidate set. A passage no strategy
retrieved is invisible to the metric. Supply manual `relevant_chunk_ids` in a
custom dataset for absolute recall.

**LLM-as-judge has known biases.** Faithfulness and answer relevance are judged
by a model, which can favour fluent-but-wrong answers and is not perfectly
reproducible. Groundedness and citation accuracy are mechanical precisely to
provide a non-judged counterweight.

**The router is rule-based.** Its rules are hand-derived and auditable, which is
a deliberate trade: transparent and debuggable, but not learned from data. A
learned router trained on routing outcomes is the obvious next step.

**Entity extraction quality bounds Graph RAG.** The graph is only as good as the
LLM extraction that built it. Sparse or noisy extraction degrades multi-hop
traversal, and the strategy falls back to dense retrieval when entity resolution
fails (reported in the trace rather than hidden).

**Cost estimates are estimates.** Computed from provider-reported token counts
and configured per-token pricing. Update the pricing variables if your rates
differ; they are not fetched live.

**The development embedder is not semantic.** With `EMBEDDING_PROVIDER=hashing`,
retrieval is lexical only. The system warns about this in the sidebar, on the
Evaluation page, and on every run record — but numbers produced in that mode
must never be reported as benchmarks.

**OCR requires Tesseract or Gemini.** Without either, scanned PDFs yield little
text, and the document records a warning rather than failing.

**Single-user by default.** The `User` table and API-key gate exist and are
wired, but there is no full multi-tenant auth flow (no per-user document
isolation, no sessions, no RBAC).

**Not load-tested.** Concurrency is handled correctly (parallel strategy
execution, async I/O throughout, batched embedding), but no throughput
benchmarking has been done.

---

## 14. Future work

**Learned routing.** Train the router on logged (query → strategy → outcome)
tuples. Every query already persists its analysis, decision and result quality,
so the training data accumulates automatically.

**Cross-encoder reranking.** Replace the LLM reranker with a local
cross-encoder — better latency and cost at comparable quality, and not a
generative model, so it stays within the cloud-only LLM policy.

**Incremental graph updates.** Re-extract only changed regions on reindex rather
than rebuilding a document's entire subgraph.

**Streaming true token generation.** The current stream verifies before emitting
the authoritative answer. A two-phase stream — draft tokens, then a verified
correction pass — would preserve the grounding guarantee while feeling faster.

**Manual labelling UI.** Let users mark retrieved passages as relevant directly
in the evidence panel, converting pooled judgements into a durable labelled
dataset over time.

**Multi-tenant auth.** Per-user document isolation, sessions and RBAC on top of
the existing `User` model.

**Query result caching with invalidation.** Cache full answers keyed by query +
corpus state, invalidated on ingest or delete.

---

<div align="center">

**RAGX** — *an intelligent research infrastructure that decides how to retrieve
before deciding what to answer.*

[Architecture](docs/ARCHITECTURE.md) · [Evaluation](docs/EVALUATION.md) · [API](docs/API.md) · [MIT License](LICENSE)

</div>
