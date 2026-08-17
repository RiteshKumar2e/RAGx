# RAGX — Evaluation Methodology

How RAGX measures itself, what the numbers mean, and what they do not mean.

> **No results are published in this repository.** Benchmark figures depend
> entirely on the corpus indexed, the embedding model configured and which
> provider answers. Publishing numbers from one private corpus as if they
> characterised the system would be fabrication. Run the experiments on your own
> corpus; the dashboard shows an empty state until you do.

---

## Contents

- [Hypothesis](#hypothesis)
- [Experimental design](#experimental-design)
- [Benchmark dataset](#benchmark-dataset)
- [Relevance labelling](#relevance-labelling)
- [Metrics](#metrics)
- [Running an experiment](#running-an-experiment)
- [Reading the results](#reading-the-results)
- [Threats to validity](#threats-to-validity)
- [Reproducibility](#reproducibility)

---

## Hypothesis

> **H₁** — Adaptive selection and composition of RAG strategies improves answer
> faithfulness and retrieval effectiveness *while reducing* unnecessary
> retrieval and inference cost, compared with a fixed RAG pipeline.

This is two claims, and they pull in opposite directions:

- **H₁a (quality)** — adaptive routing achieves higher faithfulness,
  groundedness and citation accuracy than any single fixed strategy averaged
  across a heterogeneous query set.
- **H₁b (cost)** — it does so without a proportional increase in tokens,
  latency and retrieval calls, because cheap queries are routed to cheap
  pipelines.

A system that wins on quality by always running every strategy would **falsify**
H₁b. That is precisely why both are measured, and why cost metrics sit beside
quality metrics in the comparison table rather than in a footnote.

### Sub-hypotheses

| ID | Claim | Isolated by |
|---|---|---|
| **H₂** | Verification improves grounding and reduces fabrication | `adaptive` vs `ragx` |
| **H₃** | Routing alone (no verification) beats any fixed strategy | `naive`/`hybrid`/… vs `adaptive` |
| **H₄** | Corrective retrieval recovers otherwise-failed retrievals | corrective-round counts vs quality on `poor_retrieval` |
| **H₅** | Specialised strategies win on the query classes they target | per-category breakdown |

---

## Experimental design

### Conditions

| Condition | Router | Verification | Isolates |
|---|---|---|---|
| `naive` | ✗ pinned | ✓ | Dense-only baseline — **the control** |
| `hybrid` | ✗ pinned | ✓ | Dense + BM25 fusion |
| `hyde` | ✗ pinned | ✓ | Hypothetical-document probing |
| `multimodal` | ✗ pinned | ✓ | Modality-aware retrieval |
| `corrective` | ✗ pinned | ✓ | Grade-and-repair |
| `graph` | ✗ pinned | ✓ | Entity traversal |
| `agentic` | ✗ pinned | ✓ | Multi-step planning |
| `adaptive` | ✓ | **✗** | **Routing alone** |
| `ragx` | ✓ | ✓ | **The full system** |

Two deltas carry the hypothesis:

```
naive → adaptive   =  contribution of ROUTING
adaptive → ragx    =  contribution of VERIFICATION
```

### Controls

Everything except the strategy is held constant: the same questions in the same
order, the same K, the same corpus, the same embedding model, the same providers,
and the same pooled relevance labels applied to every condition.

Pinning is implemented by the same mechanism the API exposes
(`strategies: ["hybrid"]`), so a benchmarked condition runs exactly the code path
a user gets — not a special evaluation-only branch.

### Execution protocol

Three phases, ordered so that all conditions are scored against *identical*
labels:

```
Phase A — Execution
  For each (condition, question): run the full pipeline with strategies pinned.
  Capture answer, evidence, latency, tokens, cost, retrieval calls,
  corrective rounds, abstention.
  Generation judges run here (they depend only on that run's own output).

Phase B — Pooled labelling
  Union every condition's top-N results per question into one pool.
  Judge the pool ONCE per question.

Phase C — Scoring
  Compute retrieval metrics for every run against those shared labels.
  Aggregate per condition and persist.
```

Phase B must not run per-condition — that would give each strategy its own
labels and make the comparison meaningless.

---

## Benchmark dataset

`backend/app/evaluation/datasets/ragx_benchmark.json` — 28 questions across the
nine query classes the router discriminates between.

| Category | n | Tests |
|---|---|---|
| `simple` | 4 | Single-fact lookup. **Also tests that the router does not over-provision.** |
| `keyword` | 4 | Acronyms, model names, numeric values — dense retrieval's weak spot |
| `semantic` | 3 | Conceptual phrasing unlikely to match source wording |
| `multi_hop` | 3 | Chained facts across passages |
| `relationship` | 3 | Entity relations — graph traversal |
| `cross_document` | 2 | Synthesis across documents; tests document diversity |
| `multimodal` | 3 | Figures, charts, tables |
| `poor_retrieval` | 3 | **Adversarial — the correct answer is abstention** |
| `complex_research` | 3 | Decomposable; tests agentic escalation |

### Why questions are corpus-agnostic

They are phrased against the **structure** of a technical corpus ("What is the
main contribution described in the abstract?") rather than against specific
content. This means:

- ✅ The suite runs against whatever you index, with no dataset curation.
- ✅ Every category is exercised for any research corpus.
- ⚠️ Absolute scores are **not comparable across corpora** — only *relative*
  comparison between conditions on the *same* corpus is meaningful. This is the
  central caveat when reading any RAGX result.

For corpus-specific evaluation, add a dataset with manual
`relevant_chunk_ids` per question; the runner uses manual labels in preference
to pooling whenever they exist.

### The adversarial category matters most

`poor_retrieval` questions have no answer in any reasonable technical corpus
("What were the quarterly revenue figures in 2019?"). They measure the property
that separates a trustworthy system from a fluent one, via two rates reported
separately:

- **Correct abstention rate** — adversarial questions correctly declined.
- **False abstention rate** — answerable questions wrongly declined.

A system can trivially maximise the first by abstaining always; reporting both
prevents that from looking like success.

---

## Relevance labelling

Retrieval metrics need to know which passages are relevant. RAGX obtains labels
two ways, and records which was used on every run.

### `manual` — labels shipped in the dataset
Used whenever a dataset provides `relevant_chunk_ids`. Highest fidelity.

### `pooled` — the default

The standard **TREC pooling** protocol:

1. Each condition contributes its top-N results per question to a shared pool.
2. An LLM judge labels each pooled passage once for relevance to the question.
3. All conditions are scored against those identical labels.

**The known limitation, stated plainly:** a relevant passage that *no* evaluated
strategy retrieved is never judged, and therefore never counted. Pooled
Recall@K is honestly read as **"recall over the pooled candidate set"**, not
absolute recall. This caveat is stored in `EvaluationRun.config.label_source` and
displayed in the UI above the results table.

Pooling systematically favours no single condition, because the pool is the union
of all conditions' results. It does understate recall for all of them equally.

### When labels are unavailable

If no manual labels exist and LLM judging is disabled, retrieval metrics are
returned as `null` — and render as `—`, never `0`. A metric that could not be
computed is visibly different from a metric that scored zero.

---

## Metrics

### Retrieval

| Metric | Definition |
|---|---|
| **Recall@K** | `|relevant ∩ top-K| / |relevant|` |
| **Precision@K** | `|relevant ∩ top-K| / K` |
| **MRR** | `1 / rank` of the first relevant result |
| **nDCG@K** | Position-discounted gain, normalised by the ideal ranking |
| **Context relevance** | LLM-judged proportion of retrieved passages that are useful |

### Generation

| Metric | Computed by | Definition |
|---|---|---|
| **Faithfulness** | LLM judge | Fraction of the answer entailed by the context. A correct abstention scores 1.0. |
| **Answer relevance** | LLM judge | How directly and completely the question is addressed |
| **Groundedness** | **mechanical** | Claims with supporting evidence ÷ total claims, from the verification pipeline |
| **Citation accuracy** | **mechanical** | Marker validity blended with coverage of factual sentences |

Groundedness and citation accuracy are deliberately **not** LLM-judged. They are
computed from the verification pipeline's own records, providing a
non-judged counterweight to the two judged metrics — so a systematically biased
judge cannot move every generation number in the same direction.

### System

Latency (mean / median / p95 / min / max) · total and average tokens ·
estimated cost · retrieval calls · LLM calls · corrective rounds ·
abstention rate · failures.

**Cost is estimated** from provider-reported token counts multiplied by the
configured per-token pricing. Update the pricing variables in `.env` if your
rates differ — they are not fetched live.

---

## Running an experiment

### From the UI

*Evaluation* → **Run experiment**:

1. Select conditions — `naive`, `hybrid`, `adaptive`, `ragx` is the minimal set
   that tests both deltas.
2. Select categories, or leave empty for all.
3. Set questions per run. **Start small** — each question costs several API
   calls *per condition*.
4. Set K (default 8).
5. Enable **Run LLM judges** to compute generation metrics and produce pooled
   labels.

Runs execute in the background; the page polls and updates as each completes.

### From the API

```bash
curl -X POST localhost:8000/api/v1/evaluation/run \
  -H 'Content-Type: application/json' \
  -d '{
    "strategies": ["naive", "hybrid", "adaptive", "ragx"],
    "limit": 10,
    "k": 8,
    "judge_generation": true,
    "name": "routing-ablation-v1"
  }'

curl localhost:8000/api/v1/evaluation/runs         # progress
curl localhost:8000/api/v1/evaluation/comparison   # side-by-side
```

### Cost estimate

Roughly, per question per condition:

| Condition | LLM calls | Note |
|---|---|---|
| `naive` | ~4 | analysis + generation + 2 verification |
| `hybrid` | ~4 | retrieval adds none |
| `hyde` | ~5 | + hypothesis |
| `corrective` | ~6–8 | + grading and rewriting when triggered |
| `agentic` | ~8–12 | + planning, per-step, reflection |
| `ragx` | varies | **the point** — cheap queries stay cheap |

Plus ~3 judge calls per question per condition, and ~1 pooling call per question
total. A 4-condition × 10-question run with judges is roughly 250–350 LLM calls.

### Guard rails

- The API **refuses** to start a run when no documents are indexed — otherwise
  every condition would score zero and the comparison would be meaningless.
- Warnings are attached to the run (and shown in the UI) when no LLM is
  configured, when the development embedder is active, or when labels will come
  from pooling.

---

## Reading the results

### The comparison table
Latest completed run per condition. Best value per metric is marked ★. `—` means
**not computed**, not zero.

### What supports H₁

| Observation | Interpretation |
|---|---|
| `ragx` faithfulness > every fixed strategy | H₁a supported |
| `ragx` avg tokens/cost ≈ or < the *mean* of fixed strategies | H₁b supported |
| `adaptive` > `naive` on quality | H₃ supported — routing alone helps |
| `ragx` > `adaptive` on groundedness/citation accuracy | H₂ supported — verification helps |
| `ragx` correct-abstention rate high, false-abstention low | Fabrication resistance without over-refusal |

### What would falsify it

- `ragx` matches a fixed strategy on quality but costs more → routing overhead
  without benefit.
- `ragx` wins only because it abstains more → check the **false** abstention
  rate; abstaining on answerable questions is a failure, not caution.
- Per-category results show no strategy specialisation → the router's premise
  (different queries need different retrieval) is not holding on this corpus.

### Per-category breakdown
Stored in `EvaluationRun.config.by_category`. This is where H₅ is tested — the
expectation is `graph` leading on `relationship`, `hybrid` on `keyword`,
`multimodal` on `multimodal`, and `ragx` competitive across *all* of them
without leading each individually.

### Strategy usage
`config.strategy_usage` records which underlying strategies the router actually
selected during `adaptive`/`ragx` runs — the direct evidence of whether routing
is discriminating or collapsing to one default.

---

## Threats to validity

| Threat | Mitigation | Residual risk |
|---|---|---|
| **LLM-as-judge bias** — favours fluent answers | Groundedness and citation accuracy computed mechanically | Faithfulness and answer relevance remain judge-dependent |
| **Pooling bias** — unretrieved relevant passages never judged | Pool is the union of all conditions | Absolute recall understated for all conditions equally |
| **Corpus dependence** — questions are structural | Only relative comparison claimed | Cross-corpus comparison is invalid |
| **Small n** — default runs are small | Latency reported as mean *and* p95 | Small samples are noisy; increase `limit` for stable numbers |
| **Judge non-determinism** — temperature 0 is not a guarantee | Fixed prompts, temperature 0, prompts versioned | Repeated runs can differ slightly |
| **Self-evaluation** — same model family answers and judges | Configure a different provider for judging | Shared-blind-spot risk remains |
| **Dev embedder** — lexical only | Loudly warned in UI, sidebar, run config | Numbers from that mode are not benchmarks |

---

## Reproducibility

Every run persists:

- dataset name and version, K, categories, question limit
- embedding provider and whether it is production-ready
- whether an LLM was configured, and the judging setting
- label source and its full pooling summary
- per-question results: retrieved chunk ids, all metrics, latency, tokens, cost
- per-category aggregates and the strategy-usage distribution
- warnings that applied at run time

To reproduce a reported result you need: the same corpus, the same
`EvaluationRun.config`, and the same provider/model versions. Provider model
updates are the main uncontrolled variable — pin `GEMINI_MODEL` and `GROQ_MODEL`
to specific versions when reporting.

Prompts are versioned in `app/llm/prompts.py` (`PROMPT_VERSION`); record it
alongside any published result.
