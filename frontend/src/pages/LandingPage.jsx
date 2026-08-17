import { Link } from 'react-router-dom';
import {
  ArrowRight,
  BadgeCheck,
  Boxes,
  Cloud,
  FileSearch,
  FlaskConical,
  GitBranch,
  Image as ImageIcon,
  Layers,
  Quote,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Workflow,
} from 'lucide-react';
import { STRATEGIES, STRATEGY_ORDER } from '../utils/constants';

const PIPELINE = [
  { label: 'Query', icon: FileSearch },
  { label: 'Query Understanding', icon: ScanSearch },
  { label: 'Adaptive Router', icon: Workflow },
  { label: 'Multi-Strategy Retrieval', icon: Layers },
  { label: 'Evidence Verification', icon: ShieldCheck },
  { label: 'Grounded Answer', icon: BadgeCheck },
];

const CAPABILITIES = [
  {
    icon: Workflow,
    title: 'Adaptive strategy routing',
    body: 'A query analyser scores intent, complexity, keyword and semantic pressure, multi-hop depth and modality. The router then selects the smallest set of strategies that can answer the question — and records why.',
  },
  {
    icon: ShieldCheck,
    title: 'Evidence verification',
    body: 'Answers are decomposed into atomic claims and matched back to retrieved passages. Numeric claims are checked against the evidence verbatim. Unsupported answers are withheld, not scored down.',
  },
  {
    icon: Quote,
    title: 'Traceable citations',
    body: 'Every citation resolves to a document, page, section and figure or table number. Click a marker to read the exact passage the claim came from.',
  },
  {
    icon: ImageIcon,
    title: 'Multimodal ingestion',
    body: 'PDFs, DOCX, CSV, text and images. Tables are preserved as tables, figures are extracted and described, and scanned pages are recovered with OCR.',
  },
  {
    icon: GitBranch,
    title: 'Knowledge graph',
    body: 'Entities and typed relations are extracted during ingestion and stored in a graph, enabling relationship and multi-hop questions that no single passage answers.',
  },
  {
    icon: FlaskConical,
    title: 'Real evaluation',
    body: 'A benchmark harness compares every strategy on retrieval, generation and cost metrics. Numbers appear only after you run an experiment — nothing is pre-filled.',
  },
];

function StrategyCard({ name }) {
  const meta = STRATEGIES[name];
  if (!meta) return null;

  return (
    <div className="group rounded-xl border border-ink-200 bg-white p-4 transition hover:border-ink-300 hover:shadow-card-hover">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: meta.color }} aria-hidden="true" />
        <h3 className="text-sm font-semibold text-ink-900">{meta.label}</h3>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-ink-600">{meta.description}</p>
      <p className="mt-2 text-xs text-ink-400">
        <span className="font-medium text-ink-500">Best for:</span> {meta.bestFor}
      </p>
    </div>
  );
}

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white">
      {/* ---------------------------------------------------------------- nav */}
      <header className="sticky top-0 z-30 border-b border-ink-100 bg-white/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
          <Link to="/" className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-950 text-sm font-bold text-white">
              RX
            </span>
            <span className="text-sm font-semibold tracking-tight text-ink-900">RAGX</span>
          </Link>

          <nav className="flex items-center gap-2">
            <a
              href="#architecture"
              className="hidden rounded-lg px-3 py-1.5 text-sm text-ink-600 transition hover:bg-ink-50 hover:text-ink-900 sm:block"
            >
              Architecture
            </a>
            <a
              href="#strategies"
              className="hidden rounded-lg px-3 py-1.5 text-sm text-ink-600 transition hover:bg-ink-50 hover:text-ink-900 sm:block"
            >
              Strategies
            </a>
            <Link
              to="/dashboard"
              className="rounded-lg px-3 py-1.5 text-sm text-ink-600 transition hover:bg-ink-50 hover:text-ink-900"
            >
              Dashboard
            </Link>
            <Link
              to="/research"
              className="rounded-lg bg-ink-950 px-3.5 py-1.5 text-sm font-medium text-white transition hover:bg-ink-800"
            >
              Open app
            </Link>
          </nav>
        </div>
      </header>

      {/* --------------------------------------------------------------- hero */}
      <section className="relative overflow-hidden border-b border-ink-100">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.35]"
          style={{
            backgroundImage:
              'radial-gradient(circle at 20% 10%, rgba(53,103,240,0.10), transparent 45%), radial-gradient(circle at 85% 20%, rgba(139,92,246,0.10), transparent 45%)',
          }}
          aria-hidden="true"
        />
        <div className="relative mx-auto max-w-6xl px-4 py-16 sm:px-6 sm:py-24">
          <div className="mx-auto max-w-3xl text-center">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-ink-200 bg-white px-3 py-1 text-xs font-medium text-ink-600">
              <Sparkles className="h-3 w-3 text-brand-600" aria-hidden="true" />
              Adaptive Multi-Strategy Retrieval
            </span>

            <h1 className="mt-6 text-4xl font-bold tracking-tight text-ink-950 sm:text-6xl">RAGX</h1>
            <p className="mt-3 text-lg font-medium text-ink-700 sm:text-2xl">
              Adaptive Multi-Strategy Research Intelligence
            </p>
            <p className="mt-5 text-balance text-base text-ink-600 sm:text-lg">
              One query. Multiple retrieval strategies. Verified answers.
            </p>

            <p className="mx-auto mt-5 max-w-2xl text-sm leading-relaxed text-ink-500 sm:text-base">
              RAGX is not a fixed RAG pipeline. It analyses each question first, decides how to
              retrieve before deciding what to answer, and verifies every claim against the evidence
              it actually found.
            </p>

            <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Link
                to="/research"
                className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 sm:w-auto"
              >
                Try Research Assistant
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
              <a
                href="#architecture"
                className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-white px-5 py-2.5 text-sm font-medium text-ink-800 ring-1 ring-inset ring-ink-200 transition hover:bg-ink-50 sm:w-auto"
              >
                Explore Architecture
              </a>
            </div>
          </div>

          {/* Pipeline strip */}
          <div className="mx-auto mt-14 max-w-5xl">
            <div className="flex flex-wrap items-center justify-center gap-x-1 gap-y-3">
              {PIPELINE.map((stage, index) => (
                <div key={stage.label} className="flex items-center gap-1">
                  <div className="flex items-center gap-2 rounded-lg border border-ink-200 bg-white px-3 py-2 shadow-card">
                    <stage.icon className="h-3.5 w-3.5 text-brand-600" aria-hidden="true" />
                    <span className="whitespace-nowrap text-xs font-medium text-ink-700">
                      {stage.label}
                    </span>
                  </div>
                  {index < PIPELINE.length - 1 ? (
                    <ArrowRight className="h-3 w-3 shrink-0 text-ink-300" aria-hidden="true" />
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------- architecture */}
      <section id="architecture" className="border-b border-ink-100 bg-ink-50/60 py-16 sm:py-20">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <div className="max-w-2xl">
            <h2 className="text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">
              It decides how to retrieve before deciding what to answer
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-ink-600 sm:text-base">
              Different questions fail for different reasons. Dense search misses exact model names;
              keyword search misses paraphrase; neither can answer a question whose answer spans two
              documents. RAGX measures which of those pressures a query actually has, and pays only
              for the strategies that address them.
            </p>
          </div>

          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {CAPABILITIES.map((capability) => (
              <div key={capability.title} className="rounded-xl border border-ink-200 bg-white p-5">
                <span className="inline-flex rounded-lg bg-brand-50 p-2 text-brand-600">
                  <capability.icon className="h-4 w-4" aria-hidden="true" />
                </span>
                <h3 className="mt-3 text-sm font-semibold text-ink-900">{capability.title}</h3>
                <p className="mt-1.5 text-xs leading-relaxed text-ink-600">{capability.body}</p>
              </div>
            ))}
          </div>

          {/* Routing examples */}
          <div className="mt-10 overflow-hidden rounded-xl border border-ink-200 bg-white">
            <div className="border-b border-ink-100 px-5 py-3.5">
              <h3 className="text-sm font-semibold text-ink-900">How routing decisions look</h3>
              <p className="mt-0.5 text-xs text-ink-500">
                Real decisions the router makes — a simple lookup never triggers the expensive path.
              </p>
            </div>
            <div className="divide-y divide-ink-100">
              {[
                { q: 'What optimizer was used for training?', s: ['naive'], why: 'Direct single-fact lookup.' },
                {
                  q: 'What mAP does the model reach on NEU-DET?',
                  s: ['hybrid'],
                  why: 'Exact identifiers — BM25 weighted above dense search.',
                },
                {
                  q: 'What does Figure 3 show about accuracy?',
                  s: ['multimodal', 'corrective'],
                  why: 'Figure evidence required; answer must be verifiable.',
                },
                {
                  q: 'How does this method relate to the work it builds on?',
                  s: ['graph', 'hybrid'],
                  why: 'Relationship traversal plus exact entity matching.',
                },
                {
                  q: 'What are the limitations, what causes each, and how could they be fixed?',
                  s: ['agentic', 'graph', 'corrective'],
                  why: 'Multi-hop and decomposable — planned step by step.',
                },
              ].map((row) => (
                <div key={row.q} className="flex flex-col gap-2 px-5 py-3.5 sm:flex-row sm:items-center sm:gap-4">
                  <p className="min-w-0 flex-1 text-sm text-ink-700">
                    <span className="text-ink-400">“</span>
                    {row.q}
                    <span className="text-ink-400">”</span>
                  </p>
                  <div className="flex shrink-0 flex-wrap items-center gap-1.5">
                    {row.s.map((key) => (
                      <span
                        key={key}
                        className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium"
                        style={{
                          backgroundColor: `${STRATEGIES[key].color}14`,
                          color: STRATEGIES[key].color,
                        }}
                      >
                        <span
                          className="h-1.5 w-1.5 rounded-full"
                          style={{ backgroundColor: STRATEGIES[key].color }}
                          aria-hidden="true"
                        />
                        {STRATEGIES[key].short}
                      </span>
                    ))}
                  </div>
                  <p className="w-full shrink-0 text-xs text-ink-500 sm:w-64">{row.why}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------- strategies */}
      <section id="strategies" className="border-b border-ink-100 py-16 sm:py-20">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <div className="max-w-2xl">
            <h2 className="text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">
              Eight retrieval strategies, composed per query
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-ink-600 sm:text-base">
              Each is a modular component behind one interface, so the router can call any of them
              independently or fuse several. It never runs all eight.
            </p>
          </div>

          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {STRATEGY_ORDER.filter((name) => name !== 'ragx').map((name) => (
              <StrategyCard key={name} name={name} />
            ))}
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------------- llm */}
      <section className="border-b border-ink-100 bg-ink-50/60 py-16 sm:py-20">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <div className="grid gap-8 lg:grid-cols-2 lg:gap-12">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">
                Cloud LLM architecture
              </h2>
              <p className="mt-3 text-sm leading-relaxed text-ink-600">
                Business logic never touches a provider SDK. An internal gateway selects a provider
                per call, retries transient failures, fails over to the secondary provider, and
                records provider, model, latency, token usage and estimated cost for every request.
              </p>

              <div className="mt-6 space-y-3">
                <div className="flex items-start gap-3 rounded-xl border border-ink-200 bg-white p-4">
                  <Cloud className="mt-0.5 h-4 w-4 shrink-0 text-brand-600" aria-hidden="true" />
                  <div>
                    <p className="text-sm font-semibold text-ink-900">Gemini API</p>
                    <p className="mt-0.5 text-xs text-ink-600">
                      Primary reasoning, synthesis, multimodal understanding and embeddings.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3 rounded-xl border border-ink-200 bg-white p-4">
                  <Cloud className="mt-0.5 h-4 w-4 shrink-0 text-violet-600" aria-hidden="true" />
                  <div>
                    <p className="text-sm font-semibold text-ink-900">Groq API</p>
                    <p className="mt-0.5 text-xs text-ink-600">
                      Latency-sensitive steps — query analysis, relevance grading, claim extraction —
                      and fallback generation.
                    </p>
                  </div>
                </div>
              </div>

              <p className="mt-4 text-xs leading-relaxed text-ink-500">
                API keys live only in the backend environment and are never exposed to the browser.
                All model calls go through the FastAPI backend.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">
                Storage architecture
              </h2>
              <p className="mt-3 text-sm leading-relaxed text-ink-600">
                Four separate layers, each doing what it is good at.
              </p>

              <div className="mt-6 space-y-2.5">
                {[
                  { name: 'Qdrant', role: 'Dense and multimodal embeddings', icon: Boxes },
                  { name: 'Neo4j', role: 'Entities, relations and multi-hop traversal', icon: GitBranch },
                  { name: 'PostgreSQL', role: 'Documents, metadata, query history, evaluations', icon: Layers },
                  { name: 'Object storage', role: 'Original files and extracted figures', icon: ImageIcon },
                ].map((layer) => (
                  <div
                    key={layer.name}
                    className="flex items-center gap-3 rounded-xl border border-ink-200 bg-white p-3.5"
                  >
                    <layer.icon className="h-4 w-4 shrink-0 text-ink-400" aria-hidden="true" />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-ink-900">{layer.name}</p>
                      <p className="truncate text-xs text-ink-500">{layer.role}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------------- cta */}
      <section className="py-16 sm:py-20">
        <div className="mx-auto max-w-3xl px-4 text-center sm:px-6">
          <h2 className="text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">
            Upload a paper and watch it route
          </h2>
          <p className="mt-3 text-sm leading-relaxed text-ink-600">
            Index a document, ask a simple question and a complex one, and compare the routing
            decisions in the “Why this answer?” panel.
          </p>
          <div className="mt-7 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Link
              to="/knowledge-base"
              className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 sm:w-auto"
            >
              Upload documents
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
            <Link
              to="/evaluation"
              className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-white px-5 py-2.5 text-sm font-medium text-ink-800 ring-1 ring-inset ring-ink-200 transition hover:bg-ink-50 sm:w-auto"
            >
              <FlaskConical className="h-4 w-4" aria-hidden="true" />
              Run the benchmark
            </Link>
          </div>
        </div>
      </section>

      <footer className="border-t border-ink-100 py-8">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 text-xs text-ink-500 sm:flex-row sm:px-6">
          <p>RAGX — Adaptive Multi-Strategy Research Intelligence System</p>
          <p>Cloud LLMs only (Gemini · Groq). Answers are grounded and cited, or withheld.</p>
        </div>
      </footer>
    </div>
  );
}
