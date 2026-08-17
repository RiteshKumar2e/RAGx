import { useState } from 'react';
import {
  Braces,
  ChevronDown,
  Cpu,
  Database,
  Layers,
  RefreshCcw,
  ShieldCheck,
  Workflow,
} from 'lucide-react';
import { COMPLEXITY_STYLES, CONFIDENCE_STYLES, INTENT_LABELS } from '../../utils/constants';
import { formatMs, formatRatio, formatUsd, formatNumber, formatPercent } from '../../utils/format';
import { strategyMeta } from '../../utils/strategy';
import cn from '../../utils/cn';

function Row({ label, value, mono }) {
  return (
    <div className="flex items-start justify-between gap-4 py-1.5 text-xs">
      <dt className="shrink-0 text-ink-500">{label}</dt>
      <dd className={cn('min-w-0 text-right font-medium text-ink-900', mono && 'font-mono')}>{value}</dd>
    </div>
  );
}

function Section({ title, icon: Icon, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-ink-100 last:border-0">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2.5 px-4 py-3 text-left transition hover:bg-ink-50"
        aria-expanded={open}
      >
        <Icon className="h-3.5 w-3.5 shrink-0 text-ink-400" aria-hidden="true" />
        <span className="flex-1 text-xs font-semibold uppercase tracking-wide text-ink-700">
          {title}
        </span>
        <ChevronDown
          className={cn('h-3.5 w-3.5 shrink-0 text-ink-400 transition-transform', open && 'rotate-180')}
          aria-hidden="true"
        />
      </button>
      {open ? <div className="animate-fade-in px-4 pb-4">{children}</div> : null}
    </div>
  );
}

/**
 * The full explainability panel.
 *
 * Everything shown here is recorded during the query run — routing rules,
 * per-strategy scores, corrective events, token accounting and per-claim
 * verdicts. Nothing is inferred at render time.
 */
export default function WhyThisAnswerPanel({ why }) {
  if (!why) return null;

  const { analysis, routing, retrieval, generation, verification, stage_latency_ms: stages } = why;
  const confidence = verification?.confidence;
  const citations = verification?.citations;
  const corrective = retrieval?.diagnostics?.corrective;
  const agentic = retrieval?.diagnostics?.agentic;
  const perStrategy = retrieval?.diagnostics?.per_strategy;

  return (
    <div className="overflow-hidden rounded-xl border border-ink-200 bg-white">
      <div className="border-b border-ink-100 bg-ink-50/60 px-4 py-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-700">
          Why this answer?
        </h3>
        <p className="mt-0.5 text-[11px] text-ink-500">
          Every decision the system made, recorded during this run.
        </p>
      </div>

      {/* ------------------------------------------------------ query analysis */}
      <Section title="Query analysis" icon={Braces} defaultOpen>
        <dl className="divide-y divide-ink-50">
          <Row label="Intent" value={INTENT_LABELS[analysis?.intent] || analysis?.intent || '—'} />
          <Row
            label="Complexity"
            value={
              <span
                className={cn(
                  'rounded px-1.5 py-0.5 text-[11px] ring-1 ring-inset',
                  COMPLEXITY_STYLES[analysis?.complexity]?.className || 'bg-ink-100 text-ink-700',
                )}
              >
                {COMPLEXITY_STYLES[analysis?.complexity]?.label || analysis?.complexity || '—'}
              </span>
            }
          />
          <Row label="Multi-hop required" value={analysis?.multi_hop ? 'Yes' : 'No'} />
          <Row label="Semantic requirement" value={formatRatio(analysis?.semantic_requirement, 2)} />
          <Row label="Keyword requirement" value={formatRatio(analysis?.keyword_requirement, 2)} />
          <Row
            label="Modality"
            value={
              [
                analysis?.requires_visual && 'visual',
                analysis?.requires_tabular && 'tabular',
              ]
                .filter(Boolean)
                .join(', ') || 'text'
            }
          />
          <Row label="Documents expected" value={analysis?.expected_documents ?? '—'} />
          <Row label="Analysis source" value={analysis?.source || 'heuristic'} />
        </dl>

        {analysis?.entities?.length ? (
          <div className="mt-3">
            <p className="mb-1.5 text-[11px] font-medium text-ink-500">Entities detected</p>
            <div className="flex flex-wrap gap-1">
              {analysis.entities.slice(0, 10).map((entity) => (
                <span
                  key={entity}
                  className="rounded bg-ink-100 px-1.5 py-0.5 font-mono text-[11px] text-ink-700"
                >
                  {entity}
                </span>
              ))}
            </div>
          </div>
        ) : null}

        {analysis?.sub_questions?.length ? (
          <div className="mt-3">
            <p className="mb-1.5 text-[11px] font-medium text-ink-500">Decomposed sub-questions</p>
            <ol className="list-decimal space-y-1 pl-4">
              {analysis.sub_questions.map((question) => (
                <li key={question} className="text-xs text-ink-600">
                  {question}
                </li>
              ))}
            </ol>
          </div>
        ) : null}

        {analysis?.reasoning ? (
          <p className="mt-3 rounded-lg bg-ink-50 p-2.5 text-xs italic leading-relaxed text-ink-600">
            {analysis.reasoning}
          </p>
        ) : null}
      </Section>

      {/* -------------------------------------------------------------- routing */}
      <Section title="Strategy selection" icon={Workflow} defaultOpen>
        <div className="mb-3 flex flex-wrap gap-1.5">
          {(routing?.strategies || []).map((name) => {
            const meta = strategyMeta(name);
            return (
              <span
                key={name}
                className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium"
                style={{ backgroundColor: `${meta.color}14`, color: meta.color }}
              >
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{ backgroundColor: meta.color }}
                  aria-hidden="true"
                />
                {meta.label}
              </span>
            );
          })}
        </div>

        {routing?.reason ? (
          <p className="mb-3 text-xs leading-relaxed text-ink-600">{routing.reason}</p>
        ) : null}

        {routing?.rules_fired?.length ? (
          <ul className="space-y-2">
            {routing.rules_fired.map((rule) => (
              <li key={rule.rule} className="rounded-lg bg-ink-50 p-2.5">
                <p className="font-mono text-[11px] text-brand-700">{rule.rule}</p>
                <p className="mt-0.5 text-xs leading-relaxed text-ink-600">{rule.reason}</p>
              </li>
            ))}
          </ul>
        ) : null}

        <dl className="mt-3 divide-y divide-ink-50">
          <Row label="Mode" value={routing?.mode || '—'} />
          <Row label="Top-K" value={routing?.config?.top_k ?? '—'} />
          <Row label="Candidate pool" value={routing?.config?.candidate_pool ?? '—'} />
          {routing?.config?.dense_weight != null ? (
            <Row
              label="Dense / sparse weight"
              value={`${formatRatio(routing.config.dense_weight, 2)} / ${formatRatio(routing.config.sparse_weight, 2)}`}
            />
          ) : null}
        </dl>
      </Section>

      {/* ------------------------------------------------------------ retrieval */}
      <Section title="Retrieval" icon={Database} defaultOpen>
        <dl className="divide-y divide-ink-50">
          <Row label="Chunks retrieved" value={retrieval?.chunks_retrieved ?? 0} />
          <Row label="Documents used" value={retrieval?.documents_used?.length ?? 0} />
          <Row label="Retrieval calls" value={retrieval?.retrieval_calls ?? 0} />
          <Row label="Reranking performed" value={retrieval?.reranked ? 'Yes' : 'No'} />
          <Row label="Top relevance score" value={formatRatio(retrieval?.top_score, 3)} />
          <Row label="Mean relevance" value={formatRatio(retrieval?.mean_score, 3)} />
          <Row label="Retrieval latency" value={formatMs(retrieval?.latency_ms)} />
          <Row
            label="Corrective retrieval"
            value={
              retrieval?.corrective_triggered ? (
                <span className="inline-flex items-center gap-1 rounded bg-amber-50 px-1.5 py-0.5 text-[11px] text-amber-700 ring-1 ring-inset ring-amber-600/20">
                  <RefreshCcw className="h-3 w-3" aria-hidden="true" />
                  Triggered · {retrieval.corrective_rounds} round(s)
                </span>
              ) : (
                'Not triggered'
              )
            }
          />
          <Row label="Agentic workflow" value={retrieval?.agentic_used ? 'Used' : 'Not used'} />
        </dl>

        {/* Per-strategy contribution */}
        {perStrategy && Object.keys(perStrategy).length ? (
          <div className="mt-3">
            <p className="mb-1.5 text-[11px] font-medium text-ink-500">Per-strategy results</p>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-ink-100 text-[11px] text-ink-500">
                    <th className="py-1.5 pr-3 font-medium">Strategy</th>
                    <th className="py-1.5 pr-3 text-right font-medium">Chunks</th>
                    <th className="py-1.5 pr-3 text-right font-medium">Top score</th>
                    <th className="py-1.5 text-right font-medium">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(perStrategy).map(([name, detail]) => (
                    <tr key={name} className="border-b border-ink-50 last:border-0">
                      <td className="py-1.5 pr-3">
                        <span className="inline-flex items-center gap-1.5">
                          <span
                            className="h-1.5 w-1.5 rounded-full"
                            style={{ backgroundColor: strategyMeta(name).color }}
                            aria-hidden="true"
                          />
                          {strategyMeta(name).label}
                        </span>
                      </td>
                      <td className="py-1.5 pr-3 text-right tabular-nums">{detail.chunks ?? 0}</td>
                      <td className="py-1.5 pr-3 text-right tabular-nums">
                        {formatRatio(detail.top_score, 3)}
                      </td>
                      <td className="py-1.5 text-right tabular-nums">{formatMs(detail.latency_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        {/* Corrective detail */}
        {corrective?.triggered ? (
          <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-amber-800">
              Corrective retrieval
            </p>
            <p className="mt-1 text-xs text-amber-900">
              Diagnosis: <span className="font-medium">{corrective.final_grade?.diagnosis}</span> —{' '}
              {corrective.final_grade?.diagnosis_hint}
            </p>
            {corrective.actions?.length ? (
              <p className="mt-1 text-xs text-amber-900">
                Actions: <span className="font-mono">{corrective.actions.join(' → ')}</span>
              </p>
            ) : null}
            {corrective.history?.slice(1).map((round) => (
              <div key={round.round} className="mt-2 border-t border-amber-200 pt-2">
                <p className="text-[11px] font-medium text-amber-800">Round {round.round}</p>
                {round.rewrites?.map((rewrite) => (
                  <p key={rewrite} className="mt-0.5 text-xs italic text-amber-800">
                    “{rewrite}”
                  </p>
                ))}
                <p className="mt-0.5 text-[11px] text-amber-700">
                  Quality after: {formatRatio(round.grade?.overall, 2)}
                </p>
              </div>
            ))}
          </div>
        ) : null}

        {/* Agentic plan */}
        {agentic?.plan?.length ? (
          <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 p-3">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-rose-800">
              Agentic plan · {agentic.executed_steps} step(s)
            </p>
            <ol className="mt-2 space-y-2">
              {agentic.plan.map((step) => (
                <li key={step.step} className="text-xs">
                  <p className="font-medium text-rose-900">
                    {step.step}. {step.sub_question}
                  </p>
                  <p className="mt-0.5 text-[11px] text-rose-700">
                    tool: <span className="font-mono">{step.tool}</span> · {step.chunks_found} passages ·
                    quality {formatRatio(step.evidence_quality, 2)} · {formatMs(step.latency_ms)}
                  </p>
                  {step.reason ? (
                    <p className="mt-0.5 text-[11px] italic text-rose-700">{step.reason}</p>
                  ) : null}
                </li>
              ))}
            </ol>
            {agentic.reflection?.reason ? (
              <p className="mt-2 border-t border-rose-200 pt-2 text-[11px] text-rose-800">
                <span className="font-medium">Reflection:</span> {agentic.reflection.reason}
              </p>
            ) : null}
          </div>
        ) : null}

        {retrieval?.notes?.length ? (
          <ul className="mt-3 space-y-1">
            {retrieval.notes.map((note) => (
              <li key={note} className="text-[11px] italic leading-relaxed text-ink-500">
                • {note}
              </li>
            ))}
          </ul>
        ) : null}
      </Section>

      {/* ----------------------------------------------------------- generation */}
      <Section title="Generation" icon={Cpu}>
        <dl className="divide-y divide-ink-50">
          <Row
            label="LLM provider"
            value={
              generation?.provider ? (
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" aria-hidden="true" />
                  <span className="capitalize">{generation.provider}</span>
                </span>
              ) : (
                'Not available'
              )
            }
          />
          <Row label="Model" value={generation?.model || '—'} mono />
          <Row label="Fallback used" value={generation?.fallback_used ? 'Yes' : 'No'} />
          <Row label="Generation latency" value={formatMs(generation?.latency_ms)} />
          <Row label="Prompt tokens" value={formatNumber(generation?.prompt_tokens)} />
          <Row label="Completion tokens" value={formatNumber(generation?.completion_tokens)} />
          <Row label="Estimated cost" value={formatUsd(generation?.estimated_cost_usd)} />
          {generation?.multimodal ? (
            <Row label="Images sent to model" value={generation.images_sent} />
          ) : null}
        </dl>
      </Section>

      {/* --------------------------------------------------------- verification */}
      <Section title="Verification" icon={ShieldCheck} defaultOpen>
        <dl className="divide-y divide-ink-50">
          <Row
            label="Confidence"
            value={
              <span
                className={cn(
                  'rounded px-1.5 py-0.5 text-[11px] ring-1 ring-inset',
                  CONFIDENCE_STYLES[confidence?.label]?.className || 'bg-ink-100 text-ink-700',
                )}
              >
                {CONFIDENCE_STYLES[confidence?.label]?.label || confidence?.label || '—'}
                {confidence?.label !== 'abstained' ? ` · ${formatRatio(confidence?.score, 2)}` : ''}
              </span>
            }
          />
          <Row
            label="Claims detected"
            value={verification?.claims_total ?? 0}
          />
          <Row
            label="Claims verified"
            value={`${verification?.claims_supported ?? 0} of ${verification?.claims_total ?? 0}`}
          />
          {verification?.claims_contradicted ? (
            <Row label="Claims contradicted" value={verification.claims_contradicted} />
          ) : null}
          <Row label="Citation coverage" value={formatPercent(citations?.coverage)} />
          <Row label="Citation accuracy" value={formatPercent(citations?.citation_accuracy)} />
          {citations?.hallucinated_citations ? (
            <Row
              label="Invalid citations removed"
              value={<span className="text-rose-600">{citations.hallucinated_citations}</span>}
            />
          ) : null}
          <Row label="Verification latency" value={formatMs(verification?.latency_ms)} />
        </dl>

        {/* Confidence components — show how the score was built */}
        {confidence?.components && Object.keys(confidence.components).length ? (
          <div className="mt-3">
            <p className="mb-1.5 text-[11px] font-medium text-ink-500">Confidence components</p>
            <div className="space-y-1.5">
              {Object.entries(confidence.components).map(([key, value]) => (
                <div key={key} className="flex items-center gap-2">
                  <span className="w-36 shrink-0 truncate text-[11px] capitalize text-ink-500">
                    {key.replace(/_/g, ' ')}
                  </span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-100">
                    <div
                      className="h-full rounded-full bg-brand-500"
                      style={{ width: `${Math.min(100, (value || 0) * 100)}%` }}
                    />
                  </div>
                  <span className="w-9 shrink-0 text-right font-mono text-[11px] tabular-nums text-ink-600">
                    {formatRatio(value, 2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {confidence?.penalties && Object.keys(confidence.penalties).length ? (
          <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 p-2.5">
            <p className="text-[11px] font-medium text-rose-800">Penalties applied</p>
            <ul className="mt-1 space-y-0.5">
              {Object.entries(confidence.penalties).map(([key, value]) => (
                <li key={key} className="text-[11px] capitalize text-rose-700">
                  {key.replace(/_/g, ' ')}: −{formatRatio(value, 2)}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {confidence?.rationale?.length ? (
          <ul className="mt-3 space-y-1">
            {confidence.rationale.map((line) => (
              <li key={line} className="text-[11px] leading-relaxed text-ink-600">
                • {line}
              </li>
            ))}
          </ul>
        ) : null}

        {/* Per-claim verdicts */}
        {verification?.claim_verdicts?.length ? (
          <div className="mt-3">
            <p className="mb-1.5 text-[11px] font-medium text-ink-500">Claim verdicts</p>
            <ul className="space-y-1.5">
              {verification.claim_verdicts.map((verdict) => {
                const tone =
                  {
                    supported: 'border-emerald-200 bg-emerald-50 text-emerald-800',
                    partially_supported: 'border-amber-200 bg-amber-50 text-amber-800',
                    unsupported: 'border-rose-200 bg-rose-50 text-rose-800',
                    contradicted: 'border-rose-300 bg-rose-100 text-rose-900',
                  }[verdict.verdict] || 'border-ink-200 bg-ink-50 text-ink-700';

                return (
                  <li key={verdict.claim_index} className={cn('rounded-lg border p-2.5', tone)}>
                    <p className="text-xs leading-relaxed">{verdict.claim}</p>
                    <p className="mt-1 text-[11px] opacity-80">
                      {verdict.verdict.replace(/_/g, ' ')} · support{' '}
                      {formatRatio(verdict.support_score, 2)}
                      {verdict.numeric_consistent === false ? ' · numeric mismatch' : ''}
                    </p>
                  </li>
                );
              })}
            </ul>
          </div>
        ) : null}

        {verification?.notes?.length ? (
          <ul className="mt-3 space-y-1">
            {verification.notes.map((note) => (
              <li key={note} className="text-[11px] italic leading-relaxed text-ink-500">
                • {note}
              </li>
            ))}
          </ul>
        ) : null}
      </Section>

      {/* --------------------------------------------------------- stage timing */}
      {stages && Object.keys(stages).length ? (
        <Section title="Stage timing" icon={Layers}>
          <dl className="divide-y divide-ink-50">
            {Object.entries(stages)
              .sort((a, b) => b[1] - a[1])
              .map(([stage, ms]) => (
                <Row key={stage} label={stage.replace(/[:_]/g, ' ')} value={formatMs(ms)} />
              ))}
          </dl>
        </Section>
      ) : null}
    </div>
  );
}
