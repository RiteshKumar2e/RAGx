import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AlertTriangle,
  BarChart3,
  Beaker,
  FlaskConical,
  Info,
  Play,
  RefreshCw,
  Table2,
  Trash2,
} from 'lucide-react';
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  InfoBanner,
  LoadingBlock,
  PageHeader,
} from '../components/common';
import Modal from '../components/common/Modal';
import {
  MetricRadarChart,
  StrategyComparisonChart,
} from '../components/evaluation/Charts';
import { useApi } from '../hooks/useApi';
import { usePolling } from '../hooks/usePolling';
import { useSystem } from '../context/SystemContext';
import { useToast } from '../context/ToastContext';
import { evaluationService } from '../services';
import { BENCHMARK_CATEGORIES, EVAL_METRICS, STRATEGY_ORDER } from '../utils/constants';
import { formatDateTime, formatMetric, formatNumber } from '../utils/format';
import { strategyMeta } from '../utils/strategy';
import cn from '../utils/cn';

const RADAR_METRICS = EVAL_METRICS.filter((metric) =>
  ['recall_at_k', 'precision_at_k', 'mrr', 'faithfulness', 'groundedness', 'citation_accuracy'].includes(
    metric.key,
  ),
);

const HEADLINE_METRICS = ['recall_at_k', 'faithfulness', 'citation_accuracy', 'avg_latency_ms'];

function RunConfigModal({ open, onClose, onSubmit, pending }) {
  const [strategies, setStrategies] = useState(['naive', 'hybrid', 'adaptive', 'ragx']);
  const [categories, setCategories] = useState([]);
  const [limit, setLimit] = useState(8);
  const [k, setK] = useState(8);
  const [judge, setJudge] = useState(true);

  const toggle = (list, setList, value) =>
    setList(list.includes(value) ? list.filter((item) => item !== value) : [...list, value]);

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Run a benchmark experiment"
      description="Each selected strategy answers the same questions under identical conditions."
      footer={
        <>
          <Button variant="secondary" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button
            size="sm"
            icon={Play}
            loading={pending}
            disabled={!strategies.length}
            onClick={() =>
              onSubmit({
                strategies,
                categories: categories.length ? categories : undefined,
                limit: limit || undefined,
                k,
                judgeGeneration: judge,
              })
            }
          >
            Start {strategies.length} run(s)
          </Button>
        </>
      }
    >
      <div className="space-y-5">
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-600">
            Conditions to compare
          </p>
          <div className="flex flex-wrap gap-1.5">
            {STRATEGY_ORDER.map((name) => {
              const meta = strategyMeta(name);
              const active = strategies.includes(name);
              return (
                <button
                  key={name}
                  type="button"
                  onClick={() => toggle(strategies, setStrategies, name)}
                  className={cn(
                    'rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset transition',
                    active ? 'text-white' : 'bg-white text-ink-600 ring-ink-200 hover:bg-ink-50',
                  )}
                  style={active ? { backgroundColor: meta.color, '--tw-ring-color': meta.color } : undefined}
                >
                  {meta.label}
                </button>
              );
            })}
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-ink-500">
            A named strategy pins that strategy and bypasses the router. <strong>Adaptive</strong>{' '}
            uses the router with verification disabled; <strong>RAGX</strong> is the full pipeline —
            the difference between them isolates the contribution of the verification layer.
          </p>
        </div>

        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-600">
            Question categories
          </p>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(BENCHMARK_CATEGORIES).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => toggle(categories, setCategories, key)}
                className={cn(
                  'rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset transition',
                  categories.includes(key)
                    ? 'bg-brand-600 text-white ring-brand-600'
                    : 'bg-white text-ink-600 ring-ink-200 hover:bg-ink-50',
                )}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="mt-1.5 text-[11px] text-ink-500">
            {categories.length ? `${categories.length} selected.` : 'All categories will be used.'}
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="eval-limit" className="mb-1 block text-xs font-semibold uppercase tracking-wide text-ink-600">
              Questions per run
            </label>
            <input
              id="eval-limit"
              type="number"
              min={1}
              max={50}
              value={limit}
              onChange={(event) => setLimit(Number(event.target.value))}
              className="w-full rounded-lg border border-ink-200 px-2.5 py-1.5 text-sm focus:border-brand-400 focus:outline-none"
            />
            <p className="mt-1 text-[11px] text-ink-500">
              Each question costs several API calls per strategy. Start small.
            </p>
          </div>

          <div>
            <label htmlFor="eval-k" className="mb-1 block text-xs font-semibold uppercase tracking-wide text-ink-600">
              K (top-K retrieved)
            </label>
            <input
              id="eval-k"
              type="number"
              min={1}
              max={30}
              value={k}
              onChange={(event) => setK(Number(event.target.value))}
              className="w-full rounded-lg border border-ink-200 px-2.5 py-1.5 text-sm focus:border-brand-400 focus:outline-none"
            />
          </div>
        </div>

        <label className="flex cursor-pointer items-start gap-2.5 rounded-lg border border-ink-200 p-3">
          <input
            type="checkbox"
            checked={judge}
            onChange={(event) => setJudge(event.target.checked)}
            className="mt-0.5 h-3.5 w-3.5 rounded border-ink-300 text-brand-600 focus:ring-brand-500"
          />
          <span>
            <span className="block text-xs font-medium text-ink-900">Run LLM judges</span>
            <span className="mt-0.5 block text-[11px] leading-relaxed text-ink-500">
              Computes faithfulness, answer relevance and context relevance, and produces the pooled
              relevance labels that Recall@K and MRR are measured against. Requires a configured
              provider and adds API cost.
            </span>
          </span>
        </label>
      </div>
    </Modal>
  );
}

export default function Evaluation() {
  const { hasLlm, usingDevEmbedder } = useSystem();
  const toast = useToast();
  const [configOpen, setConfigOpen] = useState(false);
  const [starting, setStarting] = useState(false);

  const {
    data: comparison,
    error,
    loading,
    refetch: refetchComparison,
  } = useApi(() => evaluationService.comparison(), []);
  const { data: runs, refetch: refetchRuns } = useApi(() => evaluationService.runs({ limit: 30 }), []);
  const { data: benchmark } = useApi(() => evaluationService.benchmark(), []);

  const activeRuns = useMemo(
    () => (runs || []).filter((run) => run.status === 'running'),
    [runs],
  );

  usePolling(
    useCallback(() => {
      refetchRuns();
      refetchComparison();
    }, [refetchRuns, refetchComparison]),
    4000,
    activeRuns.length > 0,
  );

  const start = async (config) => {
    setStarting(true);
    try {
      const result = await evaluationService.run(config);
      toast.success(result.message, { title: 'Evaluation started' });
      (result.warnings || []).forEach((warning) => toast.warning(warning));
      setConfigOpen(false);
      refetchRuns();
    } catch (caught) {
      toast.error(caught.message, { title: 'Could not start the evaluation' });
    } finally {
      setStarting(false);
    }
  };

  const removeRun = async (runId) => {
    try {
      await evaluationService.deleteRun(runId);
      toast.success('Evaluation run deleted.');
      refetchRuns();
      refetchComparison();
    } catch (caught) {
      toast.error(caught.message);
    }
  };

  const rows = comparison?.runs || [];
  const hasData = comparison?.has_data;
  const labelsAvailable = rows.some((run) => run.recall_at_k !== null && run.recall_at_k !== undefined);

  return (
    <>
      <PageHeader
        title="Evaluation"
        description="Compare retrieval strategies on the same questions under identical conditions. Every number here comes from a run you executed."
        action={
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              icon={RefreshCw}
              onClick={() => {
                refetchRuns();
                refetchComparison();
              }}
            >
              Refresh
            </Button>
            <Button size="sm" icon={FlaskConical} onClick={() => setConfigOpen(true)}>
              Run experiment
            </Button>
          </div>
        }
      />

      {/* Methodology caveats that change how results should be read */}
      <div className="mb-6 space-y-2">
        {!hasLlm ? (
          <InfoBanner variant="warning" icon={AlertTriangle} title="No LLM provider configured">
            Retrieval metrics can still be computed, but faithfulness, answer relevance and context
            relevance require a cloud provider — and without one, pooled relevance labels cannot be
            produced, so Recall@K and MRR will be unavailable.
          </InfoBanner>
        ) : null}

        {usingDevEmbedder ? (
          <InfoBanner variant="warning" icon={AlertTriangle} title="Development embedder active">
            Retrieval currently matches text lexically, not semantically. Results measured now are
            not representative of the system's real retrieval quality and must not be reported as
            benchmarks.
          </InfoBanner>
        ) : null}

        {hasData && !labelsAvailable ? (
          <InfoBanner variant="info" icon={Info} title="Retrieval metrics unavailable">
            These runs have no relevance labels, so Recall@K, Precision@K, MRR and nDCG could not be
            computed. Enable “Run LLM judges” to produce pooled relevance labels.
          </InfoBanner>
        ) : null}

        {hasData && labelsAvailable ? (
          <InfoBanner variant="neutral" icon={Info} title="How retrieval metrics were labelled">
            Recall@K, Precision@K, MRR and nDCG are measured against pooled LLM relevance judgements:
            every evaluated strategy contributed its top results to a shared pool, which was judged
            once and applied to all strategies. Recall is therefore relative to that pooled candidate
            set — a passage no strategy retrieved was never judged.
          </InfoBanner>
        ) : null}
      </div>

      {/* In-flight runs */}
      {activeRuns.length ? (
        <Card className="mb-6">
          <CardBody>
            <div className="flex items-center gap-3">
              <RefreshCw className="h-4 w-4 shrink-0 animate-spin text-brand-600" aria-hidden="true" />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-ink-900">
                  {activeRuns.length} evaluation run(s) in progress
                </p>
                <p className="mt-0.5 text-xs text-ink-500">
                  {activeRuns
                    .map(
                      (run) =>
                        `${strategyMeta(run.strategy).label} ${run.completed_count}/${run.question_count}`,
                    )
                    .join(' · ')}
                </p>
              </div>
            </div>
          </CardBody>
        </Card>
      ) : null}

      {error ? (
        <Card>
          <ErrorState error={error} onRetry={refetchComparison} />
        </Card>
      ) : loading && !comparison ? (
        <Card>
          <CardBody>
            <LoadingBlock rows={6} />
          </CardBody>
        </Card>
      ) : !hasData ? (
        <Card>
          <EmptyState
            icon={Beaker}
            title="No evaluation results yet"
            description={
              comparison?.message ||
              'Run a benchmark experiment to populate this page. Nothing here is pre-filled — every figure comes from an experiment you execute.'
            }
            action={
              <Button icon={FlaskConical} onClick={() => setConfigOpen(true)}>
                Run your first experiment
              </Button>
            }
            className="py-14"
          />
        </Card>
      ) : (
        <>
          {/* Headline charts */}
          <div className="grid gap-4 lg:grid-cols-2">
            {HEADLINE_METRICS.map((key) => {
              const metric = EVAL_METRICS.find((item) => item.key === key);
              return (
                <Card key={key}>
                  <CardHeader
                    title={metric.label}
                    description={`${metric.group} metric · ${
                      metric.higherIsBetter === null
                        ? 'context-dependent'
                        : metric.higherIsBetter
                          ? 'higher is better'
                          : 'lower is better'
                    }`}
                    icon={BarChart3}
                  />
                  <CardBody>
                    <StrategyComparisonChart
                      data={rows}
                      metricKey={metric.key}
                      metricLabel={metric.label}
                      formatter={(value) => formatMetric(value, metric.format)}
                    />
                  </CardBody>
                </Card>
              );
            })}
          </div>

          {/* Radar profile */}
          <Card className="mt-4">
            <CardHeader
              title="Metric profile"
              description="Quality metrics on a common 0–1 scale. Shows trade-offs rather than a single winner."
            />
            <CardBody>
              <MetricRadarChart runs={rows} metrics={RADAR_METRICS} />
            </CardBody>
          </Card>

          {/* Full comparison table */}
          <Card className="mt-4">
            <CardHeader
              title="Strategy comparison"
              description="Latest completed run per strategy. Best value per metric is highlighted."
              icon={Table2}
            />
            <div className="overflow-x-auto">
              <table className="w-full min-w-[880px] text-left text-xs">
                <thead>
                  <tr className="border-b border-ink-200 bg-ink-50">
                    <th className="sticky left-0 z-10 bg-ink-50 px-4 py-2.5 font-semibold text-ink-700">
                      Metric
                    </th>
                    {rows.map((run) => (
                      <th key={run.id} className="px-3 py-2.5 text-right font-semibold">
                        <span className="inline-flex items-center gap-1.5">
                          <span
                            className="h-1.5 w-1.5 rounded-full"
                            style={{ backgroundColor: strategyMeta(run.strategy).color }}
                            aria-hidden="true"
                          />
                          {strategyMeta(run.strategy).label}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {['Retrieval', 'Generation', 'System'].map((group) => (
                    <>
                      <tr key={group} className="bg-ink-50/60">
                        <td
                          colSpan={rows.length + 1}
                          className="px-4 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-ink-500"
                        >
                          {group}
                        </td>
                      </tr>
                      {EVAL_METRICS.filter((metric) => metric.group === group).map((metric) => (
                        <tr key={metric.key} className="border-b border-ink-50">
                          <td className="sticky left-0 z-10 bg-white px-4 py-2 text-ink-700">
                            {metric.label}
                          </td>
                          {rows.map((run) => {
                            const value = run[metric.key];
                            const isBest =
                              comparison.best_by_metric?.[metric.key] === run.strategy &&
                              value !== null &&
                              value !== undefined;
                            return (
                              <td
                                key={run.id}
                                className={cn(
                                  'px-3 py-2 text-right tabular-nums',
                                  isBest ? 'font-semibold text-emerald-700' : 'text-ink-700',
                                  value === null || value === undefined ? 'text-ink-300' : '',
                                )}
                                title={
                                  value === null || value === undefined
                                    ? 'Not computed — requires relevance labels or an LLM judge'
                                    : undefined
                                }
                              >
                                {formatMetric(value, metric.format)}
                                {isBest ? ' ★' : ''}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </>
                  ))}
                  <tr className="border-t border-ink-200 bg-ink-50/40">
                    <td className="sticky left-0 z-10 bg-ink-50/40 px-4 py-2 text-ink-500">
                      Questions completed
                    </td>
                    {rows.map((run) => (
                      <td key={run.id} className="px-3 py-2 text-right tabular-nums text-ink-500">
                        {run.completed_count}/{run.question_count}
                        {run.failed_count ? ` (${run.failed_count} failed)` : ''}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="border-t border-ink-100 px-4 py-2.5 text-[11px] text-ink-500">
              ★ marks the best value per metric. A dash means the metric was not computed for that
              run — never that it scored zero.
            </p>
          </Card>
        </>
      )}

      {/* Benchmark + run history */}
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Run history"
            description="Every experiment recorded, newest first."
          />
          {runs?.length ? (
            <ul className="divide-y divide-ink-50">
              {runs.map((run) => (
                <li key={run.id} className="flex items-center gap-3 px-4 py-2.5 sm:px-5">
                  <span
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: strategyMeta(run.strategy).color }}
                    aria-hidden="true"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-ink-900">{run.name}</p>
                    <p className="text-[11px] text-ink-400">
                      {run.dataset} · K={run.k} · {formatDateTime(run.created_at)}
                    </p>
                  </div>
                  <Badge
                    className={
                      run.status === 'completed'
                        ? 'bg-emerald-50 text-emerald-700 ring-emerald-600/20'
                        : run.status === 'failed'
                          ? 'bg-rose-50 text-rose-700 ring-rose-600/20'
                          : 'bg-brand-50 text-brand-700 ring-brand-600/20'
                    }
                  >
                    {run.status === 'running'
                      ? `${run.completed_count}/${run.question_count}`
                      : run.status}
                  </Badge>
                  <button
                    type="button"
                    onClick={() => removeRun(run.id)}
                    className="shrink-0 rounded p-1 text-ink-300 transition hover:bg-rose-50 hover:text-rose-600"
                    aria-label={`Delete run ${run.name}`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState icon={FlaskConical} title="No runs yet" className="py-10" />
          )}
        </Card>

        <Card>
          <CardHeader
            title="Benchmark dataset"
            description={benchmark ? `${benchmark.name} v${benchmark.version}` : 'Loading…'}
          />
          <CardBody>
            {benchmark ? (
              <>
                <p className="mb-3 text-[11px] leading-relaxed text-ink-500">
                  {formatNumber(benchmark.question_count)} questions spanning the query classes the
                  router discriminates between — including adversarial questions where the correct
                  behaviour is abstention.
                </p>
                <ul className="space-y-1">
                  {Object.entries(benchmark.categories || {}).map(([category, count]) => (
                    <li key={category} className="flex items-center justify-between gap-2 text-[11px]">
                      <span className="truncate text-ink-600">
                        {BENCHMARK_CATEGORIES[category] || category}
                      </span>
                      <span className="shrink-0 tabular-nums text-ink-400">{count}</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 border-t border-ink-100 pt-2.5 text-[11px] leading-relaxed text-ink-500">
                  {benchmark.has_relevance_labels
                    ? 'This dataset ships manual relevance labels.'
                    : 'No manual labels — relevance is judged by pooling at run time.'}
                </p>
              </>
            ) : (
              <LoadingBlock rows={4} />
            )}
            <Link
              to="/research"
              className="mt-3 block text-xs font-medium text-brand-600 hover:text-brand-700"
            >
              Try a question interactively →
            </Link>
          </CardBody>
        </Card>
      </div>

      <RunConfigModal
        open={configOpen}
        onClose={() => setConfigOpen(false)}
        onSubmit={start}
        pending={starting}
      />
    </>
  );
}
