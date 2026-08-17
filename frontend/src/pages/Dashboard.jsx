import { Link } from 'react-router-dom';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Coins,
  FileStack,
  FlaskConical,
  Library,
  MessageSquareText,
  Network,
  RefreshCw,
  ShieldCheck,
  Workflow,
} from 'lucide-react';
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  InfoBanner,
  LoadingCards,
  PageHeader,
  StatTile,
} from '../components/common';
import {
  DistributionChart,
  DonutChart,
  TimeseriesChart,
} from '../components/evaluation/Charts';
import { useApi } from '../hooks/useApi';
import { useSystem } from '../context/SystemContext';
import { systemService } from '../services';
import { CONFIDENCE_STYLES } from '../utils/constants';
import { formatCompact, formatMs, formatNumber, formatPercent, formatRatio, formatRelativeTime, formatUsd, truncate } from '../utils/format';
import { strategyMeta } from '../utils/strategy';
import cn from '../utils/cn';

function HealthRow({ component }) {
  const healthy = component.healthy;
  const Icon = healthy === false ? AlertTriangle : CheckCircle2;
  const tone =
    healthy === false ? 'text-rose-600' : healthy === true ? 'text-emerald-600' : 'text-ink-300';

  return (
    <li className="flex items-center gap-2.5 py-1.5">
      <Icon className={cn('h-3.5 w-3.5 shrink-0', tone)} aria-hidden="true" />
      <span className="flex-1 truncate text-xs capitalize text-ink-700">
        {component.name.replace(/_/g, ' ')}
      </span>
      <span className="shrink-0 text-[11px] text-ink-500">{component.status}</span>
    </li>
  );
}

export default function Dashboard() {
  const { health, warnings, hasLlm, reload: reloadSystem } = useSystem();
  const { data, error, loading, refetch } = useApi(() => systemService.analytics(30), []);

  const stats = data?.stats;
  const hasQueries = (stats?.total_queries || 0) > 0;
  const hasDocuments = (stats?.total_documents || 0) > 0;

  const refreshAll = () => {
    refetch();
    reloadSystem();
  };

  if (error && !data) {
    return (
      <>
        <PageHeader title="Dashboard" description="System overview and retrieval analytics." />
        <Card>
          <ErrorState error={error} onRetry={refetch} />
        </Card>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Documents indexed, queries executed, routing behaviour and retrieval performance — all measured from real activity."
        action={
          <Button variant="secondary" size="sm" icon={RefreshCw} onClick={refreshAll} loading={loading}>
            Refresh
          </Button>
        }
      />

      {/* Operator warnings that change how numbers should be read. */}
      {warnings?.length ? (
        <div className="mb-6 space-y-2">
          {warnings.slice(0, 2).map((warning) => (
            <InfoBanner key={warning} variant="warning" icon={AlertTriangle}>
              {warning}
            </InfoBanner>
          ))}
        </div>
      ) : null}

      {/* ------------------------------------------------------------ stats */}
      {loading && !data ? (
        <LoadingCards count={4} />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatTile
            label="Documents"
            value={formatNumber(stats?.total_documents)}
            hint={`${formatNumber(stats?.indexed_documents)} indexed · ${formatNumber(stats?.total_chunks)} chunks`}
            icon={Library}
            tone="brand"
          />
          <StatTile
            label="Queries"
            value={formatNumber(stats?.total_queries)}
            hint={`${formatNumber(stats?.queries_last_7_days)} in the last 7 days`}
            icon={MessageSquareText}
            tone="violet"
          />
          <StatTile
            label="Avg retrieval latency"
            value={formatMs(stats?.avg_retrieval_latency_ms)}
            hint={`End to end ${formatMs(stats?.avg_total_latency_ms)}`}
            icon={Clock}
            tone="amber"
          />
          <StatTile
            label="Avg confidence"
            value={hasQueries ? formatRatio(stats?.avg_confidence, 2) : '—'}
            hint={
              hasQueries
                ? `${formatPercent(stats?.abstention_rate)} abstained`
                : 'No queries executed yet'
            }
            icon={ShieldCheck}
            tone="emerald"
          />
        </div>
      )}

      {/* Secondary stats */}
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Most-used strategy"
          value={
            stats?.most_used_strategy ? strategyMeta(stats.most_used_strategy).label : '—'
          }
          hint={hasQueries ? 'Chosen by the adaptive router' : 'Run a query to populate'}
          icon={Workflow}
          tone="ink"
          loading={loading && !data}
        />
        <StatTile
          label="Graph"
          value={formatCompact(stats?.total_entities)}
          hint={`${formatCompact(stats?.total_relations)} relations extracted`}
          icon={Network}
          tone="emerald"
          loading={loading && !data}
        />
        <StatTile
          label="Corrective retrievals"
          value={hasQueries ? formatPercent(stats?.corrective_rate) : '—'}
          hint="Queries where retrieval was repaired"
          icon={RefreshCw}
          tone="amber"
          loading={loading && !data}
        />
        <StatTile
          label="Estimated API cost"
          value={formatUsd(stats?.estimated_cost_usd)}
          hint={`${formatCompact(stats?.total_tokens)} tokens total`}
          icon={Coins}
          tone="rose"
          loading={loading && !data}
        />
      </div>

      {/* --------------------------------------------------------- charts */}
      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Queries over time"
            description="Daily query volume over the last 30 days."
            icon={Activity}
          />
          <CardBody>
            <TimeseriesChart data={data?.queries_over_time} label="queries" />
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="RAG strategy usage"
            description="How often the router selected each strategy."
            icon={Workflow}
          />
          <CardBody>
            <DistributionChart data={data?.strategy_usage} useStrategyColors horizontal />
          </CardBody>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader title="Query complexity" description="Distribution of analysed complexity." />
          <CardBody>
            <DistributionChart data={data?.complexity_distribution} height={200} />
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Confidence distribution" description="Verified confidence per answer." />
          <CardBody>
            <DistributionChart data={data?.confidence_distribution} height={200} />
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Retrieval latency" description="Daily average end-to-end latency." />
          <CardBody>
            <TimeseriesChart
              data={data?.latency_over_time}
              color="#f59e0b"
              height={200}
              label="latency"
              formatter={(value) => formatMs(value)}
            />
          </CardBody>
        </Card>
      </div>

      {/* -------------------------------------------- recent + health + intent */}
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Recent research queries"
            description="The most recent runs, with the strategies the router chose."
            icon={MessageSquareText}
            action={
              <Link
                to="/research"
                className="text-xs font-medium text-brand-600 transition hover:text-brand-700"
              >
                Open assistant →
              </Link>
            }
          />
          {data?.recent_queries?.length ? (
            <ul className="divide-y divide-ink-50">
              {data.recent_queries.map((query) => (
                <li key={query.id} className="px-4 py-3 sm:px-5">
                  <div className="flex items-start justify-between gap-3">
                    <p className="min-w-0 flex-1 text-sm text-ink-800">
                      {truncate(query.question, 110)}
                    </p>
                    <span
                      className={cn(
                        'shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ring-1 ring-inset',
                        CONFIDENCE_STYLES[
                          query.insufficient_evidence ? 'abstained' : query.confidence_label
                        ]?.className || 'bg-ink-100 text-ink-600 ring-ink-600/10',
                      )}
                    >
                      {query.insufficient_evidence
                        ? 'abstained'
                        : `${formatRatio(query.confidence, 2)}`}
                    </span>
                  </div>
                  <div className="mt-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1">
                    {(query.strategies || []).map((name) => {
                      const meta = strategyMeta(name);
                      return (
                        <span
                          key={name}
                          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium"
                          style={{ backgroundColor: `${meta.color}14`, color: meta.color }}
                        >
                          {meta.short}
                        </span>
                      );
                    })}
                    <span className="text-[11px] text-ink-400">
                      {query.chunks_retrieved} passages · {formatMs(query.total_latency_ms)}
                      {query.corrective_triggered ? ' · corrective' : ''}
                    </span>
                    <span className="ml-auto text-[11px] text-ink-400">
                      {formatRelativeTime(query.created_at)}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState
              icon={MessageSquareText}
              title="No queries yet"
              description={
                hasDocuments
                  ? 'Ask a question in the Research Assistant and it will appear here.'
                  : 'Upload a document first, then ask a question.'
              }
              action={
                <Link
                  to={hasDocuments ? '/research' : '/knowledge-base'}
                  className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-3.5 py-2 text-sm font-medium text-white transition hover:bg-brand-700"
                >
                  {hasDocuments ? 'Ask a question' : 'Upload documents'}
                </Link>
              }
              className="py-10"
            />
          )}
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader title="System health" icon={Activity} />
            <CardBody>
              {health?.components?.length ? (
                <ul className="divide-y divide-ink-50">
                  {health.components.map((component) => (
                    <HealthRow key={component.name} component={component} />
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-ink-400">Health data unavailable.</p>
              )}
              <Link
                to="/settings"
                className="mt-3 block text-xs font-medium text-brand-600 transition hover:text-brand-700"
              >
                Configuration details →
              </Link>
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="LLM provider usage" />
            <CardBody>
              {hasLlm ? (
                <DonutChart data={data?.provider_usage} height={180} />
              ) : (
                <p className="py-6 text-center text-xs text-ink-400">
                  No cloud LLM configured — retrieval runs, generation does not.
                </p>
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Evaluation" icon={FlaskConical} />
            <CardBody>
              <p className="text-sm text-ink-700">
                <span className="text-2xl font-semibold tabular-nums text-ink-900">
                  {formatNumber(stats?.evaluation_runs)}
                </span>{' '}
                run(s) recorded
              </p>
              <p className="mt-1 text-xs text-ink-500">
                Benchmark numbers appear only after an experiment completes.
              </p>
              <Link
                to="/evaluation"
                className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium text-brand-600 transition hover:text-brand-700"
              >
                <FileStack className="h-3 w-3" aria-hidden="true" />
                Compare strategies →
              </Link>
            </CardBody>
          </Card>
        </div>
      </div>
    </>
  );
}
