import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Boxes,
  CheckCircle2,
  Cloud,
  Database,
  Eraser,
  Lock,
  RefreshCw,
  Save,
  ShieldCheck,
  Sliders,
  Workflow,
  XCircle,
} from 'lucide-react';
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  ErrorState,
  InfoBanner,
  LoadingBlock,
  PageHeader,
} from '../components/common';
import { useSystem } from '../context/SystemContext';
import { useToast } from '../context/ToastContext';
import { systemService } from '../services';
import { formatMs, formatNumber, formatPercent } from '../utils/format';
import { strategyMeta } from '../utils/strategy';
import cn from '../utils/cn';

function StatusDot({ healthy }) {
  if (healthy === true) return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" />;
  if (healthy === false) return <XCircle className="h-3.5 w-3.5 text-rose-600" aria-hidden="true" />;
  return <span className="h-2 w-2 rounded-full bg-ink-300" aria-hidden="true" />;
}

function StorageRow({ label, detail }) {
  return (
    <div className="flex items-start gap-2.5 py-2">
      <StatusDot healthy={detail?.healthy} />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-ink-900">{label}</p>
        <p className="truncate text-[11px] text-ink-500">
          {[detail?.backend, detail?.mode, detail?.engine, detail?.collection]
            .filter(Boolean)
            .join(' · ') || detail?.status_text || 'unknown'}
        </p>
        {detail?.error ? (
          <p className="mt-0.5 text-[11px] text-rose-600">{detail.error}</p>
        ) : null}
      </div>
      <span className="shrink-0 text-[11px] tabular-nums text-ink-400">
        {detail?.vectors != null ? `${formatNumber(detail.vectors)} vectors` : null}
        {detail?.entities != null ? `${formatNumber(detail.entities)} entities` : null}
        {detail?.documents != null ? `${formatNumber(detail.documents)} docs` : null}
      </span>
    </div>
  );
}

const TUNABLE = [
  { key: 'default_top_k', label: 'Default Top-K', min: 1, max: 30, step: 1, help: 'Passages passed to the model.' },
  { key: 'candidate_pool_size', label: 'Candidate pool', min: 5, max: 200, step: 5, help: 'Results fetched before fusion and reranking.' },
  { key: 'min_relevance_score', label: 'Min relevance', min: 0, max: 1, step: 0.01, help: 'Retrieved passages below this are dropped.' },
  { key: 'corrective_relevance_floor', label: 'Corrective floor', min: 0, max: 1, step: 0.01, help: 'Retrieval quality below this triggers corrective retrieval.' },
  { key: 'corrective_max_rounds', label: 'Max corrective rounds', min: 0, max: 5, step: 1, help: 'Repair attempts before giving up.' },
  { key: 'agentic_max_steps', label: 'Max agentic steps', min: 1, max: 10, step: 1, help: 'Upper bound on the agent loop.' },
  { key: 'insufficient_evidence_threshold', label: 'Abstention threshold', min: 0, max: 1, step: 0.01, help: 'Confidence below this withholds the answer.' },
];

export default function Settings() {
  const { settings, warnings, loading, error, reload, setSettings } = useSystem();
  const toast = useToast();

  const [form, setForm] = useState({});
  const [saving, setSaving] = useState(false);
  const [probing, setProbing] = useState(false);

  useEffect(() => {
    if (!settings) return;
    setForm({
      ...settings.retrieval,
      verification_enabled: settings.verification?.enabled,
      insufficient_evidence_threshold: settings.verification?.insufficient_evidence_threshold,
      primary_llm_provider: settings.llm?.primary,
      fallback_llm_provider: settings.llm?.fallback,
      rerank_enabled: settings.retrieval?.rerank_enabled,
    });
  }, [settings]);

  const save = async () => {
    setSaving(true);
    try {
      const updated = await systemService.updateSettings(form);
      setSettings(updated);
      toast.success('Settings updated for this process.', {
        title: 'Saved',
      });
    } catch (caught) {
      toast.error(caught.message);
    } finally {
      setSaving(false);
    }
  };

  const probe = async () => {
    setProbing(true);
    try {
      const status = await systemService.llmStatus(true);
      const healthy = status.providers.filter((provider) => provider.healthy);
      if (healthy.length) {
        toast.success(
          `${healthy.map((p) => p.provider).join(', ')} responded successfully.`,
          { title: 'Providers reachable' },
        );
      } else {
        toast.warning('No provider responded. Check the API keys in the backend environment.');
      }
      reload();
    } catch (caught) {
      toast.error(caught.message);
    } finally {
      setProbing(false);
    }
  };

  const clearCaches = async () => {
    try {
      await systemService.clearCache();
      toast.success('Embedding, analysis and answer caches cleared.');
      reload();
    } catch (caught) {
      toast.error(caught.message);
    }
  };

  if (error) {
    return (
      <>
        <PageHeader title="Settings" />
        <Card>
          <ErrorState error={error} onRetry={reload} />
        </Card>
      </>
    );
  }

  if (loading && !settings) {
    return (
      <>
        <PageHeader title="Settings" />
        <Card>
          <CardBody>
            <LoadingBlock rows={8} />
          </CardBody>
        </Card>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Settings"
        description="Provider status, storage health and retrieval tuning. Credentials live in the backend environment and are never shown or accepted here."
        action={
          <Button variant="secondary" size="sm" icon={RefreshCw} onClick={reload} loading={loading}>
            Refresh
          </Button>
        }
      />

      {warnings?.length ? (
        <div className="mb-6 space-y-2">
          {warnings.map((warning) => (
            <InfoBanner key={warning} variant="warning" icon={AlertTriangle}>
              {warning}
            </InfoBanner>
          ))}
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        {/* ------------------------------------------------------- providers */}
        <Card>
          <CardHeader
            title="LLM providers"
            description="Cloud APIs only. No local model runtime is supported."
            icon={Cloud}
            action={
              <Button variant="ghost" size="xs" onClick={probe} loading={probing}>
                Test connection
              </Button>
            }
          />
          <CardBody>
            <ul className="divide-y divide-ink-50">
              {(settings?.llm?.providers || []).map((provider) => (
                <li key={provider.provider} className="flex items-start gap-2.5 py-2.5">
                  <StatusDot healthy={provider.configured ? provider.healthy ?? true : false} />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="text-sm font-medium capitalize text-ink-900">
                        {provider.provider} API
                      </span>
                      {provider.provider === settings?.llm?.primary ? (
                        <Badge className="bg-brand-50 text-brand-700 ring-brand-600/20" size="xs">
                          primary
                        </Badge>
                      ) : null}
                      {provider.provider === settings?.llm?.fallback ? (
                        <Badge size="xs">fallback</Badge>
                      ) : null}
                      {provider.multimodal ? (
                        <Badge className="bg-violet-50 text-violet-700 ring-violet-600/20" size="xs">
                          vision
                        </Badge>
                      ) : null}
                    </div>
                    <p className="mt-0.5 truncate font-mono text-[11px] text-ink-500">
                      {provider.model}
                      {provider.fast_model ? ` · ${provider.fast_model}` : ''}
                    </p>
                    <p className="text-[11px] text-ink-500">
                      {provider.configured ? 'API key configured' : 'No API key set'}
                      {provider.latency_ms ? ` · ${formatMs(provider.latency_ms)}` : ''}
                    </p>
                    {provider.error ? (
                      <p className="mt-0.5 text-[11px] text-rose-600">{provider.error}</p>
                    ) : null}
                  </div>
                </li>
              ))}
            </ul>

            <div className="mt-3 grid gap-3 border-t border-ink-100 pt-3 sm:grid-cols-2">
              <div>
                <label htmlFor="primary-provider" className="mb-1 block text-[11px] font-medium text-ink-600">
                  Primary provider
                </label>
                <select
                  id="primary-provider"
                  value={form.primary_llm_provider || 'gemini'}
                  onChange={(event) => setForm({ ...form, primary_llm_provider: event.target.value })}
                  className="w-full rounded-lg border border-ink-200 px-2.5 py-1.5 text-xs focus:border-brand-400 focus:outline-none"
                >
                  <option value="gemini">Gemini</option>
                  <option value="groq">Groq</option>
                </select>
              </div>
              <div>
                <label htmlFor="fallback-provider" className="mb-1 block text-[11px] font-medium text-ink-600">
                  Fallback provider
                </label>
                <select
                  id="fallback-provider"
                  value={form.fallback_llm_provider || 'groq'}
                  onChange={(event) => setForm({ ...form, fallback_llm_provider: event.target.value })}
                  className="w-full rounded-lg border border-ink-200 px-2.5 py-1.5 text-xs focus:border-brand-400 focus:outline-none"
                >
                  <option value="groq">Groq</option>
                  <option value="gemini">Gemini</option>
                  <option value="none">None</option>
                </select>
              </div>
            </div>

            <div className="mt-3 flex items-start gap-2 rounded-lg bg-ink-50 p-2.5">
              <Lock className="mt-0.5 h-3 w-3 shrink-0 text-ink-400" aria-hidden="true" />
              <p className="text-[11px] leading-relaxed text-ink-500">
                API keys are read from the backend environment only. They cannot be set from this
                page and are never sent to the browser. Edit <code className="font-mono">.env</code>{' '}
                and restart the backend to change them.
              </p>
            </div>
          </CardBody>
        </Card>

        {/* --------------------------------------------------------- storage */}
        <Card>
          <CardHeader title="Storage" description="Component health across all four layers." icon={Database} />
          <CardBody>
            <div className="divide-y divide-ink-50">
              <StorageRow label="Vector store (Qdrant)" detail={settings?.storage?.vector_store} />
              <StorageRow label="Graph store" detail={settings?.storage?.graph_store} />
              <StorageRow label="BM25 index" detail={settings?.storage?.bm25_index} />
              <StorageRow label="Relational database" detail={settings?.storage?.relational} />
              <StorageRow label="Object storage" detail={settings?.storage?.object_storage} />
            </div>

            <div className="mt-3 border-t border-ink-100 pt-3">
              <p className="mb-1.5 text-[11px] font-medium text-ink-600">Embeddings</p>
              <div className="flex items-start gap-2.5">
                <StatusDot healthy={settings?.embeddings?.healthy} />
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium capitalize text-ink-900">
                    {settings?.embeddings?.provider}
                    {settings?.embeddings?.model ? ` · ${settings.embeddings.model}` : ''}
                  </p>
                  <p className="text-[11px] text-ink-500">
                    {settings?.embeddings?.dimension} dimensions ·{' '}
                    {settings?.embeddings?.production_ready ? 'production' : 'development only'}
                  </p>
                  {settings?.embeddings?.warning ? (
                    <p className="mt-1 text-[11px] leading-relaxed text-amber-700">
                      {settings.embeddings.warning}
                    </p>
                  ) : null}
                </div>
              </div>
            </div>
          </CardBody>
        </Card>
      </div>

      {/* ------------------------------------------------------- retrieval */}
      <Card className="mt-4">
        <CardHeader
          title="Retrieval &amp; verification"
          description="Runtime tuning for this process. Changes are not persisted across restarts — put permanent values in .env."
          icon={Sliders}
          action={
            <Button size="sm" icon={Save} onClick={save} loading={saving}>
              Save
            </Button>
          }
        />
        <CardBody>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {TUNABLE.map((field) => (
              <div key={field.key}>
                <label
                  htmlFor={field.key}
                  className="mb-1 block text-[11px] font-medium text-ink-600"
                >
                  {field.label}
                </label>
                <input
                  id={field.key}
                  type="number"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={form[field.key] ?? ''}
                  onChange={(event) =>
                    setForm({ ...form, [field.key]: Number(event.target.value) })
                  }
                  className="w-full rounded-lg border border-ink-200 px-2.5 py-1.5 text-sm tabular-nums focus:border-brand-400 focus:outline-none focus:ring-1 focus:ring-brand-400"
                />
                <p className="mt-1 text-[10px] leading-snug text-ink-400">{field.help}</p>
              </div>
            ))}
          </div>

          <div className="mt-4 flex flex-wrap gap-4 border-t border-ink-100 pt-4">
            <label className="flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={Boolean(form.rerank_enabled)}
                onChange={(event) => setForm({ ...form, rerank_enabled: event.target.checked })}
                className="h-3.5 w-3.5 rounded border-ink-300 text-brand-600 focus:ring-brand-500"
              />
              <span className="text-xs text-ink-700">Enable reranking</span>
            </label>
            <label className="flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={Boolean(form.verification_enabled)}
                onChange={(event) => setForm({ ...form, verification_enabled: event.target.checked })}
                className="h-3.5 w-3.5 rounded border-ink-300 text-brand-600 focus:ring-brand-500"
              />
              <span className="text-xs text-ink-700">Enable evidence verification</span>
            </label>
          </div>

          {!form.verification_enabled ? (
            <InfoBanner variant="danger" icon={ShieldCheck} className="mt-4">
              Verification is disabled. Answers will not be claim-checked and unsupported answers
              will not be withheld — grounding guarantees do not hold in this mode.
            </InfoBanner>
          ) : null}
        </CardBody>
      </Card>

      {/* -------------------------------------------------- strategies + cache */}
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Available strategies"
            description="Each is a modular component the router can select or compose."
            icon={Workflow}
          />
          <CardBody>
            <ul className="space-y-2">
              {(settings?.strategies || []).map((strategy) => {
                const meta = strategyMeta(strategy.name);
                return (
                  <li key={strategy.name} className="flex items-start gap-2.5">
                    <span
                      className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: meta.color }}
                      aria-hidden="true"
                    />
                    <div className="min-w-0">
                      <p className="flex items-center gap-2 text-xs font-medium text-ink-900">
                        {strategy.label}
                        {strategy.uses_llm ? (
                          <Badge size="xs" className="bg-amber-50 text-amber-700 ring-amber-600/20">
                            uses LLM
                          </Badge>
                        ) : null}
                      </p>
                      <p className="mt-0.5 text-[11px] leading-relaxed text-ink-500">
                        {strategy.description}
                      </p>
                    </div>
                  </li>
                );
              })}
            </ul>
          </CardBody>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader title="Caches" icon={Boxes} />
            <CardBody>
              <ul className="space-y-2">
                {Object.entries(settings?.cache || {}).map(([name, stats]) => (
                  <li key={name} className="text-[11px]">
                    <div className="flex items-center justify-between">
                      <span className="capitalize text-ink-700">{name.replace(/_/g, ' ')}</span>
                      <span className="tabular-nums text-ink-500">
                        {stats.entries}/{stats.max_entries}
                      </span>
                    </div>
                    <p className="text-ink-400">
                      hit rate {formatPercent(stats.hit_rate)} · TTL {stats.ttl_seconds}s
                    </p>
                  </li>
                ))}
              </ul>
              <Button
                variant="secondary"
                size="xs"
                icon={Eraser}
                onClick={clearCaches}
                className="mt-3 w-full"
              >
                Clear caches
              </Button>
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Ingestion" />
            <CardBody>
              <dl className="space-y-1.5 text-[11px]">
                <div className="flex justify-between gap-2">
                  <dt className="text-ink-500">Max upload</dt>
                  <dd className="font-medium text-ink-900">
                    {settings?.ingestion?.max_upload_mb} MB
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-ink-500">Chunk target</dt>
                  <dd className="font-medium text-ink-900">
                    {settings?.ingestion?.chunk_target_tokens} tokens
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-ink-500">Chunk overlap</dt>
                  <dd className="font-medium text-ink-900">
                    {settings?.ingestion?.chunk_overlap_tokens} tokens
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-ink-500">OCR</dt>
                  <dd
                    className={cn(
                      'font-medium',
                      settings?.ingestion?.ocr_available ? 'text-emerald-700' : 'text-amber-700',
                    )}
                  >
                    {settings?.ingestion?.ocr_enabled
                      ? settings?.ingestion?.ocr_available
                        ? 'Tesseract available'
                        : 'Vision fallback only'
                      : 'Disabled'}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-ink-500">Entity extraction</dt>
                  <dd className="font-medium text-ink-900">
                    {settings?.ingestion?.entity_extraction ? 'Enabled' : 'Disabled'}
                  </dd>
                </div>
              </dl>
              <p className="mt-2 border-t border-ink-100 pt-2 text-[10px] leading-relaxed text-ink-400">
                Accepted: {(settings?.ingestion?.allowed_extensions || []).join(' · ')}
              </p>
            </CardBody>
          </Card>
        </div>
      </div>
    </>
  );
}
