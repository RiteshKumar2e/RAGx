import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  ArrowLeft,
  Download,
  FileText,
  Hash,
  Image as ImageIcon,
  ListTree,
  Network,
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
  LoadingBlock,
  PageHeader,
  StatTile,
} from '../components/common';
import { ConfirmDialog } from '../components/common/Modal';
import ProcessingSteps from '../components/documents/ProcessingSteps';
import { useApi } from '../hooks/useApi';
import { usePolling } from '../hooks/usePolling';
import { useToast } from '../context/ToastContext';
import { documentService } from '../services';
import { DOCUMENT_STATUS, ENTITY_TYPE_COLORS, MODALITY_STYLES } from '../utils/constants';
import { formatBytes, formatDateTime, formatMs, formatNumber, truncate } from '../utils/format';
import cn from '../utils/cn';

const TABS = [
  { key: 'chunks', label: 'Chunks' },
  { key: 'entities', label: 'Entities' },
  { key: 'outline', label: 'Outline' },
  { key: 'metadata', label: 'Metadata' },
];

export default function DocumentDetails() {
  const { documentId } = useParams();
  const navigate = useNavigate();
  const toast = useToast();

  const [tab, setTab] = useState('chunks');
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const { data, error, loading, refetch } = useApi(
    () => documentService.detail(documentId, 120),
    [documentId],
  );

  const isActive = DOCUMENT_STATUS[data?.status]?.active;
  usePolling(refetch, 2500, Boolean(isActive));

  const handleReindex = async () => {
    setBusy(true);
    try {
      await documentService.reindex(documentId);
      toast.info('Queued for reprocessing.');
      refetch();
    } catch (caught) {
      toast.error(caught.message);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    setBusy(true);
    try {
      await documentService.remove(documentId);
      toast.success('Document deleted.');
      navigate('/knowledge-base');
    } catch (caught) {
      toast.error(caught.message);
      setBusy(false);
    }
  };

  if (error) {
    return (
      <>
        <Link
          to="/knowledge-base"
          className="mb-4 inline-flex items-center gap-1.5 text-sm text-ink-500 hover:text-ink-800"
        >
          <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
          Knowledge Base
        </Link>
        <Card>
          <ErrorState error={error} onRetry={refetch} />
        </Card>
      </>
    );
  }

  if (loading && !data) {
    return (
      <Card>
        <CardBody>
          <LoadingBlock rows={8} />
        </CardBody>
      </Card>
    );
  }

  const statusMeta = DOCUMENT_STATUS[data?.status] || {};

  return (
    <>
      <Link
        to="/knowledge-base"
        className="mb-4 inline-flex items-center gap-1.5 text-sm text-ink-500 transition hover:text-ink-800"
      >
        <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
        Knowledge Base
      </Link>

      <PageHeader
        title={data?.title || data?.filename}
        description={`${data?.filename} · ${formatBytes(data?.size_bytes)} · uploaded ${formatDateTime(data?.created_at)}`}
        action={
          <div className="flex items-center gap-2">
            <a
              href={documentService.fileUrl(documentId)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-lg bg-white px-3 py-1.5 text-xs font-medium text-ink-800 ring-1 ring-inset ring-ink-200 transition hover:bg-ink-50"
            >
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              Original
            </a>
            <Button
              variant="secondary"
              size="sm"
              icon={RefreshCw}
              onClick={handleReindex}
              disabled={busy || isActive}
            >
              Reindex
            </Button>
            <Button
              variant="ghost"
              size="sm"
              icon={Trash2}
              onClick={() => setConfirmOpen(true)}
              className="text-rose-600 hover:bg-rose-50"
            >
              Delete
            </Button>
          </div>
        }
      >
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span
            className={cn(
              'inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium',
              statusMeta.className || 'bg-ink-100 text-ink-700',
            )}
          >
            {statusMeta.active ? (
              <span className="h-1.5 w-1.5 animate-pulse-subtle rounded-full bg-current" aria-hidden="true" />
            ) : null}
            {statusMeta.label || data?.status}
          </span>
          <Badge>{data?.file_type}</Badge>
          {data?.processing_ms ? (
            <span className="text-[11px] text-ink-400">
              processed in {formatMs(data.processing_ms)}
            </span>
          ) : null}
        </div>
      </PageHeader>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <StatTile label="Chunks" value={formatNumber(data?.chunk_count)} icon={Hash} tone="brand" />
        <StatTile label="Pages" value={formatNumber(data?.page_count)} icon={FileText} tone="ink" />
        <StatTile label="Tables" value={formatNumber(data?.table_count)} icon={Table2} tone="amber" />
        <StatTile label="Figures" value={formatNumber(data?.figure_count)} icon={ImageIcon} tone="violet" />
        <StatTile label="Entities" value={formatNumber(data?.entity_count)} icon={Network} tone="emerald" />
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        {/* -------------------------------------------------------- content */}
        <Card className="lg:col-span-2">
          <div className="flex gap-1 border-b border-ink-100 px-3 pt-3 sm:px-5">
            {TABS.map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setTab(item.key)}
                className={cn(
                  'rounded-t-lg px-3 py-2 text-xs font-medium transition',
                  tab === item.key
                    ? 'border-b-2 border-brand-600 text-brand-700'
                    : 'text-ink-500 hover:text-ink-800',
                )}
              >
                {item.label}
                {item.key === 'chunks' && data?.chunks?.length ? (
                  <span className="ml-1.5 text-ink-400">{data.chunks.length}</span>
                ) : null}
                {item.key === 'entities' && data?.entities?.length ? (
                  <span className="ml-1.5 text-ink-400">{data.entities.length}</span>
                ) : null}
              </button>
            ))}
          </div>

          <CardBody>
            {/* Chunks */}
            {tab === 'chunks' ? (
              data?.chunks?.length ? (
                <ul className="space-y-2">
                  {data.chunks.map((chunk) => {
                    const modality = MODALITY_STYLES[chunk.modality] || MODALITY_STYLES.text;
                    return (
                      <li key={chunk.id} className="rounded-lg border border-ink-200 bg-white p-3">
                        <div className="mb-1.5 flex flex-wrap items-center gap-2">
                          <span className="font-mono text-[11px] text-ink-400">#{chunk.ordinal}</span>
                          <span
                            className={cn(
                              'rounded px-1.5 py-0.5 text-[10px] font-medium',
                              modality.className,
                            )}
                          >
                            {modality.label}
                          </span>
                          {chunk.page_number ? (
                            <span className="text-[11px] text-ink-500">p.{chunk.page_number}</span>
                          ) : null}
                          {chunk.section ? (
                            <span className="truncate text-[11px] text-ink-500">{chunk.section}</span>
                          ) : null}
                          {chunk.figure_label || chunk.table_label ? (
                            <span className="rounded bg-ink-100 px-1.5 py-0.5 text-[10px] text-ink-600">
                              {chunk.figure_label || chunk.table_label}
                            </span>
                          ) : null}
                          <span className="ml-auto shrink-0 text-[11px] text-ink-400">
                            {chunk.token_count} tokens
                            {chunk.indexed_in_vector_store ? ' · indexed' : ''}
                          </span>
                        </div>
                        <p className="whitespace-pre-wrap text-xs leading-relaxed text-ink-600">
                          {truncate(chunk.content, 480)}
                        </p>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <EmptyState
                  icon={Hash}
                  title="No chunks yet"
                  description={
                    isActive
                      ? 'The document is still being processed.'
                      : 'This document produced no retrievable chunks.'
                  }
                  className="py-8"
                />
              )
            ) : null}

            {/* Entities */}
            {tab === 'entities' ? (
              data?.entities?.length ? (
                <div className="flex flex-wrap gap-2">
                  {data.entities.map((entity) => {
                    const color = ENTITY_TYPE_COLORS[entity.entity_type] || ENTITY_TYPE_COLORS.CONCEPT;
                    return (
                      <div
                        key={entity.id}
                        className="max-w-xs rounded-lg border border-ink-200 bg-white p-2.5"
                      >
                        <div className="flex items-center gap-1.5">
                          <span
                            className="h-2 w-2 shrink-0 rounded-full"
                            style={{ backgroundColor: color }}
                            aria-hidden="true"
                          />
                          <span className="truncate text-xs font-medium text-ink-900">
                            {entity.name}
                          </span>
                        </div>
                        <p className="mt-0.5 text-[10px] uppercase tracking-wide text-ink-400">
                          {entity.entity_type} · {entity.mention_count} mention(s)
                        </p>
                        {entity.description ? (
                          <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-ink-500">
                            {entity.description}
                          </p>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <EmptyState
                  icon={Network}
                  title="No entities extracted"
                  description="Entity extraction needs a cloud LLM provider. Without one, only surface-form technical terms are captured."
                  className="py-8"
                />
              )
            ) : null}

            {/* Outline */}
            {tab === 'outline' ? (
              data?.outline?.length ? (
                <ul className="space-y-1">
                  {data.outline.map((item, index) => (
                    <li
                      key={`${item.title}-${index}`}
                      className="flex items-baseline gap-2 text-xs text-ink-700"
                      style={{ paddingLeft: `${Math.max(0, (item.level || 1) - 1) * 16}px` }}
                    >
                      <span className="text-ink-300" aria-hidden="true">
                        {'#'.repeat(Math.min(item.level || 1, 4))}
                      </span>
                      <span className="min-w-0 flex-1 truncate">{item.title}</span>
                      {item.page ? <span className="text-[11px] text-ink-400">p.{item.page}</span> : null}
                    </li>
                  ))}
                </ul>
              ) : (
                <EmptyState
                  icon={ListTree}
                  title="No outline detected"
                  description="This document has no heading structure the parser could recover."
                  className="py-8"
                />
              )
            ) : null}

            {/* Metadata */}
            {tab === 'metadata' ? (
              <dl className="divide-y divide-ink-50">
                {Object.entries(data?.doc_metadata || {})
                  .filter(([, value]) => value !== null && value !== undefined && value !== '')
                  .map(([key, value]) => (
                    <div key={key} className="flex items-start justify-between gap-4 py-2 text-xs">
                      <dt className="shrink-0 capitalize text-ink-500">{key.replace(/_/g, ' ')}</dt>
                      <dd className="min-w-0 break-words text-right font-medium text-ink-900">
                        {Array.isArray(value) ? value.join(', ') || '—' : String(value)}
                      </dd>
                    </div>
                  ))}
                <div className="flex items-start justify-between gap-4 py-2 text-xs">
                  <dt className="shrink-0 text-ink-500">Checksum</dt>
                  <dd className="min-w-0 truncate text-right font-mono text-[11px] text-ink-600">
                    {data?.checksum || '—'}
                  </dd>
                </div>
                <div className="flex items-start justify-between gap-4 py-2 text-xs">
                  <dt className="shrink-0 text-ink-500">Storage backend</dt>
                  <dd className="font-medium text-ink-900">{data?.storage_backend || '—'}</dd>
                </div>
              </dl>
            ) : null}
          </CardBody>
        </Card>

        {/* ----------------------------------------------------- side panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader title="Processing pipeline" />
            <CardBody>
              <ProcessingSteps steps={data?.processing_steps} errorMessage={data?.error_message} />
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Content breakdown" />
            <CardBody>
              {Object.keys(data?.modality_breakdown || {}).length ? (
                <ul className="space-y-2">
                  {Object.entries(data.modality_breakdown).map(([modality, count]) => {
                    const meta = MODALITY_STYLES[modality] || MODALITY_STYLES.text;
                    const total = Object.values(data.modality_breakdown).reduce((a, b) => a + b, 0);
                    const percentage = total ? (count / total) * 100 : 0;
                    return (
                      <li key={modality}>
                        <div className="mb-1 flex items-center justify-between text-[11px]">
                          <span className="text-ink-600">{meta.label}</span>
                          <span className="tabular-nums text-ink-500">{count}</span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-ink-100">
                          <div
                            className="h-full rounded-full bg-brand-500"
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="text-xs text-ink-400">No chunks indexed yet.</p>
              )}
            </CardBody>
          </Card>

          {data?.entity_count ? (
            <Card>
              <CardBody>
                <Link
                  to={`/graph?document=${documentId}`}
                  className="flex items-center gap-2 text-xs font-medium text-brand-600 hover:text-brand-700"
                >
                  <Network className="h-3.5 w-3.5" aria-hidden="true" />
                  View this document in the knowledge graph →
                </Link>
              </CardBody>
            </Card>
          ) : null}
        </div>
      </div>

      <ConfirmDialog
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        onConfirm={handleDelete}
        pending={busy}
        title="Delete document"
        message={`'${data?.filename}' will be removed from every index. This cannot be undone.`}
      />
    </>
  );
}
