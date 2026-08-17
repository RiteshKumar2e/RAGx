import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  FileText,
  Hash,
  Image as ImageIcon,
  Library,
  RefreshCw,
  Search,
  Table2,
  Trash2,
  Network,
} from 'lucide-react';
import {
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
import UploadDropzone from '../components/documents/UploadDropzone';
import ProcessingSteps from '../components/documents/ProcessingSteps';
import { useApi } from '../hooks/useApi';
import { useDebounce } from '../hooks/useDebounce';
import { usePolling } from '../hooks/usePolling';
import { useToast } from '../context/ToastContext';
import { documentService } from '../services';
import { DOCUMENT_STATUS } from '../utils/constants';
import { formatBytes, formatNumber, formatRelativeTime } from '../utils/format';
import cn from '../utils/cn';

const FILE_TYPES = ['.pdf', '.docx', '.txt', '.md', '.csv', '.png', '.jpg'];

function StatusBadge({ status }) {
  const meta = DOCUMENT_STATUS[status] || { label: status, className: 'bg-ink-100 text-ink-700' };
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium',
        meta.className,
      )}
    >
      {meta.active ? (
        <span className="h-1.5 w-1.5 animate-pulse-subtle rounded-full bg-current" aria-hidden="true" />
      ) : null}
      {meta.label}
    </span>
  );
}

function DocumentRow({ document, onDelete, onReindex, busy }) {
  const isActive = DOCUMENT_STATUS[document.status]?.active;
  const failed = document.status === 'failed';

  return (
    <li className="px-4 py-3.5 transition hover:bg-ink-50/60 sm:px-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 flex-1 items-start gap-3">
          <span className="mt-0.5 shrink-0 rounded-lg bg-ink-100 p-2 text-ink-500">
            <FileText className="h-4 w-4" aria-hidden="true" />
          </span>

          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Link
                to={`/knowledge-base/${document.id}`}
                className="min-w-0 truncate text-sm font-medium text-ink-900 hover:text-brand-600"
              >
                {document.title || document.filename}
              </Link>
              <StatusBadge status={document.status} />
            </div>

            <p className="mt-0.5 truncate text-[11px] text-ink-400">
              {document.filename} · {formatBytes(document.size_bytes)} ·{' '}
              {formatRelativeTime(document.created_at)}
            </p>

            {document.status === 'ready' ? (
              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-500">
                <span className="inline-flex items-center gap-1">
                  <Hash className="h-3 w-3" aria-hidden="true" />
                  {formatNumber(document.chunk_count)} chunks
                </span>
                {document.page_count ? <span>{document.page_count} pages</span> : null}
                {document.table_count ? (
                  <span className="inline-flex items-center gap-1">
                    <Table2 className="h-3 w-3" aria-hidden="true" />
                    {document.table_count} tables
                  </span>
                ) : null}
                {document.figure_count ? (
                  <span className="inline-flex items-center gap-1">
                    <ImageIcon className="h-3 w-3" aria-hidden="true" />
                    {document.figure_count} figures
                  </span>
                ) : null}
                {document.entity_count ? (
                  <span className="inline-flex items-center gap-1">
                    <Network className="h-3 w-3" aria-hidden="true" />
                    {document.entity_count} entities
                  </span>
                ) : null}
              </div>
            ) : null}

            {/* Live pipeline progress while processing, or the failure reason. */}
            {isActive || failed ? (
              <div className="mt-2.5 max-w-md rounded-lg border border-ink-200 bg-white p-2.5">
                <ProcessingSteps
                  steps={document.processing_steps}
                  compact
                  errorMessage={failed ? document.error_message : null}
                />
              </div>
            ) : null}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1 sm:flex-col sm:items-end">
          <Button
            variant="ghost"
            size="xs"
            icon={RefreshCw}
            onClick={() => onReindex(document)}
            disabled={busy || isActive}
            title="Re-run the ingestion pipeline"
          >
            Reindex
          </Button>
          <Button
            variant="ghost"
            size="xs"
            icon={Trash2}
            onClick={() => onDelete(document)}
            disabled={busy}
            className="text-rose-600 hover:bg-rose-50"
          >
            Delete
          </Button>
        </div>
      </div>
    </li>
  );
}

export default function KnowledgeBase() {
  const toast = useToast();
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('');
  const [fileType, setFileType] = useState('');
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [confirmTarget, setConfirmTarget] = useState(null);
  const [busy, setBusy] = useState(false);

  const debouncedSearch = useDebounce(search, 350);

  const {
    data: list,
    error,
    loading,
    refetch,
  } = useApi(
    () => documentService.list({ pageSize: 50, search: debouncedSearch, status, fileType }),
    [debouncedSearch, status, fileType],
  );

  const { data: stats, refetch: refetchStats } = useApi(() => documentService.stats(), []);

  const documents = list?.items || [];

  // Poll while anything is still being processed, so the checklist advances live.
  const hasActive = useMemo(
    () => documents.some((document) => DOCUMENT_STATUS[document.status]?.active),
    [documents],
  );
  usePolling(
    useCallback(() => {
      refetch();
      refetchStats();
    }, [refetch, refetchStats]),
    2500,
    hasActive,
  );

  const handleUpload = async (files) => {
    setUploading(true);
    setProgress(0);
    try {
      const result = await documentService.upload(files, { onProgress: setProgress });
      const accepted = result.uploaded?.filter((item) => !item.duplicate_of) || [];
      const duplicates = result.uploaded?.filter((item) => item.duplicate_of) || [];

      if (accepted.length) {
        toast.success(
          `${accepted.length} file(s) uploaded and queued for processing.`,
          { title: 'Upload complete' },
        );
      }
      duplicates.forEach((item) => toast.info(item.message));
      (result.rejected || []).forEach((item) =>
        toast.error(`${item.filename}: ${item.reason}`, { title: 'File rejected' }),
      );

      refetch();
      refetchStats();
    } catch (caught) {
      toast.error(caught.message, { title: 'Upload failed' });
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const handleIngestUrl = async (url) => {
    setUploading(true);
    try {
      const result = await documentService.ingestUrl(url);
      toast.success(result.message, { title: 'Page queued' });
      refetch();
    } catch (caught) {
      toast.error(caught.message, { title: 'Could not ingest URL' });
    } finally {
      setUploading(false);
    }
  };

  const handleReindex = async (document) => {
    setBusy(true);
    try {
      await documentService.reindex(document.id);
      toast.info(`'${document.filename}' was queued for reprocessing.`);
      refetch();
    } catch (caught) {
      toast.error(caught.message);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    if (!confirmTarget) return;
    setBusy(true);
    try {
      await documentService.remove(confirmTarget.id);
      toast.success(`'${confirmTarget.filename}' and all of its indexed data were deleted.`);
      setConfirmTarget(null);
      refetch();
      refetchStats();
    } catch (caught) {
      toast.error(caught.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Knowledge Base"
        description="Upload documents, watch them move through the ingestion pipeline, and inspect what was extracted."
      />

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Documents"
          value={formatNumber(stats?.total_documents)}
          hint={`${formatNumber(stats?.indexed_documents)} ready`}
          icon={Library}
          tone="brand"
        />
        <StatTile
          label="Chunks indexed"
          value={formatNumber(stats?.total_chunks)}
          hint={`${formatNumber(stats?.vectors_indexed)} vectors · ${formatNumber(stats?.bm25_documents)} BM25`}
          icon={Hash}
          tone="violet"
        />
        <StatTile
          label="Tables & figures"
          value={formatNumber((stats?.total_tables || 0) + (stats?.total_figures || 0))}
          hint={`${formatNumber(stats?.total_tables)} tables · ${formatNumber(stats?.total_figures)} figures`}
          icon={Table2}
          tone="amber"
        />
        <StatTile
          label="Graph entities"
          value={formatNumber(stats?.total_entities)}
          hint={`${formatNumber(stats?.total_relations)} relations`}
          icon={Network}
          tone="emerald"
        />
      </div>

      {/* Upload */}
      <div className="mt-6">
        <UploadDropzone
          onUpload={handleUpload}
          onIngestUrl={handleIngestUrl}
          uploading={uploading}
          progress={progress}
        />
      </div>

      {/* Document list */}
      <Card className="mt-6">
        <CardHeader
          title="Indexed documents"
          description={
            list?.total
              ? `${list.total} document(s) in the knowledge base`
              : 'Documents you upload appear here'
          }
          icon={Library}
          action={
            <Button variant="ghost" size="xs" icon={RefreshCw} onClick={refetch} loading={loading}>
              Refresh
            </Button>
          }
        />

        {/* Filters */}
        <div className="flex flex-col gap-2 border-b border-ink-100 p-3 sm:flex-row sm:items-center sm:px-5">
          <div className="relative min-w-0 flex-1">
            <Search
              className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-400"
              aria-hidden="true"
            />
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search by filename or title…"
              className="w-full rounded-lg border border-ink-200 py-1.5 pl-9 pr-3 text-sm placeholder:text-ink-400 focus:border-brand-400 focus:outline-none focus:ring-1 focus:ring-brand-400"
              aria-label="Search documents"
            />
          </div>

          <select
            value={status}
            onChange={(event) => setStatus(event.target.value)}
            className="rounded-lg border border-ink-200 px-2.5 py-1.5 text-xs text-ink-700 focus:border-brand-400 focus:outline-none"
            aria-label="Filter by status"
          >
            <option value="">All statuses</option>
            {Object.entries(DOCUMENT_STATUS).map(([key, meta]) => (
              <option key={key} value={key}>
                {meta.label}
              </option>
            ))}
          </select>

          <select
            value={fileType}
            onChange={(event) => setFileType(event.target.value)}
            className="rounded-lg border border-ink-200 px-2.5 py-1.5 text-xs text-ink-700 focus:border-brand-400 focus:outline-none"
            aria-label="Filter by file type"
          >
            <option value="">All types</option>
            {FILE_TYPES.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>

        {error ? (
          <ErrorState error={error} onRetry={refetch} />
        ) : loading && !documents.length ? (
          <CardBody>
            <LoadingBlock rows={5} />
          </CardBody>
        ) : !documents.length ? (
          <EmptyState
            icon={Library}
            title={search || status || fileType ? 'No matching documents' : 'No documents yet'}
            description={
              search || status || fileType
                ? 'Adjust the filters to see more results.'
                : 'Upload a PDF, DOCX, CSV, text file or image above. RAGX answers strictly from what you index.'
            }
            className="py-12"
          />
        ) : (
          <ul className="divide-y divide-ink-100">
            {documents.map((document) => (
              <DocumentRow
                key={document.id}
                document={document}
                onDelete={setConfirmTarget}
                onReindex={handleReindex}
                busy={busy}
              />
            ))}
          </ul>
        )}
      </Card>

      <ConfirmDialog
        open={Boolean(confirmTarget)}
        onClose={() => setConfirmTarget(null)}
        onConfirm={handleDelete}
        pending={busy}
        title="Delete document"
        message={`'${confirmTarget?.filename}' will be removed from the vector index, the BM25 index, the knowledge graph and object storage. This cannot be undone.`}
      />
    </>
  );
}
