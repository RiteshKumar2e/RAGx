import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FileText, Layers } from 'lucide-react';
import Modal from '../common/Modal';
import { ErrorState, LoadingBlock } from '../common';
import { queryService } from '../../services';
import { MODALITY_STYLES } from '../../utils/constants';
import { citationLocation, formatRatio } from '../../utils/format';
import cn from '../../utils/cn';

/**
 * Full-passage viewer for a citation.
 *
 * Fetches the complete chunk plus its neighbours, so a user can confirm a claim
 * in its original context rather than trusting a truncated excerpt.
 */
export default function EvidenceModal({ item, open, onClose }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    if (!open || !item?.chunk_id) return undefined;

    let cancelled = false;
    setLoading(true);
    setError(null);
    setDetail(null);
    setImageFailed(false);

    queryService
      .evidence(item.chunk_id)
      .then((payload) => {
        if (!cancelled) setDetail(payload);
      })
      .catch((caught) => {
        if (!cancelled) setError(caught);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, item?.chunk_id]);

  if (!item) return null;

  const modality = MODALITY_STYLES[detail?.modality || item.modality] || MODALITY_STYLES.text;
  const location = citationLocation(detail || item);

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={detail?.document_name || item.document_name}
      description={location || 'Source passage'}
      size="lg"
    >
      {/* Provenance header */}
      <div className="mb-4 flex flex-wrap items-center gap-2 rounded-lg bg-ink-50 p-3">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-brand-600 text-[11px] font-semibold text-white">
          {item.marker}
        </span>
        <span className={cn('rounded px-1.5 py-0.5 text-[11px] font-medium', modality.className)}>
          {modality.label}
        </span>
        {item.relevance != null ? (
          <span className="rounded bg-white px-1.5 py-0.5 font-mono text-[11px] text-ink-600 ring-1 ring-inset ring-ink-200">
            relevance {formatRatio(item.relevance, 3)}
          </span>
        ) : null}
        {item.used_in_answer ? (
          <span className="rounded bg-emerald-50 px-1.5 py-0.5 text-[11px] font-medium text-emerald-700 ring-1 ring-inset ring-emerald-600/20">
            cited in answer
          </span>
        ) : (
          <span className="rounded bg-ink-100 px-1.5 py-0.5 text-[11px] font-medium text-ink-600">
            retrieved, not cited
          </span>
        )}
        {detail?.document_id ? (
          <Link
            to={`/knowledge-base/${detail.document_id}`}
            onClick={onClose}
            className="ml-auto inline-flex items-center gap-1 text-[11px] font-medium text-brand-600 hover:text-brand-700"
          >
            <FileText className="h-3 w-3" aria-hidden="true" />
            Open document
          </Link>
        ) : null}
      </div>

      {loading ? <LoadingBlock rows={5} /> : null}
      {error ? <ErrorState error={error} compact /> : null}

      {detail ? (
        <>
          {/* Figure / table image */}
          {detail.has_image && !imageFailed ? (
            <figure className="mb-4 overflow-hidden rounded-lg border border-ink-200 bg-ink-50">
              <img
                src={queryService.evidenceImageUrl(detail.chunk_id)}
                alt={`${detail.figure || detail.table || 'Figure'} from ${detail.document_name}`}
                className="mx-auto max-h-[420px] w-auto max-w-full object-contain"
                onError={() => setImageFailed(true)}
              />
              {detail.figure || detail.table ? (
                <figcaption className="border-t border-ink-200 px-3 py-2 text-[11px] text-ink-500">
                  {detail.figure || detail.table}
                  {detail.page ? ` · page ${detail.page}` : ''}
                </figcaption>
              ) : null}
            </figure>
          ) : null}

          {/* Section breadcrumb */}
          {detail.section_path?.length ? (
            <nav aria-label="Section" className="mb-2 flex flex-wrap items-center gap-1 text-[11px] text-ink-500">
              {detail.section_path.map((part, index) => (
                <span key={`${part}-${index}`} className="flex items-center gap-1">
                  {index > 0 ? <span aria-hidden="true">›</span> : null}
                  <span>{part}</span>
                </span>
              ))}
            </nav>
          ) : null}

          {/* The passage itself */}
          <div className="rounded-lg border border-ink-200 bg-white p-4">
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink-800">{detail.content}</p>
          </div>

          <p className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-ink-400">
            <span>chunk #{detail.ordinal}</span>
            <span>{detail.token_count} tokens</span>
            <span className="font-mono">{detail.chunk_id}</span>
          </p>

          {/* Surrounding context */}
          {detail.neighbors?.length ? (
            <div className="mt-5">
              <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-600">
                <Layers className="h-3.5 w-3.5 text-ink-400" aria-hidden="true" />
                Surrounding context
              </h3>
              <div className="space-y-2">
                {detail.neighbors.map((neighbor) => (
                  <div
                    key={neighbor.chunk_id}
                    className="rounded-lg border border-dashed border-ink-200 bg-ink-50/50 p-3"
                  >
                    <p className="mb-1 text-[11px] text-ink-400">
                      chunk #{neighbor.ordinal}
                      {neighbor.section ? ` · ${neighbor.section}` : ''}
                    </p>
                    <p className="text-xs leading-relaxed text-ink-600">{neighbor.content}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </>
      ) : null}
    </Modal>
  );
}
