import { useState } from 'react';
import { ExternalLink, FileText, Image as ImageIcon, Quote, Search, Table2 } from 'lucide-react';
import { EmptyState } from '../common';
import EvidenceModal from './EvidenceModal';
import { MODALITY_STYLES } from '../../utils/constants';
import { citationLocation, formatRatio, truncate } from '../../utils/format';
import { strategyMeta } from '../../utils/strategy';
import cn from '../../utils/cn';

function modalityIcon(modality) {
  if (modality === 'table') return Table2;
  if (modality === 'figure' || modality === 'image') return ImageIcon;
  return FileText;
}

/** One evidence card in the sidebar. */
function EvidenceCard({ item, active, onOpen }) {
  const Icon = modalityIcon(item.modality);
  const location = citationLocation(item);
  const modality = MODALITY_STYLES[item.modality] || MODALITY_STYLES.text;

  return (
    <button
      type="button"
      id={`evidence-${item.chunk_id}`}
      onClick={() => onOpen(item)}
      className={cn(
        'w-full scroll-mt-4 rounded-xl border p-3 text-left transition',
        active
          ? 'border-brand-300 bg-brand-50/60 ring-2 ring-brand-200'
          : 'border-ink-200 bg-white hover:border-ink-300 hover:shadow-card-hover',
      )}
    >
      <div className="flex items-start gap-2.5">
        <span
          className={cn(
            'flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[11px] font-semibold',
            item.used_in_answer ? 'bg-brand-600 text-white' : 'bg-ink-100 text-ink-600',
          )}
          title={item.used_in_answer ? 'Cited in the answer' : 'Retrieved but not cited'}
        >
          {item.marker}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <p className="min-w-0 truncate text-xs font-semibold text-ink-900">
              {item.document_name}
            </p>
            <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink-500">
              {formatRatio(item.relevance, 2)}
            </span>
          </div>

          {location ? (
            <p className="mt-0.5 flex items-center gap-1 truncate text-[11px] text-ink-500">
              <Icon className="h-3 w-3 shrink-0" aria-hidden="true" />
              {location}
            </p>
          ) : null}

          <p className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-ink-600">
            {truncate(item.content || item.excerpt, 220)}
          </p>

          <div className="mt-2 flex flex-wrap items-center gap-1">
            <span className={cn('rounded px-1.5 py-0.5 text-[10px] font-medium', modality.className)}>
              {modality.label}
            </span>
            {(item.sources || [])
              .filter((source) => !source.startsWith('neighbor'))
              .slice(0, 3)
              .map((source) => (
                <span
                  key={source}
                  className="rounded px-1.5 py-0.5 text-[10px] font-medium"
                  style={{
                    backgroundColor: `${strategyMeta(source).color}14`,
                    color: strategyMeta(source).color,
                  }}
                  title={`Retrieved by ${strategyMeta(source).label}`}
                >
                  {strategyMeta(source).short}
                </span>
              ))}
          </div>

          {item.graph_path ? (
            <p className="mt-1.5 truncate font-mono text-[10px] text-emerald-700" title={item.graph_path}>
              {item.graph_path}
            </p>
          ) : null}
        </div>
      </div>
    </button>
  );
}

/**
 * Evidence sidebar.
 *
 * Cards are ordered by the citation marker the answer uses, so clicking a
 * marker in the answer scrolls to the matching card and vice versa.
 */
export default function EvidencePanel({ evidence = [], activeChunkId, onSelect, className }) {
  const [modalItem, setModalItem] = useState(null);
  const [filter, setFilter] = useState('');

  const filtered = filter
    ? evidence.filter(
        (item) =>
          item.document_name?.toLowerCase().includes(filter.toLowerCase()) ||
          (item.content || item.excerpt || '').toLowerCase().includes(filter.toLowerCase()),
      )
    : evidence;

  const citedCount = evidence.filter((item) => item.used_in_answer).length;

  return (
    <div className={cn('flex h-full min-h-0 flex-col', className)}>
      <div className="shrink-0 border-b border-ink-100 px-4 py-3">
        <div className="flex items-center justify-between gap-2">
          <h2 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-700">
            <Quote className="h-3.5 w-3.5 text-ink-400" aria-hidden="true" />
            Evidence &amp; sources
          </h2>
          {evidence.length ? (
            <span className="shrink-0 text-[11px] text-ink-500">
              {citedCount} cited / {evidence.length}
            </span>
          ) : null}
        </div>

        {evidence.length > 4 ? (
          <div className="relative mt-2.5">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-400"
              aria-hidden="true"
            />
            <input
              type="search"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Filter evidence…"
              className="w-full rounded-lg border border-ink-200 bg-white py-1.5 pl-8 pr-2.5 text-xs text-ink-800 placeholder:text-ink-400 focus:border-brand-400 focus:outline-none focus:ring-1 focus:ring-brand-400"
              aria-label="Filter evidence"
            />
          </div>
        ) : null}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {!evidence.length ? (
          <EmptyState
            icon={Quote}
            title="No evidence yet"
            description="Ask a question and the passages behind the answer will appear here, each traceable to its document, page and section."
            className="py-10"
          />
        ) : !filtered.length ? (
          <EmptyState
            icon={Search}
            title="No matching evidence"
            description="Clear the filter to see all retrieved passages."
            className="py-10"
          />
        ) : (
          <div className="space-y-2">
            {filtered.map((item) => (
              <EvidenceCard
                key={item.chunk_id}
                item={item}
                active={activeChunkId === item.chunk_id}
                onOpen={(selected) => {
                  setModalItem(selected);
                  onSelect?.(selected.chunk_id);
                }}
              />
            ))}
          </div>
        )}
      </div>

      {evidence.length ? (
        <div className="shrink-0 border-t border-ink-100 px-4 py-2.5">
          <p className="flex items-center gap-1.5 text-[11px] text-ink-500">
            <ExternalLink className="h-3 w-3" aria-hidden="true" />
            Select a card to read the full passage in context.
          </p>
        </div>
      ) : null}

      <EvidenceModal item={modalItem} open={Boolean(modalItem)} onClose={() => setModalItem(null)} />
    </div>
  );
}
