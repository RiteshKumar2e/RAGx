import { useEffect, useRef, useState } from 'react';
import { ChevronDown, CornerDownLeft, Send, SlidersHorizontal, Square } from 'lucide-react';
import { STRATEGIES, STRATEGY_ORDER } from '../../utils/constants';
import cn from '../../utils/cn';

const PINNABLE = STRATEGY_ORDER.filter((name) => name !== 'ragx' && name !== 'adaptive');

/**
 * The research query composer.
 *
 * Adaptive routing is the default. Pinning strategies is available but framed
 * as an override for comparison, so the default path stays the intended one.
 */
export default function QueryInput({
  onSubmit,
  onCancel,
  busy,
  disabled,
  disabledReason,
  documents = [],
}) {
  const [value, setValue] = useState('');
  const [showOptions, setShowOptions] = useState(false);
  const [strategies, setStrategies] = useState([]);
  const [documentIds, setDocumentIds] = useState([]);
  const [topK, setTopK] = useState('');
  const textareaRef = useRef(null);

  // Grow the textarea with its content, up to a ceiling.
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    const question = value.trim();
    if (!question || busy || disabled) return;
    onSubmit({
      question,
      strategies: strategies.length ? strategies : undefined,
      documentIds: documentIds.length ? documentIds : undefined,
      topK: topK ? Number(topK) : undefined,
    });
    setValue('');
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const toggle = (list, setList, item) =>
    setList(list.includes(item) ? list.filter((entry) => entry !== item) : [...list, item]);

  return (
    <div className="border-t border-ink-200 bg-white p-3 sm:p-4">
      {/* Advanced options */}
      {showOptions ? (
        <div className="mb-3 animate-fade-in space-y-3 rounded-xl border border-ink-200 bg-ink-50/60 p-3">
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-ink-600">
                Pin strategies
              </p>
              {strategies.length ? (
                <button
                  type="button"
                  onClick={() => setStrategies([])}
                  className="text-[11px] font-medium text-brand-600 hover:text-brand-700"
                >
                  Reset to adaptive
                </button>
              ) : null}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {PINNABLE.map((name) => {
                const meta = STRATEGIES[name];
                const active = strategies.includes(name);
                return (
                  <button
                    key={name}
                    type="button"
                    onClick={() => toggle(strategies, setStrategies, name)}
                    className={cn(
                      'rounded-full px-2.5 py-1 text-[11px] font-medium transition ring-1 ring-inset',
                      active ? 'text-white' : 'bg-white text-ink-600 ring-ink-200 hover:bg-ink-50',
                    )}
                    style={
                      active
                        ? { backgroundColor: meta.color, borderColor: meta.color, '--tw-ring-color': meta.color }
                        : undefined
                    }
                    title={meta.description}
                  >
                    {meta.label}
                  </button>
                );
              })}
            </div>
            <p className="mt-1.5 text-[11px] text-ink-500">
              {strategies.length
                ? 'Adaptive routing is bypassed — useful for comparing strategies on the same question.'
                : 'Leave empty to let the adaptive router choose (recommended).'}
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label
                htmlFor="topk-input"
                className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-ink-600"
              >
                Top-K passages
              </label>
              <input
                id="topk-input"
                type="number"
                min={1}
                max={30}
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
                placeholder="auto"
                className="w-full rounded-lg border border-ink-200 bg-white px-2.5 py-1.5 text-xs focus:border-brand-400 focus:outline-none focus:ring-1 focus:ring-brand-400"
              />
            </div>

            {documents.length ? (
              <div>
                <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-600">
                  Restrict to documents
                </p>
                <div className="max-h-24 space-y-1 overflow-y-auto rounded-lg border border-ink-200 bg-white p-1.5">
                  {documents.map((document) => (
                    <label
                      key={document.id}
                      className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-xs hover:bg-ink-50"
                    >
                      <input
                        type="checkbox"
                        checked={documentIds.includes(document.id)}
                        onChange={() => toggle(documentIds, setDocumentIds, document.id)}
                        className="h-3 w-3 rounded border-ink-300 text-brand-600 focus:ring-brand-500"
                      />
                      <span className="truncate text-ink-700">{document.filename}</span>
                    </label>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* Composer */}
      <div
        className={cn(
          'flex items-end gap-2 rounded-xl border bg-white p-2 transition',
          disabled ? 'border-ink-200 opacity-60' : 'border-ink-200 focus-within:border-brand-400 focus-within:ring-1 focus-within:ring-brand-400',
        )}
      >
        <button
          type="button"
          onClick={() => setShowOptions((open) => !open)}
          className={cn(
            'shrink-0 rounded-lg p-2 transition',
            showOptions ? 'bg-brand-50 text-brand-600' : 'text-ink-400 hover:bg-ink-100 hover:text-ink-600',
          )}
          aria-label="Retrieval options"
          aria-expanded={showOptions}
          title="Retrieval options"
        >
          <SlidersHorizontal className="h-4 w-4" />
          {strategies.length ? (
            <span className="sr-only">{strategies.length} strategies pinned</span>
          ) : null}
        </button>

        <label htmlFor="research-question" className="sr-only">
          Ask a research question
        </label>
        <textarea
          id="research-question"
          ref={textareaRef}
          rows={1}
          value={value}
          disabled={disabled}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            disabled
              ? disabledReason || 'Unavailable'
              : 'Ask a research question — try a simple lookup and a multi-hop question to compare routing…'
          }
          className="max-h-[200px] min-h-[2.25rem] flex-1 resize-none border-0 bg-transparent px-1 py-1.5 text-sm text-ink-900 placeholder:text-ink-400 focus:outline-none disabled:cursor-not-allowed"
        />

        {busy ? (
          <button
            type="button"
            onClick={onCancel}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-ink-100 px-3 py-2 text-xs font-medium text-ink-700 transition hover:bg-ink-200"
          >
            <Square className="h-3 w-3 fill-current" aria-hidden="true" />
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!value.trim() || disabled}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-brand-600 px-3 py-2 text-xs font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-ink-200 disabled:text-ink-400"
          >
            <Send className="h-3.5 w-3.5" aria-hidden="true" />
            Send
          </button>
        )}
      </div>

      <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 px-1">
        <p className="flex items-center gap-1 text-[11px] text-ink-400">
          <CornerDownLeft className="h-3 w-3" aria-hidden="true" />
          Enter to send · Shift+Enter for a new line
        </p>
        {strategies.length ? (
          <p className="text-[11px] font-medium text-amber-600">
            Adaptive routing bypassed ({strategies.length} pinned)
          </p>
        ) : (
          <p className="flex items-center gap-1 text-[11px] text-ink-400">
            Adaptive routing
            <ChevronDown className="h-3 w-3 rotate-[-90deg]" aria-hidden="true" />
            strategies chosen per query
          </p>
        )}
      </div>
    </div>
  );
}
