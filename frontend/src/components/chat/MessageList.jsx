import { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  ChevronDown,
  Clock,
  Coins,
  Copy,
  Check,
  FileStack,
  Search,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import MarkdownAnswer from './MarkdownAnswer';
import StrategyBadges from '../retrieval/StrategyBadges';
import WhyThisAnswerPanel from '../retrieval/WhyThisAnswerPanel';
import { CONFIDENCE_STYLES } from '../../utils/constants';
import { formatMs, formatNumber, formatRatio, formatUsd } from '../../utils/format';
import cn from '../../utils/cn';

/** Confidence pill. Abstention is styled distinctly — it is a decision, not a low score. */
function ConfidenceBadge({ score, label, abstained }) {
  const style = CONFIDENCE_STYLES[abstained ? 'abstained' : label] || CONFIDENCE_STYLES.low;
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset',
        style.className,
      )}
      title={abstained ? 'The system declined to answer from the available evidence.' : undefined}
    >
      <ShieldCheck className="h-3 w-3" aria-hidden="true" />
      {style.label}
      {!abstained ? ` · ${formatRatio(score, 2)}` : ''}
    </span>
  );
}

function UserMessage({ text }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-ink-900 px-4 py-2.5 text-sm text-white sm:max-w-[75%]">
        {text}
      </div>
    </div>
  );
}

/** Live pipeline status while a query runs. */
function PendingMessage({ stage, partial }) {
  const stages = [
    { key: 'analyzing', label: 'Analysing the query', icon: Search },
    { key: 'retrieved', label: 'Retrieving evidence', icon: FileStack },
    { key: 'generating', label: 'Composing a grounded answer', icon: Sparkles },
  ];
  const activeIndex = Math.max(0, stages.findIndex((item) => item.key === stage));

  return (
    <div className="rounded-2xl rounded-bl-md border border-ink-200 bg-white p-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {stages.map((item, index) => (
          <span
            key={item.key}
            className={cn(
              'inline-flex items-center gap-1.5 text-xs',
              index < activeIndex ? 'text-emerald-600' : index === activeIndex ? 'text-brand-600' : 'text-ink-300',
            )}
          >
            <item.icon
              className={cn('h-3.5 w-3.5', index === activeIndex && 'animate-pulse-subtle')}
              aria-hidden="true"
            />
            {item.label}
            {index < activeIndex ? <Check className="h-3 w-3" aria-hidden="true" /> : null}
          </span>
        ))}
      </div>

      {partial ? (
        <div className="mt-3 border-t border-ink-100 pt-3">
          <MarkdownAnswer content={partial} />
        </div>
      ) : (
        <div className="mt-3 space-y-2">
          <div className="h-3 w-3/4 animate-pulse-subtle rounded bg-ink-100" />
          <div className="h-3 w-full animate-pulse-subtle rounded bg-ink-100" />
          <div className="h-3 w-1/2 animate-pulse-subtle rounded bg-ink-100" />
        </div>
      )}
    </div>
  );
}

function AssistantMessage({ message, onCitationClick, onEvidenceFocus }) {
  const [showWhy, setShowWhy] = useState(false);
  const [copied, setCopied] = useState(false);

  const { answer, citations, why, abstained, confidence, confidence_label: label } = message;
  const trace = message.trace;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* Clipboard unavailable — the button simply does nothing. */
    }
  };

  return (
    <div
      className={cn(
        'rounded-2xl rounded-bl-md border bg-white',
        abstained ? 'border-amber-200' : 'border-ink-200',
      )}
    >
      {/* Strategy header */}
      <div className="border-b border-ink-100 px-4 py-3">
        <StrategyBadges
          strategies={message.strategies}
          analysis={why?.analysis}
          routing={why?.routing}
        />
      </div>

      {/* Abstention notice */}
      {abstained ? (
        <div className="flex items-start gap-2.5 border-b border-amber-200 bg-amber-50 px-4 py-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" aria-hidden="true" />
          <div>
            <p className="text-xs font-semibold text-amber-900">Answer withheld</p>
            <p className="mt-0.5 text-xs leading-relaxed text-amber-800">
              The retrieved evidence did not support a reliable answer, so RAGX declined rather than
              guessing. The passages it did find are listed in the evidence panel.
            </p>
          </div>
        </div>
      ) : null}

      {/* The answer */}
      <div className="px-4 py-4">
        <MarkdownAnswer
          content={answer}
          citations={citations}
          onCitationClick={(citation) => {
            onCitationClick?.(citation);
            onEvidenceFocus?.(citation.chunk_id);
          }}
        />
      </div>

      {/* Metrics footer */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-ink-100 px-4 py-2.5">
        <ConfidenceBadge score={confidence} label={label} abstained={abstained} />

        <span className="inline-flex items-center gap-1 text-[11px] text-ink-500">
          <FileStack className="h-3 w-3" aria-hidden="true" />
          {why?.retrieval?.chunks_retrieved ?? citations?.length ?? 0} passages ·{' '}
          {why?.retrieval?.documents_used?.length ?? 0} docs
        </span>

        <span className="inline-flex items-center gap-1 text-[11px] text-ink-500">
          <Clock className="h-3 w-3" aria-hidden="true" />
          {formatMs(message.total_latency_ms)}
        </span>

        {trace?.total_tokens ? (
          <span className="inline-flex items-center gap-1 text-[11px] text-ink-500">
            <Coins className="h-3 w-3" aria-hidden="true" />
            {formatNumber(trace.total_tokens)} tokens · {formatUsd(trace.estimated_cost_usd)}
          </span>
        ) : null}

        {why?.retrieval?.corrective_triggered ? (
          <span className="rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-700">
            corrective ×{why.retrieval.corrective_rounds}
          </span>
        ) : null}

        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={copy}
            className="inline-flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-ink-500 transition hover:bg-ink-100 hover:text-ink-700"
            aria-label="Copy answer"
          >
            {copied ? (
              <>
                <Check className="h-3 w-3" aria-hidden="true" /> Copied
              </>
            ) : (
              <>
                <Copy className="h-3 w-3" aria-hidden="true" /> Copy
              </>
            )}
          </button>

          {why ? (
            <button
              type="button"
              onClick={() => setShowWhy((open) => !open)}
              className="inline-flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-brand-600 transition hover:bg-brand-50"
              aria-expanded={showWhy}
            >
              Why this answer?
              <ChevronDown
                className={cn('h-3 w-3 transition-transform', showWhy && 'rotate-180')}
                aria-hidden="true"
              />
            </button>
          ) : null}
        </div>
      </div>

      {showWhy && why ? (
        <div className="animate-fade-in border-t border-ink-100 p-4">
          <WhyThisAnswerPanel why={why} />
        </div>
      ) : null}
    </div>
  );
}

export default function MessageList({ messages, pending, pendingStage, partialAnswer, onCitationClick, onEvidenceFocus }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages.length, pending, partialAnswer]);

  return (
    <div className="space-y-5">
      {messages.map((message) =>
        message.role === 'user' ? (
          <UserMessage key={message.id} text={message.content} />
        ) : (
          <AssistantMessage
            key={message.id}
            message={message}
            onCitationClick={onCitationClick}
            onEvidenceFocus={onEvidenceFocus}
          />
        ),
      )}

      {pending ? <PendingMessage stage={pendingStage} partial={partialAnswer} /> : null}

      <div ref={endRef} />
    </div>
  );
}
