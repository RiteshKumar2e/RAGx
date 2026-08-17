import { AlertCircle, Check, Circle, Loader2, MinusCircle } from 'lucide-react';
import { PROCESSING_STEPS } from '../../utils/constants';
import { formatMs } from '../../utils/format';
import cn from '../../utils/cn';

const ICONS = {
  completed: { Icon: Check, className: 'text-emerald-600' },
  running: { Icon: Loader2, className: 'text-brand-600 animate-spin' },
  failed: { Icon: AlertCircle, className: 'text-rose-600' },
  skipped: { Icon: MinusCircle, className: 'text-ink-300' },
  pending: { Icon: Circle, className: 'text-ink-200' },
};

/**
 * The ingestion pipeline as a live checklist.
 *
 * The backend writes a status and duration per stage, so this is a real
 * progress view rather than an animation.
 */
export default function ProcessingSteps({ steps = {}, compact = false, errorMessage }) {
  return (
    <div className={cn(compact ? 'space-y-1' : 'space-y-1.5')}>
      {PROCESSING_STEPS.map((step) => {
        const state = steps[step.key] || { status: 'pending' };
        const { Icon, className } = ICONS[state.status] || ICONS.pending;

        return (
          <div key={step.key} className="flex items-center gap-2.5">
            <Icon className={cn('h-3.5 w-3.5 shrink-0', className)} aria-hidden="true" />
            <span
              className={cn(
                'min-w-0 flex-1 truncate',
                compact ? 'text-[11px]' : 'text-xs',
                state.status === 'completed'
                  ? 'text-ink-700'
                  : state.status === 'running'
                    ? 'font-medium text-ink-900'
                    : state.status === 'failed'
                      ? 'text-rose-700'
                      : 'text-ink-400',
              )}
            >
              {step.label}
            </span>

            {state.duration_ms ? (
              <span className="shrink-0 font-mono text-[10px] tabular-nums text-ink-400">
                {formatMs(state.duration_ms)}
              </span>
            ) : null}
          </div>
        );
      })}

      {/* Stage detail — chunk counts, entity counts, failure reasons. */}
      {!compact
        ? PROCESSING_STEPS.map((step) => {
            const state = steps[step.key];
            if (!state?.detail || state.status === 'pending') return null;
            return (
              <p
                key={`${step.key}-detail`}
                className={cn(
                  'pl-6 text-[11px] leading-relaxed',
                  state.status === 'failed' ? 'text-rose-600' : 'text-ink-500',
                )}
              >
                {state.detail}
              </p>
            );
          })
        : null}

      {errorMessage ? (
        <p className="mt-2 rounded-lg border border-rose-200 bg-rose-50 p-2 text-[11px] leading-relaxed text-rose-700">
          {errorMessage}
        </p>
      ) : null}
    </div>
  );
}
