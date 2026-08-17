import { useState } from 'react';
import { ChevronDown, HelpCircle, Plus, Zap } from 'lucide-react';
import { COMPLEXITY_STYLES } from '../../utils/constants';
import { orderStrategies, routingCostLabel, routingHighlights, strategyMeta } from '../../utils/strategy';
import cn from '../../utils/cn';

/** A single strategy chip, tinted with that strategy's colour. */
export function StrategyChip({ name, size = 'sm', showCheck = true }) {
  const meta = strategyMeta(name);
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full font-medium ring-1 ring-inset',
        size === 'xs' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs',
      )}
      style={{
        backgroundColor: `${meta.color}12`,
        color: meta.color,
        // eslint-disable-next-line no-underscore-dangle
        '--tw-ring-color': `${meta.color}33`,
      }}
      title={meta.description}
    >
      <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: meta.color }} aria-hidden="true" />
      {meta.label}
      {showCheck ? <span aria-hidden="true">✓</span> : null}
    </span>
  );
}

/**
 * The selected-strategy display with the "Why these strategies?" disclosure.
 *
 * This is the project's central claim made visible: the composition of
 * strategies varies per query, and the reason is always available.
 */
export default function StrategyBadges({ strategies = [], analysis, routing, compact = false }) {
  const [open, setOpen] = useState(false);
  const ordered = orderStrategies(strategies);
  if (!ordered.length) return null;

  const highlights = routingHighlights(analysis, routing);
  const cost = routingCostLabel(routing);

  return (
    <div className="w-full">
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-2">
        <span className="mr-1 text-[11px] font-medium uppercase tracking-wide text-ink-400">
          Strategy
        </span>

        {ordered.map((name, index) => (
          <span key={name} className="flex items-center gap-1.5">
            {index > 0 ? <Plus className="h-3 w-3 shrink-0 text-ink-300" aria-hidden="true" /> : null}
            <StrategyChip name={name} size={compact ? 'xs' : 'sm'} />
          </span>
        ))}

        {routing ? (
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="ml-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium text-brand-700 transition hover:bg-brand-50"
            aria-expanded={open}
          >
            <HelpCircle className="h-3 w-3" aria-hidden="true" />
            Why these strategies?
            <ChevronDown
              className={cn('h-3 w-3 transition-transform', open && 'rotate-180')}
              aria-hidden="true"
            />
          </button>
        ) : null}
      </div>

      {open && routing ? (
        <div className="mt-3 animate-fade-in rounded-xl border border-ink-200 bg-ink-50/70 p-4">
          {/* Query characterisation */}
          {highlights.length ? (
            <dl className="grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2">
              {highlights.map((row) => (
                <div key={row.label} className="flex items-center justify-between gap-3 text-xs">
                  <dt className="text-ink-500">{row.label}</dt>
                  <dd className="shrink-0 font-medium capitalize text-ink-900">
                    {row.label === 'Query complexity' ? (
                      <span
                        className={cn(
                          'rounded px-1.5 py-0.5 text-[11px] ring-1 ring-inset',
                          COMPLEXITY_STYLES[row.value]?.className || 'bg-ink-100 text-ink-700 ring-ink-600/20',
                        )}
                      >
                        {COMPLEXITY_STYLES[row.value]?.label || row.value}
                      </span>
                    ) : (
                      row.value
                    )}
                  </dd>
                </div>
              ))}
            </dl>
          ) : null}

          {/* Rules the router actually fired */}
          {routing.rules_fired?.length ? (
            <div className="mt-4 border-t border-ink-200 pt-3">
              <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
                Routing rules applied
              </p>
              <ul className="space-y-2">
                {routing.rules_fired.map((rule) => (
                  <li key={rule.rule} className="flex gap-2">
                    <span
                      className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-brand-500"
                      aria-hidden="true"
                    />
                    <div className="min-w-0">
                      <p className="font-mono text-[11px] text-brand-700">{rule.rule}</p>
                      <p className="mt-0.5 text-xs leading-relaxed text-ink-600">{rule.reason}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {/* Cost posture — the point of routing is to not over-spend */}
          <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-ink-200 pt-3 text-[11px] text-ink-500">
            <span className="inline-flex items-center gap-1">
              <Zap className="h-3 w-3 text-amber-500" aria-hidden="true" />
              Relative cost: <span className="font-medium text-ink-700">{cost}</span>
            </span>
            <span>
              Mode: <span className="font-medium text-ink-700">{routing.mode}</span>
            </span>
            <span>
              Strategies used:{' '}
              <span className="font-medium text-ink-700">
                {ordered.length} of 8
              </span>
            </span>
          </div>

          {routing.reason ? (
            <p className="mt-3 border-t border-ink-200 pt-3 text-xs italic leading-relaxed text-ink-600">
              {routing.reason}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
