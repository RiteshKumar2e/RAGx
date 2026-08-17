import { useState } from 'react';
import { Cloud, CloudOff, Info } from 'lucide-react';
import { useSystem } from '../../context/SystemContext';
import cn from '../../utils/cn';

/**
 * Cloud LLM provider indicator.
 *
 * Shows which providers are configured and which models are active. It renders
 * only what the backend reports — no key material is ever sent to the browser.
 * Local model runtimes are not part of RAGX, so none can appear here.
 */
export default function ProviderIndicator({ compact = false }) {
  const { settings, hasLlm, loading } = useSystem();
  const [open, setOpen] = useState(false);

  const providers = settings?.llm?.providers || [];
  const primary = settings?.llm?.primary;

  if (loading) {
    return <div className="h-8 w-28 animate-pulse-subtle rounded-lg bg-ink-100" aria-hidden="true" />;
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        className={cn(
          'inline-flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition',
          hasLlm
            ? 'border-ink-200 bg-white text-ink-700 hover:bg-ink-50'
            : 'border-amber-200 bg-amber-50 text-amber-800 hover:bg-amber-100',
        )}
        aria-expanded={open}
        aria-haspopup="true"
      >
        {hasLlm ? (
          <Cloud className="h-3.5 w-3.5 text-brand-600" aria-hidden="true" />
        ) : (
          <CloudOff className="h-3.5 w-3.5 text-amber-600" aria-hidden="true" />
        )}
        {compact ? null : <span>{hasLlm ? 'Cloud LLM' : 'No LLM key'}</span>}
        <span className="flex items-center gap-1" aria-hidden="true">
          {providers.map((provider) => (
            <span
              key={provider.provider}
              title={`${provider.provider}: ${provider.configured ? 'configured' : 'not configured'}`}
              className={cn(
                'h-1.5 w-1.5 rounded-full',
                provider.configured ? 'bg-emerald-500' : 'bg-ink-300',
              )}
            />
          ))}
        </span>
      </button>

      {open ? (
        <div className="absolute right-0 top-full z-40 mt-2 w-72 animate-fade-in rounded-xl border border-ink-200 bg-white p-3 shadow-panel">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-500">
            LLM Provider
          </p>

          <ul className="space-y-2">
            {providers.map((provider) => (
              <li key={provider.provider} className="flex items-start gap-2">
                <span
                  className={cn(
                    'mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full',
                    provider.configured ? 'bg-emerald-500' : 'bg-ink-300',
                  )}
                  aria-hidden="true"
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm font-medium capitalize text-ink-900">
                      {provider.provider} API
                    </span>
                    {provider.provider === primary ? (
                      <span className="rounded bg-brand-50 px-1 py-0.5 text-[10px] font-medium text-brand-700">
                        primary
                      </span>
                    ) : null}
                  </div>
                  <p className="truncate font-mono text-[11px] text-ink-500">{provider.model}</p>
                  <p className="text-[11px] text-ink-500">
                    {provider.configured ? 'Configured' : 'No API key set'}
                    {provider.multimodal ? ' · vision' : ''}
                  </p>
                </div>
              </li>
            ))}
          </ul>

          <div className="mt-3 flex items-start gap-1.5 border-t border-ink-100 pt-2.5">
            <Info className="mt-0.5 h-3 w-3 shrink-0 text-ink-400" aria-hidden="true" />
            <p className="text-[11px] leading-relaxed text-ink-500">
              RAGX uses cloud LLM APIs only. Keys are held by the backend and are never sent to
              the browser.
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
