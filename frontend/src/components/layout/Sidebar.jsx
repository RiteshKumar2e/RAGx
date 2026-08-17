import { NavLink } from 'react-router-dom';
import {
  FlaskConical,
  LayoutDashboard,
  Library,
  MessageSquareText,
  Settings as SettingsIcon,
  Share2,
  X,
} from 'lucide-react';
import { useSystem } from '../../context/SystemContext';
import cn from '../../utils/cn';

const NAVIGATION = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard, description: 'System overview' },
  { to: '/research', label: 'Research Assistant', icon: MessageSquareText, description: 'Ask questions' },
  { to: '/knowledge-base', label: 'Knowledge Base', icon: Library, description: 'Manage documents' },
  { to: '/graph', label: 'Knowledge Graph', icon: Share2, description: 'Entities and relations' },
  { to: '/evaluation', label: 'Evaluation', icon: FlaskConical, description: 'Benchmark strategies' },
  { to: '/settings', label: 'Settings', icon: SettingsIcon, description: 'Providers and retrieval' },
];

export default function Sidebar({ open, onClose }) {
  const { settings, usingDevEmbedder } = useSystem();

  return (
    <>
      {/* Mobile backdrop */}
      {open ? (
        <div
          className="fixed inset-0 z-30 bg-ink-950/30 backdrop-blur-sm lg:hidden"
          onClick={onClose}
          role="presentation"
        />
      ) : null}

      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-ink-200 bg-white transition-transform duration-200 lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
        aria-label="Main navigation"
      >
        {/* Brand */}
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-ink-100 px-4">
          <NavLink to="/" className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-950 text-sm font-bold text-white">
              RX
            </span>
            <span className="flex flex-col leading-none">
              <span className="text-sm font-semibold tracking-tight text-ink-900">RAGX</span>
              <span className="mt-0.5 text-[10px] uppercase tracking-wider text-ink-400">
                Research Intelligence
              </span>
            </span>
          </NavLink>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-ink-400 transition hover:bg-ink-100 lg:hidden"
            aria-label="Close navigation"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto p-3">
          <ul className="space-y-0.5">
            {NAVIGATION.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  onClick={onClose}
                  className={({ isActive }) =>
                    cn(
                      'group flex items-start gap-3 rounded-lg px-3 py-2 text-sm transition',
                      isActive
                        ? 'bg-brand-50 font-medium text-brand-700'
                        : 'text-ink-600 hover:bg-ink-50 hover:text-ink-900',
                    )
                  }
                >
                  {({ isActive }) => (
                    <>
                      <item.icon
                        className={cn(
                          'mt-0.5 h-4 w-4 shrink-0',
                          isActive ? 'text-brand-600' : 'text-ink-400 group-hover:text-ink-600',
                        )}
                        aria-hidden="true"
                      />
                      <span className="min-w-0">
                        <span className="block truncate">{item.label}</span>
                        <span
                          className={cn(
                            'block truncate text-[11px]',
                            isActive ? 'text-brand-600/70' : 'text-ink-400',
                          )}
                        >
                          {item.description}
                        </span>
                      </span>
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Footer: storage backends in use */}
        <div className="shrink-0 border-t border-ink-100 p-3">
          {usingDevEmbedder ? (
            <div className="mb-2 rounded-lg border border-amber-200 bg-amber-50 p-2">
              <p className="text-[11px] font-medium text-amber-900">Development embedder active</p>
              <p className="mt-0.5 text-[10px] leading-snug text-amber-700">
                Lexical matching only. Set a Gemini key for semantic retrieval.
              </p>
            </div>
          ) : null}

          <dl className="space-y-1 text-[11px] text-ink-400">
            <div className="flex justify-between gap-2">
              <dt>Vectors</dt>
              <dd className="truncate font-medium text-ink-600">
                Qdrant · {settings?.storage?.vector_store?.mode || '—'}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>Graph</dt>
              <dd className="truncate font-medium text-ink-600">
                {settings?.storage?.graph_store?.backend || '—'}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>Database</dt>
              <dd className="truncate font-medium text-ink-600">
                {settings?.storage?.relational?.engine || '—'}
              </dd>
            </div>
          </dl>
        </div>
      </aside>
    </>
  );
}
