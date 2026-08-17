import { useEffect, useState } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { AlertTriangle, Menu, RefreshCw } from 'lucide-react';
import Sidebar from './Sidebar';
import ProviderIndicator from './ProviderIndicator';
import ErrorBoundary from '../common/ErrorBoundary';
import { useSystem } from '../../context/SystemContext';

const TITLES = {
  '/dashboard': 'Dashboard',
  '/research': 'Research Assistant',
  '/knowledge-base': 'Knowledge Base',
  '/graph': 'Knowledge Graph',
  '/evaluation': 'Evaluation',
  '/settings': 'Settings',
};

export default function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { hasLlm, loading, reload } = useSystem();
  const location = useLocation();

  // Close the mobile drawer on navigation.
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  const title =
    TITLES[location.pathname] ||
    (location.pathname.startsWith('/knowledge-base/') ? 'Document' : 'RAGX');

  return (
    <div className="min-h-screen bg-ink-50">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="lg:pl-64">
        {/* Top bar */}
        <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-ink-200 bg-white/90 px-4 backdrop-blur sm:px-6">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            className="rounded-lg p-1.5 text-ink-500 transition hover:bg-ink-100 lg:hidden"
            aria-label="Open navigation"
          >
            <Menu className="h-5 w-5" />
          </button>

          <h1 className="min-w-0 flex-1 truncate text-sm font-semibold text-ink-900">{title}</h1>

          <button
            type="button"
            onClick={reload}
            disabled={loading}
            className="hidden rounded-lg p-1.5 text-ink-400 transition hover:bg-ink-100 hover:text-ink-700 disabled:opacity-50 sm:block"
            aria-label="Refresh system status"
            title="Refresh system status"
          >
            <RefreshCw className={loading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
          </button>

          <ProviderIndicator />
        </header>

        {/* No-LLM banner: retrieval works, generation does not. */}
        {!loading && !hasLlm ? (
          <div className="border-b border-amber-200 bg-amber-50 px-4 py-2.5 sm:px-6">
            <div className="mx-auto flex max-w-7xl items-start gap-2.5">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" aria-hidden="true" />
              <p className="text-xs leading-relaxed text-amber-900">
                <span className="font-semibold">No cloud LLM provider is configured.</span>{' '}
                Document ingestion, retrieval and the evidence panel work as normal, but answers
                cannot be generated. Set <code className="font-mono">GEMINI_API_KEY</code> and/or{' '}
                <code className="font-mono">GROQ_API_KEY</code> in the backend environment.{' '}
                <Link to="/settings" className="font-medium underline hover:text-amber-950">
                  View Settings
                </Link>
              </p>
            </div>
          </div>
        ) : null}

        <main className="px-4 py-6 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <ErrorBoundary>
              <Outlet />
            </ErrorBoundary>
          </div>
        </main>
      </div>
    </div>
  );
}
