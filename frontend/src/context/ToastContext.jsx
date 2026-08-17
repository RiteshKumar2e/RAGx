import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react';
import cn from '../utils/cn';

const ToastContext = createContext(null);

const VARIANTS = {
  success: { icon: CheckCircle2, className: 'border-emerald-200 bg-emerald-50', iconClass: 'text-emerald-600' },
  error: { icon: XCircle, className: 'border-rose-200 bg-rose-50', iconClass: 'text-rose-600' },
  warning: { icon: AlertTriangle, className: 'border-amber-200 bg-amber-50', iconClass: 'text-amber-600' },
  info: { icon: Info, className: 'border-brand-200 bg-brand-50', iconClass: 'text-brand-600' },
};

let nextId = 0;

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const dismiss = useCallback((id) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (message, { variant = 'info', title, duration = 5000 } = {}) => {
      const id = ++nextId;
      setToasts((current) => [...current, { id, message, variant, title }]);
      if (duration) setTimeout(() => dismiss(id), duration);
      return id;
    },
    [dismiss],
  );

  const value = useMemo(
    () => ({
      push,
      dismiss,
      success: (message, options) => push(message, { ...options, variant: 'success' }),
      error: (message, options) => push(message, { ...options, variant: 'error', duration: 8000 }),
      warning: (message, options) => push(message, { ...options, variant: 'warning', duration: 7000 }),
      info: (message, options) => push(message, { ...options, variant: 'info' }),
    }),
    [push, dismiss],
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        className="pointer-events-none fixed bottom-4 right-4 z-[100] flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2"
        role="region"
        aria-label="Notifications"
      >
        {toasts.map((toast) => {
          const variant = VARIANTS[toast.variant] || VARIANTS.info;
          const Icon = variant.icon;
          return (
            <div
              key={toast.id}
              role="status"
              aria-live="polite"
              className={cn(
                'pointer-events-auto flex animate-fade-in items-start gap-3 rounded-xl border p-3 shadow-panel',
                variant.className,
              )}
            >
              <Icon className={cn('mt-0.5 h-4 w-4 shrink-0', variant.iconClass)} aria-hidden="true" />
              <div className="min-w-0 flex-1">
                {toast.title ? (
                  <p className="text-sm font-semibold text-ink-900">{toast.title}</p>
                ) : null}
                <p className="break-words text-sm text-ink-700">{toast.message}</p>
              </div>
              <button
                type="button"
                onClick={() => dismiss(toast.id)}
                className="rounded p-0.5 text-ink-400 transition hover:bg-white/60 hover:text-ink-700"
                aria-label="Dismiss notification"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

// See the note in SystemContext.jsx — provider and hook intentionally share a file.
// eslint-disable-next-line react-refresh/only-export-components
export function useToast() {
  const context = useContext(ToastContext);
  if (!context) throw new Error('useToast must be used inside a ToastProvider.');
  return context;
}

export default ToastContext;
