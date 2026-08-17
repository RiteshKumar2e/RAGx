import { forwardRef } from 'react';
import { AlertCircle, Inbox, Loader2, RefreshCw, WifiOff } from 'lucide-react';
import cn from '../../utils/cn';

/* ========================================================================== */
/* Card                                                                       */
/* ========================================================================== */
export function Card({ className, children, as: Tag = 'div', ...props }) {
  return (
    <Tag
      className={cn(
        'rounded-xl border border-ink-200/70 bg-white shadow-card transition-shadow',
        className,
      )}
      {...props}
    >
      {children}
    </Tag>
  );
}

export function CardHeader({ title, description, action, icon: Icon, className }) {
  return (
    <div className={cn('flex items-start justify-between gap-4 border-b border-ink-100 p-4 sm:p-5', className)}>
      <div className="flex min-w-0 items-start gap-3">
        {Icon ? (
          <span className="mt-0.5 rounded-lg bg-brand-50 p-2 text-brand-600">
            <Icon className="h-4 w-4" aria-hidden="true" />
          </span>
        ) : null}
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold text-ink-900">{title}</h2>
          {description ? <p className="mt-0.5 text-xs text-ink-500">{description}</p> : null}
        </div>
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}

export function CardBody({ className, children }) {
  return <div className={cn('p-4 sm:p-5', className)}>{children}</div>;
}

/* ========================================================================== */
/* Button                                                                     */
/* ========================================================================== */
const BUTTON_VARIANTS = {
  primary: 'bg-brand-600 text-white hover:bg-brand-700 focus-visible:outline-brand-600 disabled:bg-brand-300',
  secondary:
    'bg-white text-ink-800 ring-1 ring-inset ring-ink-200 hover:bg-ink-50 focus-visible:outline-ink-400 disabled:text-ink-400',
  ghost: 'text-ink-600 hover:bg-ink-100 hover:text-ink-900 focus-visible:outline-ink-400',
  danger: 'bg-rose-600 text-white hover:bg-rose-700 focus-visible:outline-rose-600 disabled:bg-rose-300',
  subtle: 'bg-brand-50 text-brand-700 hover:bg-brand-100 focus-visible:outline-brand-600',
};

const BUTTON_SIZES = {
  xs: 'px-2 py-1 text-xs gap-1',
  sm: 'px-2.5 py-1.5 text-xs gap-1.5',
  md: 'px-3.5 py-2 text-sm gap-2',
  lg: 'px-5 py-2.5 text-sm gap-2',
};

export const Button = forwardRef(function Button(
  { variant = 'primary', size = 'md', icon: Icon, loading, className, children, disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(
        'inline-flex items-center justify-center rounded-lg font-medium transition-colors',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2',
        'disabled:cursor-not-allowed disabled:opacity-70',
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className,
      )}
      {...props}
    >
      {loading ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
      ) : Icon ? (
        <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      ) : null}
      {children}
    </button>
  );
});

/* ========================================================================== */
/* Badge                                                                      */
/* ========================================================================== */
export function Badge({ children, className, color, dot, size = 'sm' }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full font-medium ring-1 ring-inset',
        size === 'xs' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-0.5 text-xs',
        className || 'bg-ink-100 text-ink-700 ring-ink-600/10',
      )}
    >
      {dot || color ? (
        <span
          className="h-1.5 w-1.5 shrink-0 rounded-full"
          style={color ? { backgroundColor: color } : undefined}
          aria-hidden="true"
        />
      ) : null}
      {children}
    </span>
  );
}

/* ========================================================================== */
/* Loading states                                                             */
/* ========================================================================== */
export function Spinner({ className, label = 'Loading' }) {
  return (
    <span role="status" aria-label={label}>
      <Loader2 className={cn('h-4 w-4 animate-spin text-brand-600', className)} aria-hidden="true" />
    </span>
  );
}

export function Skeleton({ className }) {
  return (
    <div
      className={cn(
        'animate-pulse-subtle rounded-md bg-gradient-to-r from-ink-100 via-ink-50 to-ink-100 bg-[length:400px_100%]',
        className,
      )}
      aria-hidden="true"
    />
  );
}

export function LoadingBlock({ rows = 3, className }) {
  return (
    <div className={cn('space-y-3', className)} role="status" aria-label="Loading content">
      {Array.from({ length: rows }).map((_, index) => (
        <Skeleton key={index} className={cn('h-4', index === rows - 1 ? 'w-2/3' : 'w-full')} />
      ))}
    </div>
  );
}

export function LoadingCards({ count = 4, className }) {
  return (
    <div className={cn('grid gap-4 sm:grid-cols-2 lg:grid-cols-4', className)}>
      {Array.from({ length: count }).map((_, index) => (
        <Card key={index} className="p-4">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="mt-3 h-7 w-16" />
          <Skeleton className="mt-2 h-3 w-32" />
        </Card>
      ))}
    </div>
  );
}

/* ========================================================================== */
/* Empty and error states                                                     */
/* ========================================================================== */
export function EmptyState({ icon: Icon = Inbox, title, description, action, className }) {
  return (
    <div className={cn('flex flex-col items-center justify-center px-6 py-12 text-center', className)}>
      <span className="rounded-full bg-ink-100 p-3 text-ink-400">
        <Icon className="h-6 w-6" aria-hidden="true" />
      </span>
      <h3 className="mt-4 text-sm font-semibold text-ink-900">{title}</h3>
      {description ? <p className="mt-1 max-w-md text-sm text-ink-500">{description}</p> : null}
      {action ? <div className="mt-5">{action}</div> : null}
    </div>
  );
}

/**
 * Renders a normalised ApiError. Never shows a stack trace — `error.message`
 * is already a user-facing sentence produced by the API client.
 */
export function ErrorState({ error, onRetry, className, compact }) {
  const isNetwork = error?.code === 'network_error' || error?.status === 0;
  const Icon = isNetwork ? WifiOff : AlertCircle;
  const message = error?.message || 'Something went wrong.';

  if (compact) {
    return (
      <div
        role="alert"
        className={cn('flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 p-3', className)}
      >
        <Icon className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" aria-hidden="true" />
        <p className="text-sm text-rose-800">{message}</p>
        {onRetry ? (
          <button
            type="button"
            onClick={onRetry}
            className="ml-auto shrink-0 text-xs font-medium text-rose-700 underline hover:text-rose-900"
          >
            Retry
          </button>
        ) : null}
      </div>
    );
  }

  return (
    <div
      role="alert"
      className={cn('flex flex-col items-center justify-center px-6 py-12 text-center', className)}
    >
      <span className="rounded-full bg-rose-50 p-3 text-rose-500">
        <Icon className="h-6 w-6" aria-hidden="true" />
      </span>
      <h3 className="mt-4 text-sm font-semibold text-ink-900">
        {isNetwork ? 'Cannot reach the backend' : 'Something went wrong'}
      </h3>
      <p className="mt-1 max-w-md text-sm text-ink-500">{message}</p>
      {Array.isArray(error?.detail) && error.detail.length ? (
        <ul className="mt-3 max-w-md space-y-1 text-left text-xs text-ink-500">
          {error.detail.slice(0, 5).map((item, index) => (
            <li key={index}>• {item.reason || item.issue || JSON.stringify(item)}</li>
          ))}
        </ul>
      ) : null}
      {onRetry ? (
        <Button variant="secondary" size="sm" icon={RefreshCw} onClick={onRetry} className="mt-5">
          Try again
        </Button>
      ) : null}
    </div>
  );
}

/* ========================================================================== */
/* Stat tile                                                                  */
/* ========================================================================== */
export function StatTile({ label, value, hint, icon: Icon, tone = 'brand', loading }) {
  const tones = {
    brand: 'bg-brand-50 text-brand-600',
    emerald: 'bg-emerald-50 text-emerald-600',
    amber: 'bg-amber-50 text-amber-600',
    violet: 'bg-violet-50 text-violet-600',
    rose: 'bg-rose-50 text-rose-600',
    ink: 'bg-ink-100 text-ink-600',
  };

  return (
    <Card className="p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-xs font-medium uppercase tracking-wide text-ink-500">{label}</p>
          {loading ? (
            <Skeleton className="mt-2 h-7 w-20" />
          ) : (
            <p className="mt-1 text-2xl font-semibold tabular-nums text-ink-900">{value}</p>
          )}
          {hint ? <p className="mt-1 truncate text-xs text-ink-500">{hint}</p> : null}
        </div>
        {Icon ? (
          <span className={cn('shrink-0 rounded-lg p-2', tones[tone])}>
            <Icon className="h-4 w-4" aria-hidden="true" />
          </span>
        ) : null}
      </div>
    </Card>
  );
}

/* ========================================================================== */
/* Progress bar                                                               */
/* ========================================================================== */
export function ProgressBar({ value = 0, className, barClassName, label }) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div
      className={cn('h-1.5 w-full overflow-hidden rounded-full bg-ink-100', className)}
      role="progressbar"
      aria-valuenow={Math.round(clamped)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label || 'Progress'}
    >
      <div
        className={cn('h-full rounded-full bg-brand-600 transition-[width] duration-300', barClassName)}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}

/* ========================================================================== */
/* Section heading                                                            */
/* ========================================================================== */
export function PageHeader({ title, description, action, children }) {
  return (
    <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight text-ink-900 sm:text-2xl">{title}</h1>
        {description ? <p className="mt-1 max-w-2xl text-sm text-ink-500">{description}</p> : null}
        {children}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}

/* ========================================================================== */
/* Info banner                                                                */
/* ========================================================================== */
export function InfoBanner({ variant = 'info', title, children, icon: Icon = AlertCircle, action, className }) {
  const variants = {
    info: 'border-brand-200 bg-brand-50 text-brand-900',
    warning: 'border-amber-200 bg-amber-50 text-amber-900',
    danger: 'border-rose-200 bg-rose-50 text-rose-900',
    neutral: 'border-ink-200 bg-ink-50 text-ink-800',
  };
  const iconTone = {
    info: 'text-brand-600',
    warning: 'text-amber-600',
    danger: 'text-rose-600',
    neutral: 'text-ink-500',
  };

  return (
    <div className={cn('flex items-start gap-3 rounded-xl border p-3 sm:p-4', variants[variant], className)}>
      <Icon className={cn('mt-0.5 h-4 w-4 shrink-0', iconTone[variant])} aria-hidden="true" />
      <div className="min-w-0 flex-1 text-sm">
        {title ? <p className="font-semibold">{title}</p> : null}
        <div className={cn(title && 'mt-0.5', 'opacity-90')}>{children}</div>
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}
