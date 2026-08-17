import { Component } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

/**
 * Catches render-time errors so one broken panel cannot blank the whole app.
 *
 * The user sees a recovery action, never a stack trace; details go to the
 * console for developers.
 */
export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error('[RAGX] Render error:', error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false });
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="flex min-h-[320px] flex-col items-center justify-center rounded-xl border border-ink-200 bg-white p-8 text-center">
        <span className="rounded-full bg-amber-50 p-3 text-amber-600">
          <AlertTriangle className="h-6 w-6" aria-hidden="true" />
        </span>
        <h2 className="mt-4 text-sm font-semibold text-ink-900">This section failed to render</h2>
        <p className="mt-1 max-w-md text-sm text-ink-500">
          {this.props.message ||
            'An unexpected error occurred while displaying this content. The rest of the application is unaffected.'}
        </p>
        <button
          type="button"
          onClick={this.handleReset}
          className="mt-5 inline-flex items-center gap-2 rounded-lg bg-white px-3.5 py-2 text-sm font-medium text-ink-800 ring-1 ring-inset ring-ink-200 transition hover:bg-ink-50"
        >
          <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
          Try again
        </button>
      </div>
    );
  }
}

export default ErrorBoundary;
