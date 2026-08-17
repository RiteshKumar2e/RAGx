import { SystemProvider } from './context/SystemContext';
import { ToastProvider } from './context/ToastContext';
import ErrorBoundary from './components/common/ErrorBoundary';
import AppRoutes from './routes';

export default function App() {
  return (
    <ErrorBoundary message="The application failed to start. Reload the page to try again.">
      <ToastProvider>
        <SystemProvider>
          <AppRoutes />
        </SystemProvider>
      </ToastProvider>
    </ErrorBoundary>
  );
}
