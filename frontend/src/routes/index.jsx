import { Suspense, lazy } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import AppLayout from '../components/layout/AppLayout';
import { Spinner } from '../components/common';
import LandingPage from '../pages/LandingPage';

/**
 * Route table.
 *
 * The landing page loads eagerly (it is the first paint); everything behind the
 * app shell is code-split so the heavy chart and graph libraries are only
 * fetched when their page is opened.
 */
const Dashboard = lazy(() => import('../pages/Dashboard'));
const ResearchAssistant = lazy(() => import('../pages/ResearchAssistant'));
const KnowledgeBase = lazy(() => import('../pages/KnowledgeBase'));
const DocumentDetails = lazy(() => import('../pages/DocumentDetails'));
const KnowledgeGraph = lazy(() => import('../pages/KnowledgeGraph'));
const Evaluation = lazy(() => import('../pages/Evaluation'));
const Settings = lazy(() => import('../pages/Settings'));
const NotFound = lazy(() => import('../pages/NotFound'));

function PageFallback() {
  return (
    <div className="flex min-h-[50vh] items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <Spinner className="h-6 w-6" />
        <p className="text-sm text-ink-500">Loading…</p>
      </div>
    </div>
  );
}

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />

      <Route element={<AppLayout />}>
        <Route
          path="/dashboard"
          element={
            <Suspense fallback={<PageFallback />}>
              <Dashboard />
            </Suspense>
          }
        />
        <Route
          path="/research"
          element={
            <Suspense fallback={<PageFallback />}>
              <ResearchAssistant />
            </Suspense>
          }
        />
        <Route
          path="/knowledge-base"
          element={
            <Suspense fallback={<PageFallback />}>
              <KnowledgeBase />
            </Suspense>
          }
        />
        <Route
          path="/knowledge-base/:documentId"
          element={
            <Suspense fallback={<PageFallback />}>
              <DocumentDetails />
            </Suspense>
          }
        />
        <Route
          path="/graph"
          element={
            <Suspense fallback={<PageFallback />}>
              <KnowledgeGraph />
            </Suspense>
          }
        />
        <Route
          path="/evaluation"
          element={
            <Suspense fallback={<PageFallback />}>
              <Evaluation />
            </Suspense>
          }
        />
        <Route
          path="/settings"
          element={
            <Suspense fallback={<PageFallback />}>
              <Settings />
            </Suspense>
          }
        />
        <Route
          path="*"
          element={
            <Suspense fallback={<PageFallback />}>
              <NotFound />
            </Suspense>
          }
        />
      </Route>

      <Route path="/index.html" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
