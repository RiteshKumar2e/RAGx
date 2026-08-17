import { Link } from 'react-router-dom';
import { Compass } from 'lucide-react';
import { EmptyState } from '../components/common';

export default function NotFound() {
  return (
    <EmptyState
      icon={Compass}
      title="Page not found"
      description="That route does not exist in RAGX. Use the navigation to reach the Dashboard, Research Assistant, Knowledge Base, Graph or Evaluation."
      action={
        <Link
          to="/dashboard"
          className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-3.5 py-2 text-sm font-medium text-white transition hover:bg-brand-700"
        >
          Back to the dashboard
        </Link>
      }
      className="min-h-[60vh]"
    />
  );
}
