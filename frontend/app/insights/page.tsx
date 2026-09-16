import type { Metadata } from 'next';

import { InsightsBoard } from '@/components/insights/InsightsBoard';
import { PageHeader } from '@/components/ui/PageHeader';
import { routeFor } from '@/lib/routes';

const route = routeFor('/insights/');

export const metadata: Metadata = {
  title: 'Insights',
  description: route?.summary,
};

/**
 * Insights (T133). Aggregate counts of the operator's own History, read from
 * the API at runtime: it is the operator's own record, not a precomputed
 * result, so it cannot come from `generated/`. The page declares no number.
 */
export default function Page() {
  return (
    <div className="space-y-8">
      <PageHeader title="Insights" lede={route?.summary ?? ''} />
      <InsightsBoard />
    </div>
  );
}
