import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';
import { routeFor } from '@/lib/routes';

export const metadata: Metadata = {
  title: 'Insights',
  description: routeFor('/insights/')?.summary,
};

/** Insights (T127.3). Built in Phase 133. */
export default function Page() {
  return <PagePlaceholder title="Insights" href="/insights/" />;
}
