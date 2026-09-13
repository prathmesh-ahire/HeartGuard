import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';
import { routeFor } from '@/lib/routes';

export const metadata: Metadata = {
  title: 'History',
  description: routeFor('/history/')?.summary,
};

/** History (T127.3). Built in Phases 129 and 132. */
export default function Page() {
  return <PagePlaceholder title="History" href="/history/" />;
}
