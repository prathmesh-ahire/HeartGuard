import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Search Optimization', description: 'Search space, methods compared and parameters selected.' };

export default function Page() {
  return <PagePlaceholder title="Search Optimization" href="/optimization/" phase="Phase 115" />;
}
