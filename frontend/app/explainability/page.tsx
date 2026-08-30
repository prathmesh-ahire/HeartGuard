import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Explainability', description: 'Global feature importance and per-sample explanation.' };

export default function Page() {
  return <PagePlaceholder title="Explainability" href="/explainability/" phase="Phase 117" />;
}
