import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Robustness Analytics', description: 'Results under noise, shortened recordings and a different corpus.' };

export default function Page() {
  return <PagePlaceholder title="Robustness Analytics" href="/robustness/" phase="Phase 117" />;
}
