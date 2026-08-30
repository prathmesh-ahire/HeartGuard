import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Murmur and Outcome Analysis', description: 'CirCor murmur and clinical-outcome output.' };

export default function Page() {
  return <PagePlaceholder title="Murmur and Outcome Analysis" href="/predict/murmur/" phase="Phase 116" />;
}
