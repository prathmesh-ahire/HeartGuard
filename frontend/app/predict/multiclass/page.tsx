import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Multiclass Prediction', description: 'PASCAL A and PASCAL B acoustic-event output.' };

export default function Page() {
  return <PagePlaceholder title="Multiclass Prediction" href="/predict/multiclass/" phase="Phase 116" />;
}
