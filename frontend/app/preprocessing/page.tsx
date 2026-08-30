import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Signal Preprocessing', description: 'Resampling, filtering, normalization and signal-quality assessment.' };

export default function Page() {
  return <PagePlaceholder title="Signal Preprocessing" href="/preprocessing/" phase="Phase 114" />;
}
