import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Model Comparison', description: 'The declared models and their fold-wise comparison.' };

export default function Page() {
  return <PagePlaceholder title="Model Comparison" href="/models/" phase="Phase 115" />;
}
