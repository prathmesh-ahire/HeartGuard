import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Feature Extraction', description: 'The locked 138-feature registry and the selected subset.' };

export default function Page() {
  return <PagePlaceholder title="Feature Extraction" href="/features/" phase="Phase 115" />;
}
