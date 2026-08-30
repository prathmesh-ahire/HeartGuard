import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Dataset Overview', description: 'The four public PCG corpora as audited against the files on disk.' };

export default function Page() {
  return <PagePlaceholder title="Dataset Overview" href="/dataset/" phase="Phase 114" />;
}
