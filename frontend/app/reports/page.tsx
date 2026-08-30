import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Reports', description: 'Generated reports and the evidence browser.' };

export default function Page() {
  return <PagePlaceholder title="Reports" href="/reports/" phase="Phase 117" />;
}
