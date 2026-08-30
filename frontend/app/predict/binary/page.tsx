import type { Metadata } from 'next';

import { PagePlaceholder } from '@/components/PagePlaceholder';

export const metadata: Metadata = { title: 'Binary Prediction', description: 'Normal / abnormal screening indication with its confidence.' };

export default function Page() {
  return <PagePlaceholder title="Binary Prediction" href="/predict/binary/" phase="Phase 116" />;
}
