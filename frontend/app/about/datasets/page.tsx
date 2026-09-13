import type { Metadata } from 'next';

import { DatasetSection } from '@/app/about/_sections/DatasetSection';
import { routeFor } from '@/lib/routes';

const tab = routeFor('/about/datasets/');

export const metadata: Metadata = {
  title: 'Datasets',
  description: tab?.summary,
};

export default function Page() {
  return <DatasetSection />;
}
