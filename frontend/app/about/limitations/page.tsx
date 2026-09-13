import type { Metadata } from 'next';

import { LimitationsSection } from '@/app/about/_sections/LimitationsSection';
import { routeFor } from '@/lib/routes';

const tab = routeFor('/about/limitations/');

export const metadata: Metadata = {
  title: 'Limitations',
  description: tab?.summary,
};

export default function Page() {
  return <LimitationsSection />;
}
