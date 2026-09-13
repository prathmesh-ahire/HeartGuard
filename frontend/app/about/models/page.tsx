import type { Metadata } from 'next';

import { ModelsSection } from '@/app/about/_sections/ModelsSection';
import { OptimizationSection } from '@/app/about/_sections/OptimizationSection';
import { routeFor } from '@/lib/routes';

const tab = routeFor('/about/models/');

export const metadata: Metadata = {
  title: 'Models',
  description: tab?.summary,
};

export default function Page() {
  return (
    <div className="space-y-10">
      <ModelsSection />
      <OptimizationSection />
    </div>
  );
}
