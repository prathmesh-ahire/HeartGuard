import type { Metadata } from 'next';

import { ExplainabilitySection } from '@/app/about/_sections/ExplainabilitySection';
import { RobustnessSection } from '@/app/about/_sections/RobustnessSection';
import { routeFor } from '@/lib/routes';

const tab = routeFor('/about/performance/');

export const metadata: Metadata = {
  title: 'Performance',
  description: tab?.summary,
};

export default function Page() {
  return (
    <div className="space-y-10">
      <RobustnessSection />
      <ExplainabilitySection />
    </div>
  );
}
