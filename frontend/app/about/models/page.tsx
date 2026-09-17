import type { Metadata } from 'next';

import { ModelsSection } from '@/app/about/_sections/ModelsSection';
import { OptimizationSection } from '@/app/about/_sections/OptimizationSection';
import { Reveal } from '@/components/motion/Reveal';
import { routeFor } from '@/lib/routes';

const tab = routeFor('/about/models/');

export const metadata: Metadata = {
  title: 'Models',
  description: tab?.summary,
};

export default function Page() {
  return (
    <div className="space-y-10">
      <Reveal>
        <ModelsSection />
      </Reveal>
      <Reveal>
        <OptimizationSection />
      </Reveal>
    </div>
  );
}
