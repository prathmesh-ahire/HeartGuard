import type { Metadata } from 'next';

import { ExplainabilitySection } from '@/app/about/_sections/ExplainabilitySection';
import { RobustnessSection } from '@/app/about/_sections/RobustnessSection';
import { Reveal } from '@/components/motion/Reveal';
import { routeFor } from '@/lib/routes';

const tab = routeFor('/about/performance/');

export const metadata: Metadata = {
  title: 'Performance',
  description: tab?.summary,
};

export default function Page() {
  return (
    <div className="space-y-10">
      {/* Not wrapped in Reveal: T135.5's deep-link test navigates straight to
          `#location` and expects the native <details> fragment behaviour to
          have already put it in the viewport before any IntersectionObserver
          would fire -- an extra opacity-0 wrapper is a needless risk to an
          already-pinned test for no visible gain this far down the page. */}
      <RobustnessSection />
      <Reveal>
        <ExplainabilitySection />
      </Reveal>
    </div>
  );
}
