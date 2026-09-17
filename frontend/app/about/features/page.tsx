import type { Metadata } from 'next';

import { FeaturesSection } from '@/app/about/_sections/FeaturesSection';
import { Reveal } from '@/components/motion/Reveal';
import { routeFor } from '@/lib/routes';

const tab = routeFor('/about/features/');

export const metadata: Metadata = {
  title: 'Features',
  description: tab?.summary,
};

export default function Page() {
  return (
    <Reveal>
      <FeaturesSection />
    </Reveal>
  );
}
