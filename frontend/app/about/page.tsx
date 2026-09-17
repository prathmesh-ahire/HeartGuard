import type { Metadata } from 'next';

import { PreprocessingSection } from '@/app/about/_sections/PreprocessingSection';
import { Objectives } from '@/components/objectives/Objectives';
import { Reveal } from '@/components/motion/Reveal';
import { PipelineWalkthrough } from '@/components/pipeline/PipelineWalkthrough';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { routeFor } from '@/lib/routes';

const tab = routeFor('/about/');

export const metadata: Metadata = {
  title: 'How it works',
  description: tab?.summary,
};

/** About the Model -- How it works: the objectives, the pipeline, preprocessing. */
export default function Page() {
  return (
    <div className="space-y-10">
      <Reveal>
        <section>
          <SectionHeader
            eyebrow="Scope"
            title="The six research objectives"
            description="Quoted exactly as the source document fixes them."
          />
          <Objectives className="mt-4" />
        </section>
      </Reveal>

      <section>
        <SectionHeader
          eyebrow="Method"
          title="From a recording to a result"
          description="The steps every recording goes through, in order."
        />
        <PipelineWalkthrough className="mt-4" />
      </section>

      <Reveal>
        <PreprocessingSection />
      </Reveal>
    </div>
  );
}
