import type { Metadata } from 'next';

import { PredictionPanel } from '@/components/predict/PredictionPanel';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { prediction } from '@/lib/generated/prediction';
import { routeFor } from '@/lib/routes';

/**
 * PASCAL A and PASCAL B (T116.3).
 *
 * Two label spaces on one page and never merged into one: the selector switches
 * between two separate models with separate class vocabularies. PASCAL A's
 * `artifact` is a recording-quality label rather than a cardiac class, and the
 * caveat is rendered from the task's own declared description rather than
 * retyped here, so it cannot drift from the predictor's.
 */

const route = routeFor('/predict/multiclass/');

export const metadata: Metadata = {
  title: 'Multiclass Prediction',
  description: route?.summary,
};

const PASCAL_A = prediction.tasks.find((task) => task.task === 'pascal_a');

export default function Page() {
  return (
    <div className="space-y-12">
      <SectionHeader
        level={1}
        eyebrow="Prediction"
        title="Multiclass acoustic events"
        description={route?.summary ?? ''}
      />

      <PredictionPanel
        offered={[
          { task: 'pascal_a', label: 'PASCAL A — four classes' },
          { task: 'pascal_b', label: 'PASCAL B — three classes' },
        ]}
      />

      <section>
        <SectionHeader
          eyebrow="Scope"
          title="Two datasets, two label spaces, one page"
          description=""
        />
        <ul className="mt-4 max-w-prose list-disc space-y-2 pl-5 text-sm text-slate-600 dark:text-slate-400">
          <li>
            PASCAL A and PASCAL B are separate tasks with separate models. This page offers
            both and merges neither: a recording is scored against one vocabulary or the
            other, never against their union.
          </li>
          {PASCAL_A !== undefined ? <li>{PASCAL_A.description}</li> : null}
          <li>{prediction.disclaimer}</li>
          <li>{prediction.low_confidence.note}</li>
        </ul>
      </section>
    </div>
  );
}
