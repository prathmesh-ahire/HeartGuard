import type { Metadata } from 'next';

import { PredictionPanel } from '@/components/predict/PredictionPanel';
import { BatchPanel } from '@/components/predict/BatchPanel';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { prediction } from '@/lib/generated/prediction';
import { routeFor } from '@/lib/routes';

/**
 * Binary screening (T116.1, T116.2, T116.5, T116.6).
 *
 * A server component that renders the copy and hands the interactive work to
 * two client islands. The page itself declares no number: the operating point,
 * the low-confidence margin and the upload bounds all come from
 * `generated/prediction.json`, which `src/reporting/samples.py` produced from
 * the deployed bundle's own manifest and the predictor's own constants.
 */

const route = routeFor('/predict/binary/');

export const metadata: Metadata = {
  title: 'Binary Prediction',
  description: route?.summary,
};

export default function Page() {
  return (
    <div className="space-y-6">
      <PageHeader title="Binary screening" lede={route?.summary ?? ''} />

      <PredictionPanel offered={[{ task: 'binary', label: 'Binary (PhysioNet 2016)' }]} />

      <section>
        <SectionHeader
          eyebrow="T116.5"
          title="Batch screening"
          description="Several recordings in one pass, with the results exported as CSV."
        />
        <BatchPanel task="binary" className="mt-4" />
      </section>

      <section>
        <SectionHeader
          eyebrow="Scope"
          title="What this page does not do"
          description=""
        />
        <ul className="mt-4 max-w-prose list-disc space-y-2 rounded-xl border border-line bg-panel p-4 pl-8 text-body-md text-ink-2 shadow-panel">
          <li>{prediction.disclaimer}</li>
          <li>{prediction.operating_point.note}</li>
          <li>{prediction.reference.note}</li>
          <li>{prediction.upload.duration_note}</li>
        </ul>
      </section>
    </div>
  );
}
