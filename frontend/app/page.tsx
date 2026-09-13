import type { Metadata } from 'next';

import { BatchPanel } from '@/components/predict/BatchPanel';
import { PatientPanel } from '@/components/predict/PatientPanel';
import { PredictionPanel } from '@/components/predict/PredictionPanel';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { routeFor } from '@/lib/routes';

const route = routeFor('/');

export const metadata: Metadata = {
  title: 'Analyse',
  description: route?.summary,
};

/**
 * Analyse (home, T127.3).
 *
 * The three old prediction pages, joined: one panel offering every label space
 * -- still five separate tasks with five separate models; PASCAL A and B are
 * offered side by side and the panel merges neither -- plus
 * batch scoring and the per-patient view. Phase 130 rebuilds this page; until
 * then it keeps every capability the old routes had, so nothing redirected here
 * lands on less than it left.
 *
 * The page declares no number. Every value on it arrives formatted from the
 * server or from `generated/prediction.json`.
 */
export default function Page() {
  return (
    <div className="space-y-8">
      <PageHeader title="Analyse a recording" lede={route?.summary ?? ''} />

      <PredictionPanel
        offered={[
          { task: 'binary', label: 'Normal / abnormal' },
          { task: 'pascal_a', label: 'PASCAL A — four classes' },
          { task: 'pascal_b', label: 'PASCAL B — three classes' },
          { task: 'murmur', label: 'Murmur annotation' },
          { task: 'outcome', label: 'Clinical outcome' },
        ]}
      />

      <section>
        <SectionHeader
          eyebrow="Batch"
          title="Several recordings at once"
          description="Normal / abnormal screening for a set of files, with the results downloadable as CSV."
        />
        <BatchPanel task="binary" className="mt-4" />
      </section>

      <section>
        <SectionHeader
          eyebrow="Patient"
          title="Recording level and patient level"
          description="One subject at every auscultation location, combined into a patient-level murmur result by each declared rule."
        />
        <PatientPanel task="murmur" className="mt-4" />
      </section>
    </div>
  );
}
