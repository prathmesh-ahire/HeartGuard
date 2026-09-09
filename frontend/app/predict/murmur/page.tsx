import type { Metadata } from 'next';

import { PredictionPanel } from '@/components/predict/PredictionPanel';
import { PatientPanel } from '@/components/predict/PatientPanel';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { prediction } from '@/lib/generated/prediction';
import { routeFor } from '@/lib/routes';

/**
 * CirCor murmur and outcome (T116.4).
 *
 * Two label spaces again, and a second control the other prediction pages do
 * not have: CirCor labels the subject rather than the recording, and screens at
 * four auscultation locations, so a patient-level indication is a collapse over
 * several recordings. All three declared collapse rules are shown.
 */

const route = routeFor('/predict/murmur/');

export const metadata: Metadata = {
  title: 'Murmur and Outcome Analysis',
  description: route?.summary,
};

const MURMUR = prediction.tasks.find((task) => task.task === 'murmur');

export default function Page() {
  return (
    <div className="space-y-12">
      <SectionHeader
        eyebrow="Prediction"
        title="Murmur and clinical outcome"
        description={route?.summary ?? ''}
      />

      <PredictionPanel
        offered={[
          { task: 'murmur', label: 'Murmur annotation' },
          { task: 'outcome', label: 'Clinical outcome' },
        ]}
      />

      <section>
        <SectionHeader
          eyebrow="T116.4"
          title="Recording level and patient level"
          description="The same subject at all four auscultation locations, collapsed by every declared rule."
        />
        <PatientPanel task="murmur" className="mt-6" />
      </section>

      <section>
        <SectionHeader eyebrow="Scope" title="What these two labels are" description="" />
        <ul className="mt-4 max-w-prose list-disc space-y-2 pl-5 text-sm text-slate-600 dark:text-slate-400">
          {MURMUR !== undefined ? <li>{MURMUR.description}</li> : null}
          <li>
            Murmur and outcome are separate label spaces with separate models. A murmur
            annotation is what an expert heard; an outcome is a clinical label from the
            per-subject text files. Neither is derived from the other here.
          </li>
          <li>{prediction.patient_group.note}</li>
          <li>{prediction.disclaimer}</li>
        </ul>
      </section>
    </div>
  );
}
