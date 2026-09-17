import type { Metadata } from 'next';

import { Reveal } from '@/components/motion/Reveal';
import { AnalyseHero } from '@/components/predict/AnalyseHero';
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
 * Analyse (home, T130).
 *
 * One primary action: add a recording and analyse it. The three checks group
 * the five label spaces in plain words; inside a check each task is still its
 * own model -- PASCAL A and B sit side by side under "Sound type" and the panel
 * merges neither, and CirCor murmur and outcome likewise.
 *
 * Several recordings at once, and comparing two of them, live inside the panel
 * (T131: "Several recordings"), under the same chosen check. The per-patient
 * view stays below: no Part XII task replaces it, and removing it would leave a
 * redirected page with less than it had.
 *
 * The page declares no number. Every value on it arrives formatted from the
 * server or from `generated/prediction.json`.
 */
export default function Page() {
  return (
    <div className="space-y-12">
      <Reveal>
        <AnalyseHero />
      </Reveal>
      <Reveal delay={0.05}>
        <PageHeader title="Analyse a heart sound" lede={route?.summary ?? ''} />
      </Reveal>

      <Reveal delay={0.1}>
        <PredictionPanel
          checks={[
            {
              id: 'normal',
              label: 'Normal or abnormal',
              description: 'Screens the heart sound as normal, or as abnormal and worth a closer look.',
              tasks: [{ task: 'binary', label: 'Normal / abnormal' }],
            },
            {
              id: 'sound',
              label: 'Sound type',
              description: 'Sorts the sound into categories such as murmur or an extra heart sound.',
              tasks: [
                { task: 'pascal_a', label: 'PASCAL A — four classes' },
                { task: 'pascal_b', label: 'PASCAL B — three classes' },
              ],
            },
            {
              id: 'murmur',
              label: 'Murmur and outcome',
              description: 'Looks for a murmur, or screens the overall result, in a child’s recording.',
              tasks: [
                { task: 'murmur', label: 'Murmur annotation' },
                { task: 'outcome', label: 'Clinical outcome' },
              ],
            },
          ]}
        />
      </Reveal>

      <Reveal>
        <section>
          <SectionHeader
            eyebrow="Patient"
            title="Recording level and patient level"
            description="One subject at every auscultation location, combined into a patient-level murmur result by each declared rule."
          />
          <PatientPanel task="murmur" className="mt-4" />
        </section>
      </Reveal>
    </div>
  );
}
