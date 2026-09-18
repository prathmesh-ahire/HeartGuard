import type { Metadata } from 'next';

import { HomeBackgroundReveal } from '@/components/motion/HomeBackgroundReveal';
import { Reveal } from '@/components/motion/Reveal';
import { AnalyseHero } from '@/components/predict/AnalyseHero';
import { PatientPanel } from '@/components/predict/PatientPanel';
import { PredictionPanel } from '@/components/predict/PredictionPanel';
import { EcgLine } from '@/components/ui/EcgLine';
import { ScrollCue } from '@/components/ui/ScrollCue';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { cn } from '@/lib/cn';
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
 *
 * The hero (redesign, 2026-09-17) is one bento panel rather than two stacked
 * blocks: the title/lede and the heart share it, on the same ambient glow, so
 * the heart reads as part of the page's one statement instead of a separate
 * ornament above it. `PageHeader` is not reused here on purpose -- its title
 * is fixed at `text-title` (30px) and this hero wants a larger, page-specific
 * statement; every other page keeps `PageHeader` unchanged.
 *
 * `HomeBackgroundReveal` opens the page on a plain white ground that fades
 * onto the ordinary cream/wine surface over the first stretch of scroll, in
 * step with the heart in `AnalyseHero` growing and fading over the same
 * distance -- this page only; every other page keeps its normal ground from
 * the first paint.
 */
export default function Page() {
  return (
    <div className="space-y-12">
      <HomeBackgroundReveal />
      <Reveal>
        <section className={cn('relative isolate overflow-hidden')}>
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_closest-side_at_78%_28%,rgb(var(--accent)/0.20),transparent_75%)] motion-safe:animate-pulse-subtle"
          />
          <EcgLine className="pointer-events-none absolute inset-x-0 bottom-6 -z-10 h-10 w-full text-accent-line/25" />
          <div className="relative grid gap-8 p-6 sm:p-10 lg:grid-cols-[1.15fr_0.85fr] lg:items-center lg:gap-12 lg:p-14">
            <div className="max-w-reading">
              <p className="text-label-md uppercase text-accent">Analyse</p>
              <h1 className="mt-2 text-[2.25rem] font-semibold leading-[1.08] tracking-tight text-ink sm:text-[2.75rem]">
                Analyse <span className="font-normal text-ink-2">a heart sound</span>
              </h1>
              {route?.summary ? (
                <p className="mt-4 text-lede text-ink-2">{route.summary}</p>
              ) : null}
            </div>
            <AnalyseHero />
          </div>
          <div className="relative flex justify-center pb-4">
            <ScrollCue />
          </div>
        </section>
      </Reveal>

      <Reveal delay={0.1}>
        <PredictionPanel
          className="mx-auto w-full max-w-3xl"
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
