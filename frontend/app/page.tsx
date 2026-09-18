import type { Metadata } from 'next';

import {
  DatasetCards,
  ExperimentCards,
  LandingFooterSummary,
  LandingHero,
  MetricsStrip,
} from '@/components/landing/LandingSections';
import { HomeBackgroundReveal } from '@/components/motion/HomeBackgroundReveal';
import { Reveal } from '@/components/motion/Reveal';
import { PatientPanel } from '@/components/predict/PatientPanel';
import { PredictionPanel } from '@/components/predict/PredictionPanel';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { prediction } from '@/lib/generated/prediction';
import { routeFor } from '@/lib/routes';

const route = routeFor('/');

/**
 * T141.1: what the deployed models can find, at a glance, above the tool.
 * The class list comes straight from `generated/prediction.json` -- the same
 * `tasks[].classes` `CheckSelector`'s own caption already reads -- deduplicated
 * across the five tasks (binary and murmur both declare "abnormal", for
 * instance). A server component: this never ships as client JS.
 */
function DetectsBar() {
  const classes = Array.from(new Set(prediction.tasks.flatMap((entry) => entry.classes)));
  return (
    <div
      role="list"
      aria-label="What this tool can detect"
      className="mt-4 flex flex-wrap items-center gap-2"
    >
      <span className="label-micro">Detects</span>
      {classes.map((name) => (
        <span
          key={name}
          role="listitem"
          className="rounded-full border border-line bg-panel px-3 py-1 text-label-md capitalize text-ink-2"
        >
          {name}
        </span>
      ))}
    </div>
  );
}

export const metadata: Metadata = {
  title: 'Analyse',
  description: route?.summary,
};

/**
 * Analyse (home, T130; landing sections added above it, Phase 140).
 *
 * Everything through `LandingFooterSummary` is new presentation work above
 * the tool (T140.2-T140.6): a hero statement, the headline metrics, one card
 * per real experiment, one per dataset, and a closing summary. **No route
 * moves and the tool itself does not move** -- `PredictionPanel` and the
 * patient-level panel below it are unchanged from Phase 130/131, in the same
 * order, taking the same props.
 *
 * All five landing sections live in one module, `components/landing/
 * LandingSections.tsx`, not five separate files -- that module's own comment
 * has the reason: five distinct top-level imports from this page, regardless
 * of what any of them rendered, was enough on its own to push
 * `/about/datasets` over its bundle budget (`output: 'export'` embeds `/`'s
 * own page chunk in every route via `TopBar`'s logo link, and crossing a
 * chunk-count threshold on `/` changed how Next redistributed already-shared
 * library code onto other pages). One module, five exports, fixed it.
 *
 * The tool's own former hero (the 3D heart in its own bento panel) is now
 * inside `LandingHero`: one `AnalyseHero`/`Hero3D` mount for the whole page,
 * not two. This section's own intro is a plain title, `id="analyse"` so the
 * new hero's "Analyse a recording" button has somewhere to scroll to.
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
 * server, from `generated/prediction.json`, or (the new sections) from
 * `generated/landing.json`.
 */
export default function Page() {
  return (
    <div className="space-y-16">
      <HomeBackgroundReveal />

      <Reveal>
        <LandingHero />
      </Reveal>

      <Reveal delay={0.05}>
        <MetricsStrip />
      </Reveal>

      <Reveal delay={0.05}>
        <ExperimentCards />
      </Reveal>

      <Reveal delay={0.05}>
        <DatasetCards />
      </Reveal>

      <Reveal delay={0.05}>
        <LandingFooterSummary />
      </Reveal>

      <Reveal>
        <section id="analyse" className="scroll-mt-24">
          <SectionHeader
            eyebrow="Analyse"
            title="Analyse a heart sound"
            description={route?.summary}
          />
          <DetectsBar />
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
