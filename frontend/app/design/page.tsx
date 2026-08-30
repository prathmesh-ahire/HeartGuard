'use client';

import { useState } from 'react';

import { cn } from '@/lib/cn';
import { tables, theme } from '@/lib/generated';
import { PALETTE_CONTRAST, SERIES_COLORS, SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { EnsembleVote } from '@/components/ensemble/EnsembleVote';
import { Reveal } from '@/components/motion/Reveal';
import { PipelineWalkthrough } from '@/components/pipeline/PipelineWalkthrough';
import { Hero3D } from '@/components/three/Hero3D';
import { Badge } from '@/components/ui/Badge';
import { FileUpload } from '@/components/ui/FileUpload';
import { GlassCard } from '@/components/ui/GlassCard';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { StatTile } from '@/components/ui/StatTile';
import { Tabs } from '@/components/ui/Tabs';
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States';
import { Tooltip, TooltipProvider } from '@/components/ui/Tooltip';

/**
 * The design reference page (T111.6): every component, in every state, on one
 * scrollable page, so a visual regression is visible in a single pass instead
 * of being hunted across twelve document pages.
 *
 * **Every value shown here comes from `generated/`.** It would have been easier
 * to type a plausible number into a StatTile example, and that is precisely the
 * habit the codegen boundary exists to prevent: a placeholder literal is how a
 * fabricated result got into a page in the parallel implementation this project
 * was warned by. Reading the real payloads also makes this page a live check
 * that those payloads are shaped usefully.
 */

function columnOf(id: string, name: string) {
  return tables[id]?.columns.find((column) => column.name === name);
}

export default function DesignPage() {
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'done' | 'error'>('idle');
  const [fileName, setFileName] = useState<string | null>(null);

  const inventory = tables['T01'];
  const distribution = tables['T02'];
  const files = columnOf('T01', 'total_files');
  const usable = columnOf('T01', 'usable_files');
  const share = columnOf('T02', 'share');

  return (
    <TooltipProvider>
      <div className="space-y-12">
        <SectionHeader
          level={1}
          eyebrow="Design system reference"
          title="Components, in every state"
          description={
            <>
              Every component the dashboard is built from, rendered in each of its states
              for visual QA. Nothing on this page is a result. Every value shown is read
              from the generated payloads rather than typed in, so this page is also a
              live check that those payloads are usable.
            </>
          }
        />

        <section className="space-y-4">
          <SectionHeader
            level={2}
            title="Palette"
            description={
              <>
                {theme.palette.name}. The order is meaningful: index 0 is the first series
                in every figure, so a chart in the browser and the 300 dpi PNG beside it
                colour the same series identically. Exported from{' '}
                <span className="font-mono">src/reporting/plot_style.py</span>, never
                retyped.
              </>
            }
          />
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
            {SERIES_COLORS.map((color, index) => (
              <div key={color} className={cn(SURFACE.card, 'overflow-hidden')}>
                <div className="h-16 w-full" style={{ backgroundColor: color }} />
                <div className="p-2">
                  <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>series {index}</p>
                  <p className={cn(TYPE_SCALE.caption, 'font-mono')}>{color}</p>
                </div>
              </div>
            ))}
          </div>

          <div className={cn(SURFACE.sunken, 'overflow-x-auto p-4')}>
            <p className={cn(TYPE_SCALE.body, 'font-medium')}>
              Measured contrast, {PALETTE_CONTRAST.standard}
            </p>
            <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1 max-w-prose')}>
              Okabe-Ito guarantees the eight hues stay distinguishable{' '}
              <em>from each other</em> under the common colour-vision deficiencies. That
              is a different property from luminance contrast against a page, and four of
              the eight fall below the threshold on one ground or the other — the yellow
              is nearly as bright as white. Dropping a colour would break the fixed series
              order every figure depends on, so the chart layer strokes the mark instead.
            </p>
            <table className={cn(TYPE_SCALE.caption, 'mt-3 w-full min-w-[28rem]')}>
              <thead className={SURFACE.subtle}>
                <tr className="text-left">
                  <th className="py-1 pr-4 font-medium">Series</th>
                  <th className="py-1 pr-4 font-medium">Colour</th>
                  <th className="py-1 pr-4 font-medium">On light</th>
                  <th className="py-1 pr-4 font-medium">On dark</th>
                  <th className="py-1 font-medium">Needs a stroke on</th>
                </tr>
              </thead>
              <tbody>
                {PALETTE_CONTRAST.series.map((entry) => (
                  <tr key={entry.colour} className="border-t border-slate-200 dark:border-slate-800">
                    <td className="py-1 pr-4 tabular-nums">{entry.index}</td>
                    <td className="py-1 pr-4 font-mono">{entry.colour}</td>
                    <td className="py-1 pr-4 tabular-nums">{entry.on_light}</td>
                    <td className="py-1 pr-4 tabular-nums">{entry.on_dark}</td>
                    <td className="py-1">
                      {entry.needs_outline_on.length === 0 ? (
                        <Badge tone="good">neither</Badge>
                      ) : (
                        <span className="flex flex-wrap gap-1">
                          {entry.needs_outline_on.map((ground) => (
                            <Badge key={ground} tone="warn">
                              {ground}
                            </Badge>
                          ))}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="space-y-4">
          <SectionHeader level={2} title="Type scale" />
          <GlassCard className="space-y-3">
            <p className={TYPE_SCALE.display}>Display</p>
            <p className={TYPE_SCALE.h1}>Heading 1</p>
            <p className={TYPE_SCALE.h2}>Heading 2</p>
            <p className={TYPE_SCALE.h3}>Heading 3</p>
            <p className={TYPE_SCALE.lead}>Lead paragraph</p>
            <p className={TYPE_SCALE.body}>Body text</p>
            <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>Caption</p>
            <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>Micro label</p>
          </GlassCard>
        </section>

        <section className="space-y-4">
          <SectionHeader
            level={2}
            title="Surfaces"
            description="Solid is the default. Glass is only for the hero, where nothing is being read through it."
          />
          <div className="grid gap-4 sm:grid-cols-3">
            <GlassCard variant="solid">
              <p className={TYPE_SCALE.h3}>Solid</p>
              <p className={cn(TYPE_SCALE.body, SURFACE.muted, 'mt-1')}>
                Everything that carries data.
              </p>
            </GlassCard>
            <GlassCard variant="glass">
              <p className={TYPE_SCALE.h3}>Glass</p>
              <p className={cn(TYPE_SCALE.body, SURFACE.muted, 'mt-1')}>
                Over the 3D hero only.
              </p>
            </GlassCard>
            <GlassCard variant="sunken">
              <p className={TYPE_SCALE.h3}>Sunken</p>
              <p className={cn(TYPE_SCALE.body, SURFACE.muted, 'mt-1')}>
                Recessed panels and skeletons.
              </p>
            </GlassCard>
          </div>
        </section>

        <section className="space-y-4">
          <SectionHeader
            level={2}
            title="Stat tiles"
            description="Counts may animate. Metrics never do: counting up to a metric would mean formatting it in the browser on every frame."
          />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {inventory && files && usable ? (
              <>
                <StatTile
                  animate
                  label="Recordings on disk"
                  display={files.display[0] ?? ''}
                  value={files.values?.[0] ?? null}
                  unit="files"
                  hint="Animated, because it is a count."
                  source={inventory.source_csv}
                />
                <StatTile
                  label="Labeled and modelled"
                  display={usable.display[0] ?? ''}
                  unit="files"
                  hint="Static: no animation requested."
                  source={inventory.source_csv}
                />
              </>
            ) : null}
            {distribution && share ? (
              <StatTile
                label="Class share"
                display={share.display[0] ?? ''}
                hint="A metric. Rendered exactly as Python formatted it."
                source={distribution.source_csv}
              />
            ) : null}
            <StatTile
              label="Unavailable measure"
              display="n/a"
              hint="A missing value reads as n/a, never as zero."
            />
          </div>
        </section>

        <section className="space-y-4">
          <SectionHeader level={2} title="Badges" />
          <div className="flex flex-wrap gap-2">
            <Badge tone="neutral">neutral</Badge>
            <Badge tone="info">info</Badge>
            <Badge tone="good">good</Badge>
            <Badge tone="warn">warn</Badge>
            <Badge tone="danger">danger</Badge>
          </div>
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
            Status colours are deliberately not the series palette: a badge must not
            borrow the colour a chart is using for a series in the same viewport.
          </p>
        </section>

        <section className="space-y-4">
          <SectionHeader level={2} title="Tabs and tooltips" />
          <GlassCard>
            <Tabs
              ariaLabel="Design reference examples"
              items={[
                {
                  value: 'one',
                  label: 'First',
                  content: (
                    <p className={TYPE_SCALE.body}>
                      Radix underneath, so roving focus and the ARIA wiring are correct
                      rather than approximated.
                    </p>
                  ),
                },
                {
                  value: 'two',
                  label: 'Second',
                  content: (
                    <p className={TYPE_SCALE.body}>
                      Arrow keys move between tabs; Home and End jump to the ends.
                    </p>
                  ),
                },
                {
                  value: 'three',
                  label: 'Third',
                  content: (
                    <Tooltip content="Tooltip content is text, or a pre-formatted display string. Never a number computed here.">
                      <button
                        type="button"
                        className="rounded border border-slate-300 px-3 py-1.5 text-sm dark:border-slate-700"
                      >
                        Hover or focus me
                      </button>
                    </Tooltip>
                  ),
                },
              ]}
            />
          </GlassCard>
        </section>

        <section className="space-y-4">
          <SectionHeader
            level={2}
            title="Loading, empty and error"
            description="A failed request must render a visible error, never an empty chart. An empty chart is indistinguishable from a real result of zero."
          />
          <div className="grid gap-4 lg:grid-cols-3">
            <LoadingState label="Running inference" />
            <EmptyState
              title="No recordings selected"
              description="Choose a dataset and a class to see matching records."
            />
            <ErrorState
              title="Inference service unreachable"
              detail={
                <>
                  <span className="font-mono">POST /predict</span> did not respond. The
                  local inference service may not be running.
                </>
              }
              onRetry={() => undefined}
            />
          </div>
        </section>

        <section className="space-y-4">
          <SectionHeader
            level={2}
            title="Recording upload"
            description="Validation happens before the network. A rejected file says what was wrong with it."
          />
          <div className="grid gap-4 lg:grid-cols-2">
            <FileUpload
              phase={phase === 'error' ? 'idle' : phase}
              fileName={fileName}
              progress={phase === 'uploading' ? 62 : null}
              error={
                phase === 'error'
                  ? 'That file is not a WAV recording. This prototype reads uncompressed .wav only, because a lossy format changes the spectral content the features are computed from.'
                  : null
              }
              onFile={(file) => {
                setFileName(file.name);
                setPhase('done');
              }}
            />
            <GlassCard variant="sunken" className="space-y-2">
              <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>Force a state</p>
              <div className="flex flex-wrap gap-2">
                {(['idle', 'uploading', 'done', 'error'] as const).map((state) => (
                  <button
                    key={state}
                    type="button"
                    onClick={() => setPhase(state)}
                    className={cn(
                      'rounded border px-3 py-1.5 text-sm',
                      phase === state
                        ? 'border-sky-600 font-medium text-sky-700 dark:border-sky-400 dark:text-sky-300'
                        : 'border-slate-300 dark:border-slate-700',
                    )}
                  >
                    {state}
                  </button>
                ))}
              </div>
              <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
                Drop a non-WAV file on the control to see the real validation message
                rather than this forced one.
              </p>
            </GlassCard>
          </div>
        </section>

        {/* ------------------------------------------------------------------
            Phase 112 -- the 3D and motion layer, on the same QA surface.

            T112.7 asks for three things a person has to look at: that the
            scene loads without an SSR error, that reduced motion is honoured,
            and that the fallback is a designed panel rather than a broken
            box. All three are visible here.
        ------------------------------------------------------------------- */}
        <section className="space-y-4">
          <SectionHeader
            level={2}
            title="3D layer"
            description={
              'Lazy-loaded behind a dynamic import with ssr: false, so three.js is in ' +
              'its own chunk and never runs during the static export. Under ' +
              'prefers-reduced-motion the model is posed rather than beating; with no ' +
              'WebGL context it is replaced by a designed panel, because an absent GPU ' +
              'is an ordinary outcome and not an error. The beat is decoration with no ' +
              'data behind it.'
            }
          />
          <Hero3D className="rounded-lg border border-slate-200 dark:border-slate-800" />
        </section>

        <section className="space-y-4">
          <SectionHeader
            level={2}
            title="Ensemble vote"
            description={
              'SVM, Random Forest and Gradient Boosting weights from SO-05. The bars ' +
              'sit almost on the equal-weight line because that is what the search ' +
              'found; the caption says so rather than letting the picture imply ' +
              'otherwise.'
            }
          />
          <EnsembleVote />
        </section>

        <section className="space-y-4">
          <SectionHeader
            level={2}
            title="Pipeline walkthrough"
            description={
              'The twelve architecture steps, read from generated/pipeline.json. Every ' +
              'step names a module and an outputs/ directory that the exporter ' +
              'confirmed exists. Scroll-linked highlighting is off entirely under ' +
              'reduced motion, where this renders as a plain ordered list.'
            }
          />
          <PipelineWalkthrough />
        </section>

        <section className="space-y-4">
          <SectionHeader
            level={2}
            title="Reveal"
            description={
              'Framer Motion card reveal. Content is never gated on the animation: if ' +
              'the viewport callback never fires, the card is still there and still ' +
              'readable.'
            }
          />
          <Reveal>
            <GlassCard className="p-5">
              <p className={cn(TYPE_SCALE.body)}>
                This card faded in on scroll, or did not, depending on the motion
                setting. Either way it says the same thing.
              </p>
            </GlassCard>
          </Reveal>
        </section>
      </div>
    </TooltipProvider>
  );
}
