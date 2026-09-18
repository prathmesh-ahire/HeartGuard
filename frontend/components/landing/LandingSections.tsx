import Link from 'next/link';

import { Stagger, StaggerItem } from '@/components/motion/Reveal';
import { AnalyseHero } from '@/components/predict/AnalyseHero';
import { AnimatedCounter } from '@/components/ui/AnimatedCounter';
import { EcgLine } from '@/components/ui/EcgLine';
import { GlassCard } from '@/components/ui/GlassCard';
import type { IconName } from '@/components/ui/Icon';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { StatTile } from '@/components/ui/StatTile';
import { landing } from '@/lib/generated/landing';
import { segmentation } from '@/lib/generated';
import { cn } from '@/lib/cn';
import { ROUTES } from '@/lib/routes';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * The five landing sections (Phase 140), in one module rather than five.
 *
 * They started as five separate files, one per component. That structure
 * alone -- five distinct top-level modules imported from `app/page.tsx` --
 * was enough to push `/about/datasets` from 234.8 kB to 265.9 kB against its
 * 260 kB budget, independent of what any of the five actually rendered (an
 * emptied-out placeholder component reproduced the exact same failure).
 * `output: 'export'`'s static HTML embeds a real, blocking `<script src>`
 * for `/`'s own page chunk in every other route (via `TopBar`'s `<Link
 * href="/">`), and crossing some chunk-count threshold on `/` changed how
 * Next distributed already-shared code (framer-motion, `Button`, the
 * waveform player) across pages -- consolidating five small page-specific
 * chunks into fewer, larger ones that other routes then also paid for.
 * Reducing dependencies (dropped `ButtonLink`'s spring animation, dropped
 * per-card `Stagger`) and raising webpack's `maxInitialRequests` both had no
 * effect; only reducing the module count did. One file, five exports.
 */

/** One icon per metric key (T140.3). Purely presentational -- the key itself is data. */
const ICONS: Record<string, IconName> = {
  sensitivity: 'pulse',
  specificity: 'check',
  balanced_accuracy: 'optimization',
  f1: 'binary',
  roc_auc: 'explainability',
  n_features: 'features',
};

/** Four corners around the hero's heart stage; `chips` never has more than four entries. */
const CHIP_POSITION = ['left-0 top-2', 'right-0 top-10', 'left-2 bottom-6', 'right-2 bottom-0'] as const;

const TECHNOLOGY = [
  'Python 3.11',
  'scikit-learn',
  'FastAPI',
  'Next.js 14',
  'React',
  'TypeScript',
  'Tailwind CSS',
  'Three.js',
] as const;

/**
 * The landing hero (T140.2). Reuses `AnalyseHero`'s 3D heart rather than
 * mounting a second Three.js scene -- one canvas, still the tool's own, now
 * framed by the new page-level statement around it.
 *
 * The floating chips are `segmentation.json`'s real legend entries (S1,
 * systole, S2, diastole) for the one pinned CirCor recording -- the same
 * source `SignalExplorer` uses for the segmentation overlay elsewhere. Their
 * `seconds_display` strings are Python-formatted; nothing here rounds them.
 */
export function LandingHero() {
  const chips = segmentation.legend.filter(
    (item) => item.key !== 'unannotated' && item.n_segments > 0,
  );

  return (
    <section className="relative isolate overflow-hidden rounded-3xl border border-line bg-panel/50 p-6 sm:p-10 lg:p-14">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_closest-side_at_20%_15%,rgb(var(--accent)/0.16),transparent_70%)]"
      />
      <EcgLine className="pointer-events-none absolute inset-x-0 top-4 -z-10 h-8 w-full text-accent-line/20" />

      <div className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
        <div>
          <p className="text-label-md uppercase text-accent">PV-MEPCG / PulseVision</p>
          <h1 className={cn(TYPE_SCALE.hero, 'mt-3 text-ink')}>
            Heart-sound screening, <span className="text-gradient-accent">verified</span> end to
            end.
          </h1>
          <p className="mt-4 max-w-reading text-lede text-ink-2">{landing.footer.results_summary}</p>

          <div className="mt-8 grid grid-cols-2 gap-6 sm:grid-cols-4">
            {landing.hero.headline.map((stat) => (
              <div key={stat.key}>
                <AnimatedCounter value={stat.value} display={stat.display} />
                <p className="mt-1 text-label-sm uppercase text-ink-3">{stat.label}</p>
              </div>
            ))}
          </div>

          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="#analyse"
              className="inline-flex items-center justify-center rounded-lg border border-accent-strong bg-accent px-4 py-2 text-label-lg uppercase text-on-accent shadow-accent transition-colors hover:bg-accent-strong"
            >
              Analyse a recording
            </Link>
            <Link
              href="/about/"
              className="inline-flex items-center justify-center rounded-lg border border-line bg-panel px-4 py-2 text-label-lg uppercase text-ink-2 transition-colors hover:border-accent-line hover:text-accent-strong"
            >
              See how it works
            </Link>
          </div>
        </div>

        <div className="relative">
          <AnalyseHero />
          <div aria-hidden="true" className="pointer-events-none absolute inset-0 hidden sm:block">
            {chips.map((chip, index) => (
              <span
                key={chip.key}
                className={cn(
                  'absolute rounded-full border border-line bg-panel/80 px-2.5 py-1 text-label-sm text-ink-2 shadow-panel backdrop-blur-md motion-safe:animate-float-subtle',
                  CHIP_POSITION[index % CHIP_POSITION.length],
                )}
                style={{ animationDelay: index * 0.35 + 's' }}
              >
                {chip.name} <span className="text-ink-3">· {chip.seconds_display}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/**
 * Six headline tiles off the deployed binary model (EXP-A2), plus the
 * trust-line chips beneath it (T140.3). Every value is
 * `landing.metrics_strip`/`landing.trust_line`, exported by
 * `landing_payload()` -- a selection from `experiments.json` and the T16/T20/
 * T23/T25/T28 tables, never computed here.
 */
export function MetricsStrip() {
  return (
    <section className="space-y-6">
      <SectionHeader
        eyebrow={'Deployed model ' + (landing.champion_model_id ?? '')}
        title="Headline performance"
        description="Sensitivity and balanced accuracy lead, never accuracy alone (research rule 6)."
      />

      <Stagger className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {landing.metrics_strip.map((metric) => (
          <StaggerItem key={metric.key}>
            <StatTile
              label={metric.label}
              display={metric.display}
              value={metric.value}
              icon={ICONS[metric.key] ?? 'pulse'}
              animate
            />
          </StaggerItem>
        ))}
      </Stagger>

      <div className="flex flex-wrap gap-2">
        {landing.trust_line.map((item) => (
          <span
            key={item.label}
            title={item.detail}
            className={cn(
              'inline-flex max-w-full items-center gap-1.5 rounded-full border border-line bg-panel px-3 py-1.5',
              TYPE_SCALE.caption,
              SURFACE.muted,
            )}
          >
            <span className="shrink-0 font-medium text-ink-2">{item.label}:</span>
            <span className="truncate">{item.detail}</span>
          </span>
        ))}
      </div>

      {landing.evidence_highlights.length > 0 ? (
        <div className="grid gap-2 sm:grid-cols-3">
          {landing.evidence_highlights.map((item) => (
            <div key={item.label} className="rounded-xl border border-line bg-sunken px-3 py-2">
              <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>{item.label}</p>
              <p className="mt-0.5 text-body-md text-ink">{item.display}</p>
              <p className={cn(TYPE_SCALE.caption, SURFACE.subtle, 'mt-0.5 font-mono')}>
                {item.source}
              </p>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

/**
 * One card per declared experiment (T140.4), linking into About > Models
 * where the full metric table, per-fold values and confusion matrix live.
 * An experiment that has not run shows its stated reason rather than being
 * silently dropped -- `experiments_payload()`'s own rule, carried through.
 */
export function ExperimentCards() {
  return (
    <section className="space-y-6">
      <SectionHeader
        eyebrow="Real runs, not a demo"
        title="Every declared experiment"
        description="Five declared runs across two label families. Binary, PASCAL A and PASCAL B are separate tasks and are never merged."
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {landing.experiments.map((experiment) => (
          <Link key={experiment.exp_id} href={experiment.link} className="block h-full">
            <GlassCard as="article" className="h-full" eyebrow={experiment.exp_id} title={experiment.title}>
              {experiment.available ? (
                <>
                  <dl className="grid grid-cols-2 gap-x-3 gap-y-2">
                    {experiment.metrics.map((metric) => (
                      <div key={metric.key}>
                        <dt className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>{metric.label}</dt>
                        <dd className={cn(TYPE_SCALE.stat, 'text-ink')}>{metric.display}</dd>
                      </div>
                    ))}
                  </dl>
                  {experiment.finding ? (
                    <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-3')}>{experiment.finding}</p>
                  ) : null}
                </>
              ) : (
                <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>{experiment.reason}</p>
              )}
            </GlassCard>
          </Link>
        ))}
      </div>
    </section>
  );
}

/**
 * One card per public corpus (T140.5): files, subjects, hours and native
 * sample rate come from `dataset_summary.json`/the audit CSV via
 * `landing_payload()`; class chips are the real label values found in that
 * corpus's own label column, not a fixed list.
 */
export function DatasetCards() {
  return (
    <section className="space-y-6">
      <SectionHeader
        eyebrow="Four public corpora"
        title="What the models learned from"
        description="Every file audited and counted from disk, not taken from a paper's documentation."
      />

      <Stagger className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {landing.datasets.map((dataset) => (
          <StaggerItem key={dataset.dataset_source}>
            <Link href="/about/datasets/" className="block h-full">
              <GlassCard
                as="article"
                className="h-full"
                eyebrow={dataset.dataset_source}
                title={dataset.dataset_name}
              >
                <dl className="space-y-1.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <dt className={cn(TYPE_SCALE.caption, SURFACE.muted)}>Files</dt>
                    <dd className="font-mono text-body-md text-ink">{dataset.n_files_display}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-2">
                    <dt className={cn(TYPE_SCALE.caption, SURFACE.muted)}>Modelled</dt>
                    <dd className="font-mono text-body-md text-ink">{dataset.n_modelled_display}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-2">
                    <dt className={cn(TYPE_SCALE.caption, SURFACE.muted)}>Subjects</dt>
                    <dd className="font-mono text-body-md text-ink">{dataset.n_subjects_display}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-2">
                    <dt className={cn(TYPE_SCALE.caption, SURFACE.muted)}>Hours modelled</dt>
                    <dd className="font-mono text-body-md text-ink">{dataset.hours_modelled_display}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-2">
                    <dt className={cn(TYPE_SCALE.caption, SURFACE.muted)}>Native rate</dt>
                    <dd className="font-mono text-body-md text-ink">{dataset.sample_rate_display}</dd>
                  </div>
                </dl>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {dataset.classes.map((label) => (
                    <span
                      key={label}
                      className="rounded-full border border-line bg-sunken px-2 py-0.5 text-label-sm text-ink-2"
                    >
                      {label}
                    </span>
                  ))}
                </div>
              </GlassCard>
            </Link>
          </StaggerItem>
        ))}
      </Stagger>
    </section>
  );
}

/**
 * The landing page's own closing section (T140.6): brand line, site links,
 * a results summary, the technology chips, and the version.
 *
 * T140.6's wording was "version + run id", but the run id and the git
 * commit are exactly what `scripts/45_audit_displayed_values.py`'s
 * `internal_info` check exists to keep off every page -- Phase 127 already
 * settled this (see that footer's own comment) after a session found 189
 * pieces of internal/provenance information on screen. The audit is not
 * weakened for this task; only the version (a plain semver string, not
 * provenance) is shown.
 *
 * Deliberately not a change to the persistent `Footer.tsx` every route
 * renders from the root layout -- that footer stays light on purpose (see
 * its own comment).
 */
export function LandingFooterSummary() {
  const footer = landing.footer;

  return (
    <GlassCard as="section" className="space-y-5" ariaLabel="Results summary">
      <div className="flex flex-wrap items-start justify-between gap-6">
        <div className="max-w-md">
          <p className="text-label-md uppercase text-accent-strong">{footer.brand}</p>
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>{footer.tagline}</p>
        </div>
        <nav aria-label="Site" className="flex flex-wrap gap-x-5 gap-y-1.5">
          {ROUTES.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              className={cn(TYPE_SCALE.caption, 'text-ink-2 hover:text-accent-strong')}
            >
              {route.label}
            </Link>
          ))}
        </nav>
      </div>

      <p className={cn(TYPE_SCALE.body, 'max-w-3xl', SURFACE.muted)}>{footer.results_summary}</p>

      <div className="flex flex-wrap gap-1.5">
        {TECHNOLOGY.map((name) => (
          <span key={name} className="rounded-full border border-line bg-sunken px-2 py-0.5 text-label-sm text-ink-3">
            {name}
          </span>
        ))}
      </div>

      <div
        className={cn(
          'flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line pt-3 font-mono',
          TYPE_SCALE.caption,
          SURFACE.subtle,
        )}
      >
        <span>v{footer.version}</span>
      </div>
    </GlassCard>
  );
}
