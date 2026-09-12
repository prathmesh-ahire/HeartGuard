import Link from 'next/link';

import { Objectives } from '@/components/objectives/Objectives';
import { Hero3D } from '@/components/three/Hero3D';
import { PipelineWalkthrough } from '@/components/pipeline/PipelineWalkthrough';
import { ButtonLink } from '@/components/ui/Button';
import { GlassCard } from '@/components/ui/GlassCard';
import { Icon, type IconName } from '@/components/ui/Icon';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { StatTile } from '@/components/ui/StatTile';
import { datasetSummary, manifest } from '@/lib/generated';
import { GROUP_LABELS, ROUTES } from '@/lib/routes';

/**
 * Home (T114.1): the hero, the six locked objectives verbatim, the animated
 * pipeline, and the dataset summary tiles.
 *
 * A server component. The interactive parts — the 3D heart and the scroll-driven
 * pipeline — are client components imported into it, so KaTeX's mistake from
 * Phase 112 (a server component pulled across a client boundary and into the
 * browser bundle) cannot repeat here: nothing on this page renders a formula
 * and nothing on it computes.
 *
 * Every number below comes from `generated/`, formatted in Python. The tiles
 * read `datasetSummary.summary`, which reports **both** populations per corpus, because
 * the corpus and the modelled subset differ for three of the four families and a
 * tile showing one while the reader assumes the other is how a wrong count gets
 * into a thesis.
 */
const PAGE_ICONS: Record<string, IconName> = {
  '/dataset/': 'dataset',
  '/preprocessing/': 'waveform',
  '/features/': 'features',
  '/models/': 'models',
  '/optimization/': 'optimization',
  '/robustness/': 'robustness',
  '/explainability/': 'explainability',
  '/reports/': 'reports',
  '/predict/binary/': 'binary',
  '/predict/multiclass/': 'multiclass',
  '/predict/murmur/': 'murmur',
};

export default function HomePage() {
  const groups = (['overview', 'method', 'results', 'predict'] as const).map((group) => ({
    group,
    routes: ROUTES.filter((route) => route.group === group && route.href !== '/'),
  }));

  return (
    <div className="space-y-6">
      {/* ----------------------------------------------------------------- */}
      {/* The hero reads as the instrument's front panel: identity on the
          left, the specimen on the right, and the operating parameters along
          the bottom rail in instrument type. */}
      <GlassCard marked flush as="section" className="relative">
        <div
          aria-hidden="true"
          className="telemetry-grid pointer-events-none absolute inset-0"
        />
        <div className="relative grid items-center gap-6 p-6 lg:grid-cols-[1.15fr_1fr] lg:p-8">
          <div>
            <p className="label-micro flex items-center gap-2">
              <span aria-hidden="true" className="h-2.5 w-0.5 bg-accent" />
              Phonocardiogram heart-sound classification
            </p>
            <h1 className="mt-3 text-headline-xl text-ink sm:text-[40px] sm:leading-[46px]">
              {manifest.framework}
            </h1>
            <p className="mt-4 max-w-xl text-body-lg text-ink-2">
              A search-optimized heterogeneous ensemble over engineered acoustic
              features, evaluated across four public PCG corpora under subject-grouped
              cross-validation.
            </p>
            <p className="mt-3 max-w-xl text-body-md text-ink-3">
              This site is the reporting surface for that work. Every precomputed value
              it shows was generated from the pipeline&rsquo;s own output files at build
              time and can be traced back to the CSV that produced it. The only thing
              computed while you are here is a prediction you ask for yourself.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <ButtonLink href="/dataset/" tone="primary" size="lg" icon="dataset">
                Explore the corpus
              </ButtonLink>
              <ButtonLink href="/predict/binary/" size="lg" icon="upload">
                Screen a recording
              </ButtonLink>
            </div>
          </div>
          <Hero3D height="20rem" interactive />
        </div>

        {/* The operating parameters, as an instrument rail. Every value here is
            read from the export manifest -- nothing is stated about the run
            that the run did not record. */}
        <dl className="relative flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-line bg-sunken px-6 py-2.5">
          {[
            { term: 'Corpora', detail: datasetSummary.summary.length },
            { term: 'Tables exported', detail: manifest.n_tables },
            { term: 'Figures exported', detail: manifest.n_figures },
            { term: 'Branch', detail: manifest.git_branch ?? 'unknown' },
          ].map((entry) => (
            <div key={entry.term} className="flex items-center gap-2">
              <dt className="label-micro">{entry.term}</dt>
              <dd className="stat font-mono text-label-md text-ink">{entry.detail}</dd>
            </div>
          ))}
        </dl>
      </GlassCard>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="Corpus"
          title="Four public datasets, audited against the files on disk"
          description={datasetSummary.scope_note}
        />
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {datasetSummary.summary.map((row, index) => (
            <StatTile
              key={row.dataset_source}
              marked={index === 0}
              label={row.dataset_name}
              display={row.n_modelled_display}
              value={row.n_modelled}
              unit="modelled"
              source={datasetSummary.source}
              hint={
                <>
                  {row.n_files_display} files on disk · {row.n_subjects_display} subjects ·{' '}
                  {row.hours_modelled_display} hours modelled
                </>
              }
            />
          ))}
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="Scope"
          title="The six locked objectives"
          description="Quoted exactly as the source document fixes them."
        />
        <Objectives className="mt-4" />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="Method"
          title="From a recording to a screened result"
          description="Twelve steps, each naming the module that implements it and the outputs directory that evidences it. Both are checked when this page is built."
        />
        <PipelineWalkthrough className="mt-4" />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader eyebrow="Contents" title="Pages" level={2} />
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {groups.map(({ group, routes }) => (
            <GlassCard key={group} eyebrow={GROUP_LABELS[group]} flush>
              <ul className="divide-y divide-line">
                {routes.map((route) => (
                  <li key={route.href}>
                    <Link
                      href={route.href}
                      className="group flex gap-2.5 p-3 transition-colors hover:bg-accent-soft/50"
                    >
                      <Icon
                        name={PAGE_ICONS[route.href] ?? 'features'}
                        className="mt-0.5 h-4 w-4 shrink-0 text-ink-3 transition-colors group-hover:text-accent"
                      />
                      <span className="min-w-0">
                        <span className="block font-mono text-label-md uppercase text-accent-strong">
                          {route.label}
                        </span>
                        <span className="mt-1 block text-body-sm text-ink-3">
                          {route.summary}
                        </span>
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            </GlassCard>
          ))}
        </div>
      </section>
    </div>
  );
}
