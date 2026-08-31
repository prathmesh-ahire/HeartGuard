import Link from 'next/link';

import { Objectives } from '@/components/objectives/Objectives';
import { Hero3D } from '@/components/three/Hero3D';
import { PipelineWalkthrough } from '@/components/pipeline/PipelineWalkthrough';
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
export default function HomePage() {
  const groups = (['overview', 'method', 'results', 'predict'] as const).map((group) => ({
    group,
    routes: ROUTES.filter((route) => route.group === group && route.href !== '/'),
  }));

  return (
    <div className="space-y-16">
      {/* ----------------------------------------------------------------- */}
      <section className="grid items-center gap-8 lg:grid-cols-2">
        <div>
          <p className="text-xs uppercase tracking-widest text-slate-500">
            Phonocardiogram heart-sound classification
          </p>
          <h1 className="mt-2 text-4xl font-semibold tracking-tight sm:text-5xl">
            {manifest.framework}
          </h1>
          <p className="mt-5 max-w-xl text-lg text-slate-600 dark:text-slate-400">
            A search-optimized heterogeneous ensemble over engineered acoustic
            features, evaluated across four public PCG corpora under subject-grouped
            cross-validation.
          </p>
          <p className="mt-4 max-w-xl text-slate-600 dark:text-slate-400">
            This site is the reporting surface for that work. Every precomputed value
            it shows was generated from the pipeline&rsquo;s own output files at build
            time and can be traced back to the CSV that produced it. The only thing
            computed while you are here is a prediction you ask for yourself.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link
              href="/dataset/"
              className="rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-800 dark:bg-sky-600 dark:hover:bg-sky-500"
            >
              Explore the corpus
            </Link>
            <Link
              href="/predict/binary/"
              className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-900"
            >
              Screen a recording
            </Link>
          </div>
        </div>
        <Hero3D height="22rem" interactive />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="Corpus"
          title="Four public datasets, audited against the files on disk"
          description={datasetSummary.scope_note}
        />
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {datasetSummary.summary.map((row) => (
            <StatTile
              key={row.dataset_source}
              label={row.dataset_name}
              display={row.n_modelled_display}
              value={row.n_modelled}
              unit="modelled recordings"
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
        <Objectives className="mt-6" />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="Method"
          title="From a recording to a screened result"
          description="Twelve steps, each naming the module that implements it and the outputs directory that evidences it. Both are checked when this page is built."
        />
        <PipelineWalkthrough className="mt-6" />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader eyebrow="Contents" title="Pages" level={2} />
        <div className="mt-6 grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {groups.map(({ group, routes }) => (
            <div key={group}>
              <h3 className="text-xs uppercase tracking-widest text-slate-400 dark:text-slate-500">
                {GROUP_LABELS[group]}
              </h3>
              <ul className="mt-3 space-y-3">
                {routes.map((route) => (
                  <li key={route.href}>
                    <Link
                      href={route.href}
                      className="font-medium text-sky-700 hover:underline dark:text-sky-400"
                    >
                      {route.label}
                    </Link>
                    <p className="mt-0.5 text-sm text-slate-600 dark:text-slate-400">
                      {route.summary}
                    </p>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
