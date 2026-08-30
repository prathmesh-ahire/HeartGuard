import Link from 'next/link';

import { manifest } from '@/lib/generated';
import { GROUP_LABELS, ROUTES } from '@/lib/routes';

/**
 * Home, in its Phase 110 scaffold form: the route map and the provenance of
 * this build. The hero, the 3D heart, the six locked objectives and the
 * animated pipeline arrive in Phase 114 (T114.1).
 *
 * The counts below come from `manifest`, which Python wrote. Nothing on this
 * page is typed in, and nothing is a result.
 */
export default function HomePage() {
  const groups = (['overview', 'method', 'results', 'predict'] as const).map((group) => ({
    group,
    routes: ROUTES.filter((route) => route.group === group && route.href !== '/'),
  }));

  return (
    <div className="space-y-10">
      <section className="max-w-3xl">
        <p className="text-xs uppercase tracking-widest text-slate-500">
          Phonocardiogram heart-sound classification
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">
          {manifest.framework}
        </h1>
        <p className="mt-4 text-slate-600 dark:text-slate-400">
          A search-optimized heterogeneous ensemble over engineered acoustic features,
          evaluated across four public PCG corpora under subject-grouped
          cross-validation. This site is the reporting surface for that work: every
          precomputed value it shows is generated from the pipeline&rsquo;s own output
          files at build time.
        </p>
        <div className="mt-6 rounded border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-600 dark:border-slate-700 dark:bg-slate-900/40 dark:text-slate-400">
          <p className="font-medium text-slate-800 dark:text-slate-200">
            Route tree scaffolded; page content is built in Phases 114&ndash;117.
          </p>
          <p className="mt-1">
            The hero, the six research objectives and the animated pipeline walkthrough
            belong to Phase 114. Nothing on this page is a measured result.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
          Pages
        </h2>
        <div className="mt-4 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {groups.map(({ group, routes }) => (
            <div key={group}>
              <h3 className="text-xs uppercase tracking-widest text-slate-400 dark:text-slate-500">
                {GROUP_LABELS[group]}
              </h3>
              <ul className="mt-2 space-y-3">
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

      <section>
        <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
          This build
        </h2>
        <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
            <dt className="text-xs uppercase tracking-widest text-slate-500">Tables</dt>
            <dd className="mt-1 text-2xl font-semibold tabular-nums">
              {manifest.n_tables}
            </dd>
          </div>
          <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
            <dt className="text-xs uppercase tracking-widest text-slate-500">Figures</dt>
            <dd className="mt-1 text-2xl font-semibold tabular-nums">
              {manifest.n_figures}
            </dd>
          </div>
          <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
            <dt className="text-xs uppercase tracking-widest text-slate-500">
              Source files
            </dt>
            <dd className="mt-1 text-2xl font-semibold tabular-nums">
              {manifest.sources.length}
            </dd>
          </div>
          <div className="rounded border border-slate-200 p-4 dark:border-slate-800">
            <dt className="text-xs uppercase tracking-widest text-slate-500">Pages</dt>
            <dd className="mt-1 text-2xl font-semibold tabular-nums">{ROUTES.length}</dd>
          </div>
        </dl>
      </section>
    </div>
  );
}
