import type { Metadata } from 'next';

import { FeatureExplorer } from '@/app/features/FeatureExplorer';
import { GroupedBars } from '@/components/charts/Charts';
import { FigureDownload } from '@/components/charts/FigureDownload';
import { EquationList } from '@/components/equations/Equations';
import { ResultsTable } from '@/components/table/ResultsTable';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { StatTile } from '@/components/ui/StatTile';
import { figure } from '@/lib/generated/figures';
import { features } from '@/lib/generated/features';
import { table } from '@/lib/generated/tables';

export const metadata: Metadata = {
  title: 'Feature Extraction',
  description: 'The 138 engineered features, their families, and which of them the search kept.',
};

/**
 * Feature Extraction (T115.1, T115.2).
 *
 * A **server** component. `EquationList` renders KaTeX at build time and must
 * never cross a client boundary — Phase 112 lost that property when `/design`
 * imported it from a `'use client'` page and 74 kB of KaTeX went into the
 * browser with nothing on screen changing. The interactive registry table is a
 * separate client island.
 */
export default function Page() {
  const selected = features.selected;

  return (
    <div className="space-y-14">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight">Feature Extraction</h1>
        <p className="mt-3 max-w-3xl text-slate-600 dark:text-slate-400">
          Every recording becomes the same 138 numbers, in the same order, whichever
          corpus it came from. That fixed vector is what makes a PASCAL recording and a
          PhysioNet one comparable at all, and its column order is a locked literal
          rather than something the extractor happens to produce.
        </p>
        <p className="mt-3 max-w-3xl text-sm text-slate-600 dark:text-slate-400">
          {features.registry_note}
        </p>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader eyebrow="Composition" title="The six families" level={2} />
        <div className="mt-6 grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
          {features.families.map((item) => (
            <StatTile
              key={item.family}
              label={item.family}
              display={item.n_features_display}
              value={item.n_features}
              unit="features"
              source="outputs/03_features/feature_inventory.csv"
            />
          ))}
        </div>
        <div className="mt-8">
          <GroupedBars
            source={figure('G10')}
            categoryColumn="family"
            valueColumns={['n_features']}
            label="Features per family"
            caption="G10 — the family composition of the 138-vector, in registry order."
            height={300}
          />
          <FigureDownload figureId="G10" className="mt-2" />
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader eyebrow="T05" title="Feature inventory and counts" level={2} />
        <ResultsTable
          className="mt-5"
          table={table('T05')}
          caption="The per-family counts are recomputed from the 138-row inventory and checked against the extractor's own summary; the table refuses to build if they disagree or if the total is not 138."
        />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="T115.1"
          title="Every feature, and one record's values"
          description="The registry in its locked order, filterable by family or name. Turn on the value column to see the full vector for a single recording."
          level={2}
        />
        <div className="mt-6">
          <FeatureExplorer />
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="T115.2"
          title="What the search kept"
          description={selected.stability_note ?? undefined}
          level={2}
        />
        {selected.available ? (
          <div className="mt-5 overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase tracking-widest text-slate-500 dark:bg-slate-900/60">
                <tr>
                  <th scope="col" className="px-3 py-2">
                    Rank
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Feature
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Family
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Ranker
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Kept in
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Share of folds
                  </th>
                </tr>
              </thead>
              <tbody>
                {selected.features.map((row) => (
                  <tr
                    key={String(row.feature)}
                    className="border-t border-slate-100 dark:border-slate-800"
                  >
                    <td className="px-3 py-1.5 tabular-nums">{String(row.rank)}</td>
                    <td className="px-3 py-1.5 font-mono">{String(row.feature)}</td>
                    <td className="px-3 py-1.5">{String(row.family)}</td>
                    <td className="px-3 py-1.5 text-slate-500">{String(row.ranker)}</td>
                    <td className="px-3 py-1.5 tabular-nums">
                      {String(row.selected_in_folds)} of {String(row.n_folds)} folds
                    </td>
                    <td className="px-3 py-1.5 tabular-nums">{String(row.share_display)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="mt-5 text-sm text-amber-800 dark:text-amber-300">{selected.reason}</p>
        )}
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="Definitions"
          title="The formulas behind the families"
          description="Rendered at build time from the source document's own section 11, each cross-checked against the module that implements it."
          level={2}
        />
        <EquationList className="mt-6" />
      </section>
    </div>
  );
}
