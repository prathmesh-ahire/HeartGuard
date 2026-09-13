import { FeatureExplorer } from '@/app/about/_sections/FeatureExplorer';
import { GroupedBars } from '@/components/charts/Charts';
import { FigureDownload } from '@/components/charts/FigureDownload';
import { EquationList } from '@/components/equations/Equations';
import { ResultsTable } from '@/components/table/ResultsTable';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { StatTile } from '@/components/ui/StatTile';
import { G10 } from '@/lib/generated/figures/G10';
import { features } from '@/lib/generated/features';
import { table } from '@/lib/generated/tables';

/**
 * Feature Extraction (T115.1, T115.2; moved under About the Model by T127.3).
 *
 * A **server** component. `EquationList` renders KaTeX at build time and must
 * never cross a client boundary — Phase 112 lost that property when `/design`
 * imported it from a `'use client'` page and 74 kB of KaTeX went into the
 * browser with nothing on screen changing. The interactive registry table is a
 * separate client island.
 */
export function FeaturesSection() {
  const selected = features.selected;

  return (
    <div className="space-y-6">
      <PageHeader
        level={2}
        title="Feature Extraction"
        lede="Every recording becomes the same fixed vector, in the same order, whichever corpus it came from. That is what makes a PASCAL recording and a PhysioNet one comparable at all, and its column order is a locked literal rather than something the extractor happens to produce."
        note="The column order is fixed and fingerprinted; two runs that disagree on it are not comparable. The feature list below is in that order and is never sorted by name."
      />

      <section>
        <SectionHeader eyebrow="Composition" title="The six families" level={2} />
        <div className="mt-4 grid gap-3 sm:grid-cols-3 xl:grid-cols-6">
          {features.families.map((item) => (
            <StatTile
              key={item.family}
              label={item.family}
              display={item.n_features_display}
              value={item.n_features}
              unit="features"
            />
          ))}
        </div>
        <GlassCard className="mt-3" eyebrow="G10" title="Features per family">
          <GroupedBars
            source={G10}
            categoryColumn="family"
            valueColumns={['n_features']}
            label="Features per family"
            caption="G10 — the family composition of the 138-vector, in registry order."
            height={300}
          />
          <FigureDownload figureId="G10" className="mt-2" />
        </GlassCard>
      </section>

      <section>
        <SectionHeader eyebrow="T05" title="Feature inventory and counts" level={2} />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable
            table={table('T05')}
            caption="The per-family counts are recomputed from the registry inventory and checked against the extractor's own summary; the table refuses to build if they disagree or if the total is wrong."
          />
        </GlassCard>
      </section>

      <section>
        <SectionHeader
          eyebrow="Registry"
          title="Every feature, and one record's values"
          description="The registry in its locked order, filterable by family or name. Turn on the value column to see the full vector for a single recording."
          level={2}
        />
        <div className="mt-4">
          <FeatureExplorer />
        </div>
      </section>

      <section>
        <SectionHeader
          eyebrow="Selection"
          title="What the search kept"
          description={selected.stability_note ?? undefined}
          level={2}
        />
        {selected.available ? (
          <div className="mt-4 overflow-x-auto rounded-lg border border-line bg-panel">
            <table className="min-w-full border-collapse text-left text-body-sm">
              <thead className="border-b-2 border-line bg-sunken font-mono text-label-sm uppercase text-ink-3">
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
                    className="border-t border-line hover:bg-accent-soft/40"
                  >
                    <td className="stat px-3 py-1.5">{String(row.rank)}</td>
                    <td className="px-3 py-1.5 font-mono">{String(row.feature)}</td>
                    <td className="px-3 py-1.5">{String(row.family)}</td>
                    <td className="px-3 py-1.5 text-ink-3">{String(row.ranker)}</td>
                    <td className="stat px-3 py-1.5">
                      {String(row.selected_in_folds)} of {String(row.n_folds)} folds
                    </td>
                    <td className="stat px-3 py-1.5">{String(row.share_display)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-body-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
            {selected.reason}
          </p>
        )}
      </section>

      <section>
        <SectionHeader
          eyebrow="Definitions"
          title="The formulas behind the families"
          description="Rendered from the source document's own formulas, each cross-checked against the code that implements it."
          level={2}
        />
        <EquationList className="mt-4" />
      </section>
    </div>
  );
}
