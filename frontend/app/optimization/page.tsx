import type { Metadata } from 'next';

import { ConvergencePanel, FrameTable } from '@/app/optimization/SearchViews';
import { ResultsTable } from '@/components/table/ResultsTable';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { optimization } from '@/lib/generated/optimization';
import { table } from '@/lib/generated/tables';

export const metadata: Metadata = {
  title: 'Search Optimization',
  description:
    'The seven search runs: what each searched over, how it converged, and what it selected.',
};

/**
 * Search Optimization (T115.5).
 *
 * Convergence, search space, selected parameters, selected features and the
 * Pareto front. Every frame is a committed CSV under
 * `outputs/05_search_optimization/`, passed through the exporter as
 * pre-formatted columns.
 *
 * The fold-safety note leads the page because it is the reason there is a curve
 * per outer fold rather than one curve, and a reader seeing five overlapping
 * traces should know immediately that they are five independent searches and
 * not five attempts at the same one.
 */
export default function Page() {
  return (
    <div className="space-y-14">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight">Search Optimization</h1>
        <p className="mt-3 max-w-3xl text-slate-600 dark:text-slate-400">
          Seven search runs sit between the raw feature matrix and the deployed model:
          two over hyperparameters, two over feature masks, one sweep over subset size,
          one over ensemble weights, and one multi-objective front.
        </p>
        <p className="mt-4 max-w-3xl rounded border-l-2 border-sky-400 bg-sky-50/60 py-2 pl-3 text-sm text-sky-900 dark:bg-sky-950/30 dark:text-sky-200">
          {optimization.fold_safety_note}
        </p>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader eyebrow="T115.5" title="How each search converged" level={2} />
        <div className="mt-6 space-y-10">
          {optimization.runs.map((run) => (
            <ConvergencePanel key={run.run_id} run={run} />
          ))}
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="T07"
          title="Search space and selected parameters"
          description="Hyperparameter values render at full precision, deliberately outside the three-decimal metric rule: a C rounded to three places is a different model and cannot be pasted back into a config."
          level={2}
        />
        <ResultsTable className="mt-5" table={table('T07')} />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="SO-04"
          title="How performance moves with subset size"
          description="Each row is one subset size under one ranker, scored across the outer folds. Dropping features is a trade, and this is the shape of it."
          level={2}
        />
        <FrameTable className="mt-5" frame={optimization.feature_count_curve} />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="SO-05"
          title="Ensemble weights, and whether the search moved them"
          description="Weight stability across folds, and the searched weighting compared against equal weighting on the same folds."
          level={2}
        />
        <FrameTable className="mt-5" frame={optimization.weight_stability} />
        <FrameTable className="mt-8" frame={optimization.equal_vs_optimized} maxRows={25} />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="SO-06"
          title="The Pareto front"
          description="Performance against complexity. A configuration on the front is one no other configuration beats on both axes at once; it is not automatically the one to deploy."
          level={2}
        />
        <FrameTable className="mt-5" frame={optimization.pareto} />
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="SO-01 vs SO-02"
          title="Random against Bayesian search"
          description="The same space, the same folds, two search strategies. Both inner and outer scores are shown, because an inner-fold improvement that does not survive to the outer fold is the search fitting the inner split."
          level={2}
        />
        <FrameTable className="mt-5" frame={optimization.method_comparison} />
      </section>
    </div>
  );
}
