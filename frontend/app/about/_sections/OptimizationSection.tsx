import { ConvergencePanel, FrameTable } from '@/app/about/_sections/SearchViews';
import { ResultsTable } from '@/components/table/ResultsTable';
import { Disclosure } from '@/components/ui/Disclosure';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { optimization } from '@/lib/generated/optimization';
import { table } from '@/lib/generated/tables';

/**
 * Search Optimization (T115.5; moved under About the Model by T127.3).
 *
 * Convergence, search space, selected parameters, selected features and the
 * Pareto front, each passed through the exporter as pre-formatted columns.
 *
 * The fold-safety note leads the section because it is the reason there is a
 * curve per outer fold rather than one curve, and a reader seeing five
 * overlapping traces should know immediately that they are five independent
 * searches and not five attempts at the same one.
 */
export function OptimizationSection() {
  return (
    <div className="space-y-6">
      <PageHeader
        level={2}
        title="Search Optimization"
        lede="Seven search runs sit between the raw feature matrix and the deployed model: two over hyperparameters, two over feature masks, one sweep over subset size, one over ensemble weights, and one multi-objective front."
        note={optimization.fold_safety_note}
      />

      <section>
        <SectionHeader eyebrow="Convergence" title="How each search converged" level={2} />
        <div className="mt-4 space-y-3">
          {optimization.runs.map((run) => (
            <ConvergencePanel key={run.run_id} run={run} />
          ))}
        </div>
      </section>

      <section>
        <SectionHeader
          eyebrow="T07"
          title="Search space and selected parameters"
          description="Hyperparameter values render at full precision, deliberately outside the three-decimal metric rule: a C rounded to three places is a different model and cannot be pasted back into a config."
          level={2}
        />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable table={table('T07')} />
        </GlassCard>
      </section>

      <section>
        <SectionHeader eyebrow="Detail" title="Feature count, ensemble weights and search method" level={2} />
        <div className="mt-4 space-y-3">
          <Disclosure
            summary={
              <>
                <span className="mr-2 font-mono text-label-sm uppercase text-ink-3">SO-04</span>
                How performance moves with subset size
              </>
            }
          >
            <p className="mb-3 text-body-sm text-ink-2">
              Each row is one subset size under one ranker, scored across the outer folds.
              Dropping features is a trade, and this is the shape of it.
            </p>
            <GlassCard bodyClassName="p-3">
              <FrameTable frame={optimization.feature_count_curve} />
            </GlassCard>
          </Disclosure>

          <Disclosure
            summary={
              <>
                <span className="mr-2 font-mono text-label-sm uppercase text-ink-3">SO-05</span>
                Ensemble weights, and whether the search moved them
              </>
            }
          >
            <p className="mb-3 text-body-sm text-ink-2">
              Weight stability across folds, and the searched weighting compared against equal
              weighting on the same folds.
            </p>
            <GlassCard eyebrow="Weight stability" bodyClassName="p-3">
              <FrameTable frame={optimization.weight_stability} />
            </GlassCard>
            <GlassCard className="mt-3" eyebrow="Searched against equal weighting" bodyClassName="p-3">
              <FrameTable frame={optimization.equal_vs_optimized} maxRows={25} />
            </GlassCard>
          </Disclosure>

          <Disclosure
            summary={
              <>
                <span className="mr-2 font-mono text-label-sm uppercase text-ink-3">SO-06</span>
                The Pareto front
              </>
            }
          >
            <p className="mb-3 text-body-sm text-ink-2">
              Performance against complexity. A configuration on the front is one no other
              configuration beats on both axes at once; it is not automatically the one to deploy.
            </p>
            <GlassCard bodyClassName="p-3">
              <FrameTable frame={optimization.pareto} />
            </GlassCard>
          </Disclosure>

          <Disclosure
            summary={
              <>
                <span className="mr-2 font-mono text-label-sm uppercase text-ink-3">
                  SO-01 vs SO-02
                </span>
                Random against Bayesian search
              </>
            }
          >
            <p className="mb-3 text-body-sm text-ink-2">
              The same space, the same folds, two search strategies. Both inner and outer scores
              are shown, because an inner-fold improvement that does not survive to the outer
              fold is the search fitting the inner split.
            </p>
            <GlassCard bodyClassName="p-3">
              <FrameTable frame={optimization.method_comparison} />
            </GlassCard>
          </Disclosure>
        </div>
      </section>
    </div>
  );
}
