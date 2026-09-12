import type { Metadata } from 'next';

import { ModelComparison } from '@/app/models/ModelComparison';
import { ResultsTable } from '@/components/table/ResultsTable';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { experiments } from '@/lib/generated/experiments';
import { table } from '@/lib/generated/tables';

export const metadata: Metadata = {
  title: 'Model Comparison',
  description:
    'Every model across every experiment, with the spread across folds and the curves behind each number.',
};

/**
 * Model Comparison (T115.3, T115.4).
 *
 * Server component; the sortable table and the model selector are the client
 * island. Every number in it came from an experiment's own
 * `aggregate_metrics.csv`, `per_fold_metrics.csv` or `confusion_matrices.json`,
 * formatted in Python.
 *
 * The two notes at the top are load-bearing rather than decorative. The first
 * says accuracy is not the headline, which is what stops a reader sorting by it
 * and drawing a conclusion the selection rule does not support. The second says
 * the five label spaces are separate, which is what stops a PASCAL number being
 * compared against a PhysioNet one.
 */
export default function Page() {
  const declared = experiments.experiments;
  const unavailable = declared.filter((item) => !item.available);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Comparison"
        lede={
          <>
            {experiments.n_available} of {experiments.n_declared} declared experiments have
            produced results. Each is a separate run with its own fold map, its own
            configuration snapshot and its own run manifest; nothing below is pooled
            across them.
          </>
        }
        note={
          <span className="space-y-1">
            <span className="block">{experiments.selection_note}</span>
            <span className="block">{experiments.label_space_note}</span>
          </span>
        }
      />

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="T115.3 / T115.4"
          title="Results, fold spread, curves and confusion"
          level={2}
        />
        <div className="mt-4">
          <ModelComparison />
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      <section>
        <SectionHeader
          eyebrow="T06"
          title="What each model is"
          description="The registry of estimators, their search dimensions and whether each is an ensemble."
          level={2}
        />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable table={table('T06')} />
        </GlassCard>
      </section>

      {/* ----------------------------------------------------------------- */}
      {unavailable.length > 0 ? (
        <section>
          <SectionHeader
            eyebrow="Not yet produced"
            title="Declared experiments with no results"
            description="Listed rather than dropped: a page showing four results where five were declared says nothing about the fifth, and a reader counts what they see."
            level={2}
          />
          <ul className="mt-4 grid gap-2 sm:grid-cols-2">
            {unavailable.map((item) => (
              <li
                key={item.exp_id}
                className="rounded-lg border border-dashed border-line bg-panel p-3 text-body-sm"
              >
                <span className="font-mono text-label-md uppercase text-accent-strong">
                  {item.exp_id}
                </span>{' '}
                — {item.title}
                <p className="mt-1 text-ink-3">{item.reason}</p>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {/* ----------------------------------------------------------------- */}
      <GlassCard as="section" eyebrow="Provenance" bodyClassName="text-body-sm text-ink-2">
        <ul className="space-y-1">
          {declared
            .filter((item) => item.available)
            .map((item) => (
              <li key={item.exp_id}>
                <span className="font-medium">{item.exp_id}</span> · run{' '}
                <span className="font-mono">{item.run_id ?? 'n/a'}</span> · commit{' '}
                <span className="font-mono">{(item.git_commit ?? 'n/a').slice(0, 12)}</span> ·
                seed {item.seed ?? 'n/a'} · {item.cv ?? 'n/a'}
              </li>
            ))}
        </ul>
      </GlassCard>
    </div>
  );
}
