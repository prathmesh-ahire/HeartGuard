import { ModelComparison } from '@/app/about/_sections/ModelComparison';
import { ResultsTable } from '@/components/table/ResultsTable';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { experiments } from '@/lib/generated/experiments';
import { table } from '@/lib/generated/tables';

/**
 * Model Comparison (T115.3, T115.4; moved under About the Model by T127.3).
 *
 * Server component; the sortable table and the model selector are the client
 * island. Every number in it was formatted in Python.
 *
 * The two notes at the top are load-bearing rather than decorative. The first
 * says accuracy is not the headline, which is what stops a reader sorting by it
 * and drawing a conclusion the selection rule does not support. The second says
 * the five label spaces are separate, which is what stops a PASCAL number being
 * compared against a PhysioNet one.
 */
export function ModelsSection() {
  const unavailable = experiments.experiments.filter((item) => !item.available);

  return (
    <div className="space-y-6">
      <PageHeader
        level={2}
        title="Model Comparison"
        lede={
          <>
            {experiments.n_available} of {experiments.n_declared} declared experiments have
            produced results. Each is a separate run with its own fold map and its own
            configuration; nothing below is pooled across them.
          </>
        }
        note={
          <span className="space-y-1">
            <span className="block">{experiments.selection_note}</span>
            <span className="block">{experiments.label_space_note}</span>
          </span>
        }
      />

      <section>
        <SectionHeader
          eyebrow="Comparison"
          title="Results, fold spread, curves and confusion"
          level={2}
        />
        <div className="mt-4">
          <ModelComparison />
        </div>
      </section>

      <section>
        <SectionHeader
          eyebrow="T06"
          title="What each model is"
          description="The registry of estimators, their search dimensions and whether each is an ensemble."
          level={2}
        />
        <GlassCard className="mt-4" bodyClassName="p-3">
          <ResultsTable table={table('T06')} hideColumns={['unavailable_reason']} />
        </GlassCard>
      </section>

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
                <span className="text-label-md uppercase text-accent-strong">
                  {item.exp_id}
                </span>{' '}
                — {item.title}
                <p className="mt-1 text-ink-3">{item.reason}</p>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
