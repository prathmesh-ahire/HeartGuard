import type { Metadata } from 'next';

import { ImportanceView } from '@/app/explainability/ImportanceView';
import { LastPrediction } from '@/app/explainability/LastPrediction';
import { FigurePanel } from '@/components/charts/FigurePanel';
import { EvidenceLink } from '@/components/evidence/EvidenceLink';
import { EmptyState } from '@/components/ui/States';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { explainability } from '@/lib/generated/explainability';

export const metadata: Metadata = {
  title: 'Explainability',
  description:
    'Global feature importance, family-level contribution, and a per-sample explanation of the most recent prediction.',
};

const CELL = 'px-3 py-1.5 text-xs tabular-nums';
const HEAD = 'px-3 py-2 text-xs font-semibold';

/**
 * Explainability (T117.2).
 *
 * Three questions, kept apart because they have different answers: what the
 * models lean on across the corpus (permutation importance, held-out), how that
 * splits by feature family, and what drove ONE recording's decision. The last
 * is exact only for a linear model -- the deployed binary model is one -- and
 * is never substituted by global importance for a model where it is not.
 */
export default function Page() {
  const payload = explainability;
  const example = payload.example;
  const sourceOf = (stem: string): string =>
    payload.sources.find((source) => source.path.endsWith(stem))?.path ?? stem;

  return (
    <div className="space-y-14">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight">Explainability</h1>
        <p className="mt-3 max-w-3xl text-slate-600 dark:text-slate-400">
          Which of the 138 features the fitted models actually use, measured on held-out rows,
          and what drove the most recent screening indication made in this browser tab.
        </p>
        <ul className="mt-3 max-w-3xl list-disc space-y-1 pl-5 text-sm text-slate-600 dark:text-slate-400">
          {payload.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </section>

      {!payload.available ? (
        <EmptyState title="Explainability outputs are not in this build" description={payload.reason} />
      ) : (
        <>
          <section>
            <SectionHeader
              eyebrow="Global"
              title="Feature importance"
              description="Permutation importance on each fold's held-out rows. The error is the standard deviation across folds."
              level={2}
            />
            <div className="mt-5">
              <ImportanceView groups={payload.importance} />
            </div>
            <p className="mt-3 text-xs text-slate-500">
              source <EvidenceLink path={sourceOf('importance_summary.csv')} artifact="explainability" />
            </p>
            <FigurePanel className="mt-8 max-w-4xl" figureId="G19" />
          </section>

          <section>
            <SectionHeader
              eyebrow="Families"
              title="Family-level contribution"
              description="Each family's share of the positive importance mass, averaged over folds, beside how many folds the family's net importance was negative."
              level={2}
            />
            {payload.families.map((group) => (
              <div key={group.task + group.model_id + group.kind} className="mt-5 overflow-x-auto">
                <p className="mb-2 text-xs text-slate-500">
                  {group.task} · {group.model_id} · {group.kind}
                </p>
                <table className="min-w-full border-collapse text-left">
                  <thead>
                    <tr className="border-b border-slate-300 dark:border-slate-700">
                      <th scope="col" className={HEAD}>Rank</th>
                      <th scope="col" className={HEAD}>Family</th>
                      <th scope="col" className={HEAD}>Features</th>
                      <th scope="col" className={HEAD}>Share of positive importance</th>
                      <th scope="col" className={HEAD}>Total importance</th>
                      <th scope="col" className={HEAD}>Folds net negative</th>
                    </tr>
                  </thead>
                  <tbody>
                    {group.rows.map((row) => (
                      <tr key={row.family} className="border-b border-slate-200 dark:border-slate-800">
                        <td className={CELL}>{row.rank}</td>
                        <td className={CELL}>{row.family}</td>
                        <td className={CELL}>{row.n_features_display}</td>
                        <td className={CELL}>
                          {row.share_display} ± {row.share_sd_display}
                        </td>
                        <td className={CELL}>{row.total_display}</td>
                        <td className={CELL}>{row.net_negative_display}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
            <p className="mt-3 text-xs text-slate-500">
              source{' '}
              <EvidenceLink path={sourceOf('feature_family_importance.csv')} artifact="explainability" />
            </p>
            <FigurePanel className="mt-8 max-w-4xl" figureId="G18" />
            {payload.coverage.length > 0 ? (
              <div className="mt-6">
                <p className="text-sm font-medium">Which models were explained, and why not the others</p>
                <ul className="mt-2 space-y-1 text-xs text-slate-600 dark:text-slate-400">
                  {payload.coverage.map((row) => (
                    <li key={row.model_id}>
                      <span className="font-mono">{row.model_id}</span>:{' '}
                      {row.methods.length > 0 ? row.methods.join(', ') : 'not explained'}
                      {row.excluded_reason ? ' — ' + row.excluded_reason : ''}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </section>

          <section>
            <SectionHeader
              eyebrow="Per sample · live"
              title="The last prediction in this tab"
              description="The decomposition the inference API returned for the most recent recording scored on a prediction page."
              level={2}
            />
            <div className="mt-5">
              <LastPrediction />
            </div>
          </section>

          {example.available && example.rows ? (
            <section>
              <SectionHeader
                eyebrow="Per sample · stored"
                title={'A worked example: ' + (example.record_uid ?? '')}
                description={
                  'Selected as the ' +
                  (example.selection_rule ?? '') +
                  '. Computed offline by the same decomposition, so it can be read without a running API.'
                }
                level={2}
              />
              <dl className="mt-4 grid gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
                <div className="flex gap-2">
                  <dt className="text-slate-500">Model</dt>
                  <dd>
                    {example.task} · {example.model_id}
                  </dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-slate-500">Reference label · indication</dt>
                  <dd>
                    {example.true_class} · {example.predicted_class}
                  </dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-slate-500">Probability</dt>
                  <dd className="tabular-nums">{example.probability_display}</dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-slate-500">Intercept</dt>
                  <dd className="tabular-nums">{example.base_value_display}</dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-slate-500">Decision value</dt>
                  <dd className="tabular-nums">{example.decision_value_display}</dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-slate-500">Terms shown</dt>
                  <dd>
                    {example.n_shown_display} of {example.n_features_display}
                  </dd>
                </div>
              </dl>
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-full border-collapse text-left">
                  <thead>
                    <tr className="border-b border-slate-300 dark:border-slate-700">
                      <th scope="col" className={HEAD}>Feature</th>
                      <th scope="col" className={HEAD}>Family</th>
                      <th scope="col" className={HEAD}>Raw value</th>
                      <th scope="col" className={HEAD}>Scaled</th>
                      <th scope="col" className={HEAD}>Weight</th>
                      <th scope="col" className={HEAD}>Contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {example.rows.map((row) => (
                      <tr key={row.feature} className="border-b border-slate-200 dark:border-slate-800">
                        <td className={CELL + ' font-mono'}>{row.feature}</td>
                        <td className={CELL}>{row.family}</td>
                        <td className={CELL}>{row.raw_value_display}</td>
                        <td className={CELL}>{row.scaled_value_display}</td>
                        <td className={CELL}>{row.weight_display}</td>
                        <td className={CELL}>
                          {row.contribution_display} {row.direction}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-slate-600 dark:text-slate-400">
                {(example.caveats ?? []).map((caveat) => (
                  <li key={caveat}>{caveat}</li>
                ))}
              </ul>
              <p className="mt-3 text-xs text-slate-500">
                source{' '}
                <EvidenceLink path={sourceOf('per_sample_explanation.json')} artifact="explainability" />
              </p>
            </section>
          ) : null}
        </>
      )}
    </div>
  );
}
