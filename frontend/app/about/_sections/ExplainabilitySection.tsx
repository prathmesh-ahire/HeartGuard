import { ImportanceView } from '@/app/about/_sections/ImportanceView';
import { LastPrediction } from '@/app/about/_sections/LastPrediction';
import { FigurePanel } from '@/components/charts/FigurePanel';
import { EmptyState } from '@/components/ui/States';
import { GlassCard } from '@/components/ui/GlassCard';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { explainability } from '@/lib/generated/explainability';

const CELL = 'stat px-3 py-1.5 text-body-sm';
const HEAD = 'whitespace-nowrap px-3 py-2 font-mono text-label-sm uppercase text-ink-3';

/**
 * Explainability (T117.2; moved under About the Model by T127.3).
 *
 * Three questions, kept apart because they have different answers: what the
 * models lean on across the corpus (permutation importance, held-out), how that
 * splits by feature family, and what drove ONE recording's decision. The last
 * is exact only for a linear model -- the deployed binary model is one -- and
 * is never substituted by global importance for a model where it is not.
 */
export function ExplainabilitySection() {
  const payload = explainability;
  const example = payload.example;

  return (
    <div className="space-y-6">
      <PageHeader
        level={2}
        title="Explainability"
        lede="Which features the fitted models actually use, measured on held-out rows, and what drove the most recent screening indication made in this browser tab."
        note={
          <ul className="list-disc space-y-1 pl-4">
            {payload.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        }
      />

      {!payload.available ? (
        <EmptyState title="Explainability results are not available" description={payload.reason} />
      ) : (
        <>
          <section>
            <SectionHeader
              eyebrow="Global"
              title="Feature importance"
              description="Permutation importance on each fold's held-out rows. The error is the standard deviation across folds."
              level={2}
            />
            <GlassCard className="mt-4">
              <ImportanceView groups={payload.importance} />
            </GlassCard>
            <GlassCard className="mt-3" eyebrow="G19">
              <FigurePanel className="max-w-4xl" figureId="G19" />
            </GlassCard>
          </section>

          <section>
            <SectionHeader
              eyebrow="Families"
              title="Family-level contribution"
              description="Each family's share of the positive importance mass, averaged over folds, beside how many folds the family's net importance was negative."
              level={2}
            />
            {payload.families.map((group) => (
              <GlassCard
                key={group.task + group.model_id + group.kind}
                className="mt-3"
                eyebrow={group.task + ' · ' + group.model_id + ' · ' + group.kind}
                flush
                bodyClassName="overflow-x-auto"
              >
                <table className="min-w-full border-collapse text-left">
                  <thead className="border-b-2 border-line bg-sunken">
                    <tr>
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
                      <tr key={row.family} className="border-b border-line last:border-0 hover:bg-accent-soft/40">
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
              </GlassCard>
            ))}
            <GlassCard className="mt-3" eyebrow="G18">
              <FigurePanel className="max-w-4xl" figureId="G18" />
            </GlassCard>
            {payload.coverage.length > 0 ? (
              <GlassCard
                className="mt-3"
                eyebrow="Coverage"
                title="Which models were explained, and why not the others"
              >
                <ul className="space-y-1 text-body-sm text-ink-2">
                  {payload.coverage.map((row) => (
                    <li key={row.model_id}>
                      <span className="font-mono">{row.model_id}</span>:{' '}
                      {row.methods.length > 0 ? row.methods.join(', ') : 'not explained'}
                      {row.excluded_reason ? ' — ' + row.excluded_reason : ''}
                    </li>
                  ))}
                </ul>
              </GlassCard>
            ) : null}
          </section>

          <section>
            <SectionHeader
              eyebrow="Per sample · live"
              title="The last prediction in this tab"
              description="The decomposition returned for the most recent recording scored on the Analyse page."
              level={2}
            />
            <div className="mt-4">
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
                  '. Computed offline by the same decomposition, so it can be read without a running service.'
                }
                level={2}
              />
              <dl className="mt-4 grid gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
                <div className="flex gap-2">
                  <dt className="text-ink-3">Model</dt>
                  <dd>
                    {example.task} · {example.model_id}
                  </dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-ink-3">Reference label · indication</dt>
                  <dd>
                    {example.true_class} · {example.predicted_class}
                  </dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-ink-3">Probability</dt>
                  <dd className="tabular-nums">{example.probability_display}</dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-ink-3">Intercept</dt>
                  <dd className="tabular-nums">{example.base_value_display}</dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-ink-3">Decision value</dt>
                  <dd className="tabular-nums">{example.decision_value_display}</dd>
                </div>
                <div className="flex gap-2">
                  <dt className="text-ink-3">Terms shown</dt>
                  <dd>
                    {example.n_shown_display} of {example.n_features_display}
                  </dd>
                </div>
              </dl>
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-full border-collapse text-left">
                  <thead>
                    <tr className="border-b border-line">
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
                      <tr key={row.feature} className="border-b border-line">
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
              <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-ink-2">
                {(example.caveats ?? []).map((caveat) => (
                  <li key={caveat}>{caveat}</li>
                ))}
              </ul>
            </section>
          ) : null}
        </>
      )}
    </div>
  );
}
