import type { Metadata } from 'next';
import Link from 'next/link';

import { EvidenceLink } from '@/components/evidence/EvidenceLink';
import { EmptyState } from '@/components/ui/States';
import { SectionHeader } from '@/components/ui/SectionHeader';
import type { GeneratedPayloadSource } from '@/lib/generated/types';
import { limitations } from '@/lib/generated/limitations';

export const metadata: Metadata = {
  title: 'Limitations',
  description:
    'What the results cannot support: population mismatch, PASCAL sample sizes, the CirCor public subset and the recording-source confound.',
};

const CELL = 'px-3 py-1.5 text-xs tabular-nums';
const HEAD = 'px-3 py-2 text-xs font-semibold';

function Sources({ sources }: { sources: GeneratedPayloadSource[] }) {
  return (
    <p className="mt-3 text-xs text-slate-500">
      {sources.length === 1 ? 'source ' : 'sources '}
      {sources.map((source, index) => (
        <span key={source.path}>
          {index > 0 ? ', ' : null}
          <EvidenceLink path={source.path} />
        </span>
      ))}
    </p>
  );
}

/**
 * Limitations (T117.5).
 *
 * Every number on this page is a cell read from a committed file by
 * `src/reporting/pages_10_12.py` and formatted there. The prose is fixed; the
 * numbers are not typed into it.
 */
export default function Page() {
  const { population, pascal, circor, source_holdout: holdout } = limitations;

  return (
    <div className="max-w-4xl space-y-14">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight">Limitations</h1>
        <p className="mt-3 text-slate-600 dark:text-slate-400">
          What the results on this site can and cannot support. These are properties of the
          public corpora, not open bugs, and each one is stated in the thesis rather than worked
          around. Read them before quoting any number from the{' '}
          <Link href="/models/" className="underline">
            model
          </Link>{' '}
          or{' '}
          <Link href="/robustness/" className="underline">
            robustness
          </Link>{' '}
          pages.
        </p>
      </section>

      <section>
        <SectionHeader
          eyebrow="EXP-D1"
          title="The cross-dataset test is adult to paediatric"
          level={2}
        />
        {population.available ? (
          <div className="mt-4 space-y-3 text-sm text-slate-700 dark:text-slate-300">
            <p>{population.design}</p>
            <p>
              The training corpus, {population.train_dataset}, records a numeric age for{' '}
              {population.train_n_with_age_display} recordings: median{' '}
              {population.train_age_median_display} years, with {population.train_n_under_18_display}{' '}
              under 18 ({population.train_share_under_18_display}). The test corpus,{' '}
              {population.test_dataset}, has {population.test_n_patients_display} patients, of whom{' '}
              {population.test_n_with_age_band_display} have an age band and{' '}
              {population.test_n_paediatric_display} of those are paediatric (
              {population.test_share_paediatric_display}). CirCor records an age band, not an age:{' '}
              {population.test_age_scale}.
            </p>
            <p className="font-medium">{population.framing_rule}</p>
            <p>{population.second_known_cause}</p>
            {population.transfer && population.transfer.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse text-left">
                  <caption className="caption-bottom pt-2 text-left text-xs text-slate-500">
                    Balanced accuracy of the deployed binary model in-domain (PhysioNet 2016
                    cross-validation) and applied unchanged to CirCor, from T16.
                  </caption>
                  <thead>
                    <tr className="border-b border-slate-300 dark:border-slate-700">
                      <th scope="col" className={HEAD}>Level</th>
                      <th scope="col" className={HEAD}>Collapse rule</th>
                      <th scope="col" className={HEAD}>Units</th>
                      <th scope="col" className={HEAD}>In-domain</th>
                      <th scope="col" className={HEAD}>CirCor</th>
                    </tr>
                  </thead>
                  <tbody>
                    {population.transfer.map((row) => (
                      <tr key={row.level + row.rule} className="border-b border-slate-200 dark:border-slate-800">
                        <td className={CELL}>{row.level}</td>
                        <td className={CELL}>{row.rule}</td>
                        <td className={CELL}>{row.n_units_display}</td>
                        <td className={CELL}>{row.in_domain_display}</td>
                        <td className={CELL}>{row.external_display}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            <Sources sources={population.sources} />
          </div>
        ) : (
          <EmptyState className="mt-4" title="Not generated" description={population.reason} />
        )}
      </section>

      <section>
        <SectionHeader
          eyebrow="EXP-F3"
          title="Pooled PhysioNet results do not survive an unseen recording setup"
          level={2}
        />
        {holdout.available ? (
          <div className="mt-4 space-y-3 text-sm text-slate-700 dark:text-slate-300">
            <p>
              Holding out one PhysioNet sub-collection at a time ({holdout.n_folds_display} folds),
              the deployed model {holdout.model_id} falls from a pooled AUC of{' '}
              {holdout.auc_pooled_display} to {holdout.auc_holdout_display}, and from a pooled
              balanced accuracy of {holdout.balanced_accuracy_pooled_display} to{' '}
              {holdout.balanced_accuracy_holdout_display}. AUC is rank-based, so this is not a
              threshold problem: the ranking is not there.
            </p>
            <p>
              Each condition in PhysioNet 2016 was recorded on one or two setups only, so the
              recording source and the label never appear apart, and no method fitted to this
              corpus can separate them. Every pooled PhysioNet figure on this site is therefore a
              within-corpus cross-validation. No claim of generalization, deployment-readiness or
              field performance is made anywhere.
            </p>
            <Sources sources={holdout.sources} />
          </div>
        ) : (
          <EmptyState className="mt-4" title="Not generated" description={holdout.reason} />
        )}
      </section>

      <section>
        <SectionHeader eyebrow="PASCAL" title="The PASCAL tracks are small" level={2} />
        {pascal.available && pascal.tracks ? (
          <div className="mt-4 space-y-4 text-sm text-slate-700 dark:text-slate-300">
            {pascal.tracks.map((track) => (
              <div key={track.task}>
                <p>
                  {track.title}: {track.n_records_display} labelled recordings in total; the
                  smallest class, <span className="font-mono">{track.smallest_class}</span>, has{' '}
                  {track.smallest_n_display}.
                </p>
                <ul className="mt-1 flex flex-wrap gap-x-4 text-xs text-slate-600 dark:text-slate-400">
                  {track.classes.map((row) => (
                    <li key={row.class}>
                      <span className="font-mono">{row.class}</span> {row.n_records_display}{' '}
                      recordings / {row.n_subjects_display} subjects
                    </li>
                  ))}
                </ul>
              </div>
            ))}
            <p>
              At these sizes every model&apos;s interval overlaps every other&apos;s, so a ranking
              on PASCAL orders models that are not distinguishable. Below roughly a hundred
              training records a per-fold search fits noise. PASCAL A&apos;s{' '}
              <span className="font-mono">artifact</span> is a recording-quality label, so its
              four-class model is never a four-class cardiac classifier. PASCAL A and B are separate
              tasks and are never merged.
            </p>
            <Sources sources={pascal.sources} />
          </div>
        ) : (
          <EmptyState className="mt-4" title="Not generated" description={pascal.reason} />
        )}
      </section>

      <section>
        <SectionHeader eyebrow="CirCor" title="CirCor is the public subset only" level={2} />
        {circor.available ? (
          <div className="mt-4 space-y-3 text-sm text-slate-700 dark:text-slate-300">
            <p>
              The public CirCor DigiScope 2022 release is the Challenge&apos;s training portion:{' '}
              {circor.n_patients_display} patients and {circor.n_recordings_display} recordings.
              The Challenge&apos;s validation and test patients were never published, so no result
              here is comparable to a Challenge leaderboard score, and the audited count differs
              from the full cohort the source documents cite.
            </p>
            <p>
              {circor.unknown_murmur_patients_display} patients carry an Unknown murmur label. The
              three-class murmur task keeps them, matching the Challenge; a two-class version on
              the known patients is reported beside it. CirCor labels the patient, not the
              recording, so every recording-level score has a patient-level counterpart under a
              declared collapse rule.
            </p>
            <Sources sources={circor.sources} />
          </div>
        ) : (
          <EmptyState className="mt-4" title="Not generated" description={circor.reason} />
        )}
      </section>
    </div>
  );
}
