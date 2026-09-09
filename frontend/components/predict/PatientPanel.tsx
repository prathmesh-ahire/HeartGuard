'use client';

import { useCallback, useState } from 'react';

import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import { ApiError, predictPatient, type PatientResult } from '@/lib/api';

/**
 * One subject, four auscultation locations, three collapses (T116.4).
 *
 * ## Why this is not the same control as the single-recording panel
 *
 * CirCor labels the **subject**. A single recording carries the subject's label
 * because it was propagated to it, so a recording-level indication and a
 * patient-level one are different quantities that happen to share a vocabulary.
 * Showing them in one control would invite reading a recording-level number as
 * the subject's result.
 *
 * ## All three rules, never one
 *
 * `max`, `mean` and `any_present` are the rules the CirCor experiments used, and
 * they disagree in exactly the interesting case: one confident recording and
 * three quiet ones. `max` re-thresholds a pooled score, `any_present` unions
 * decisions already made. Displaying whichever one agrees with the label would
 * be choosing the rule after seeing the answer.
 *
 * Every number rendered is a `*_display` string the server rounded; the
 * collapse itself happens on the server for the same reason.
 */

const LOCATION_NAMES: Record<string, string> = {
  AV: 'Aortic valve',
  PV: 'Pulmonary valve',
  TV: 'Tricuspid valve',
  MV: 'Mitral valve',
  Phc: 'Other chest position',
};

export function PatientPanel({ task, className }: { task: string; className?: string }) {
  const group = prediction.patient_group;
  const [result, setResult] = useState<PatientResult | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const members = group.sample_ids
    .map((id) => prediction.samples.find((entry) => entry.sample_id === id))
    .filter((entry): entry is NonNullable<typeof entry> => entry !== undefined);

  const run = useCallback(async () => {
    setBusy(true);
    setFailure(null);
    setResult(null);
    try {
      setResult(await predictPatient(group.sample_ids, task));
    } catch (error) {
      setFailure(error instanceof ApiError ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [group.sample_ids, task]);

  return (
    <section className={className} aria-label="Patient-level analysis">
      <p className={cn(TYPE_SCALE.body, SURFACE.muted, 'max-w-prose')}>{group.note}</p>

      <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
        <table className="min-w-full text-left text-sm">
          <caption className={cn(TYPE_SCALE.caption, SURFACE.muted, 'p-3 text-left')}>
            Dataset sample {group.subject_id}: the four recordings this subject was screened
            with. Not a patient and not a case — de-identified paediatric screening data.
          </caption>
          <thead className="bg-slate-50 text-xs uppercase tracking-widest text-slate-500 dark:bg-slate-900/60">
            <tr>
              <th scope="col" className="px-3 py-2">Recording</th>
              <th scope="col" className="px-3 py-2">Location</th>
              <th scope="col" className="px-3 py-2">Duration</th>
              <th scope="col" className="px-3 py-2">Corpus label</th>
              <th scope="col" className="px-3 py-2">Indication</th>
            </tr>
          </thead>
          <tbody>
            {members.map((member) => {
              const scored = result?.recordings.find(
                (recording) => recording.source === member.sample_id,
              );
              const location = result?.locations[member.sample_id] ?? member.recording_location;
              return (
                <tr
                  key={member.sample_id}
                  className="border-t border-slate-100 dark:border-slate-800"
                >
                  <td className="px-3 py-1.5 font-mono">{member.record_uid}</td>
                  <td className="px-3 py-1.5">
                    {location ?? 'n/a'}
                    {location !== null && location !== undefined && location in LOCATION_NAMES ? (
                      <span className={cn(SURFACE.muted, 'ml-2')}>
                        {LOCATION_NAMES[location]}
                      </span>
                    ) : null}
                  </td>
                  <td className="px-3 py-1.5 tabular-nums">{member.duration_display}</td>
                  <td className="px-3 py-1.5">{member.labels[task] ?? 'n/a'}</td>
                  <td className="px-3 py-1.5">
                    {scored === undefined ? (
                      <span className={SURFACE.muted}>not scored</span>
                    ) : (
                      <>
                        {scored.predicted_class}{' '}
                        <span className="tabular-nums">
                          {scored.display.probabilities[result?.positive_class ?? ''] ?? ''}
                        </span>
                        {scored.low_confidence ? (
                          <Badge tone="warn" className="ml-2">
                            low confidence
                          </Badge>
                        ) : null}
                      </>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <button
        type="button"
        onClick={() => void run()}
        disabled={busy}
        className={cn(
          'mt-4 rounded px-4 py-2 text-sm font-semibold text-white',
          'bg-sky-700 hover:bg-sky-800 disabled:opacity-50 dark:bg-sky-600 dark:hover:bg-sky-500',
        )}
      >
        {busy ? 'Scoring all four recordings…' : 'Score this subject at every location'}
      </button>

      {busy ? <LoadingState className="mt-4" label="Scoring four recordings, one at a time" /> : null}

      {failure !== null ? (
        <ErrorState
          className="mt-4"
          title="No patient-level indication was produced"
          detail={failure}
          onRetry={() => void run()}
        />
      ) : null}

      {result !== null ? (
        <div className="mt-6">
          <h3 className={cn(TYPE_SCALE.h3)}>Patient-level collapse</h3>
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1 max-w-prose')}>
            All three declared rules, over {result.n_recordings} recordings. Where they
            disagree, that disagreement is the result — it is not a bug and one of them is
            not the right answer.
          </p>
          <ul className="mt-3 grid gap-3 sm:grid-cols-3">
            {result.rules.map((rule) => (
              <li
                key={rule.rule}
                className="rounded-lg border border-slate-200 p-4 dark:border-slate-800"
              >
                <p className="font-mono text-xs uppercase tracking-widest text-slate-500">
                  {rule.rule}
                </p>
                <p className={cn(TYPE_SCALE.h3, 'mt-1')}>{rule.predicted_class}</p>
                <p className="tabular-nums text-sm">{rule.score_display}</p>
                <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>{rule.note}</p>
              </li>
            ))}
          </ul>
          <p
            className={cn(
              TYPE_SCALE.caption,
              'mt-4 rounded border border-sky-300 bg-sky-50 p-3 text-sky-900',
              'dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-100',
            )}
          >
            {result.disclaimer}
          </p>
        </div>
      ) : null}

      {result === null && !busy && failure === null ? (
        <EmptyState
          className="mt-4"
          title="No patient-level indication yet"
          description="Nothing is collapsed until all four recordings have actually been scored."
        />
      ) : null}
    </section>
  );
}
