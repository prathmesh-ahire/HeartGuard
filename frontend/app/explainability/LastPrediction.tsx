'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import { cn } from '@/lib/cn';
import { EmptyState } from '@/components/ui/States';
import { LAST_PREDICTION_EVENT, recallPrediction, type StoredPrediction } from '@/lib/lastPrediction';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * The per-sample explanation of the last prediction made in this tab (T117.2).
 *
 * Everything shown is the API's own response, stored by the prediction page:
 * the terms, their order and their rounding are the server's. Nothing here
 * recomputes a contribution. Before the first prediction -- and in the static
 * HTML -- it says plainly that there is nothing to explain yet.
 */
export function LastPrediction() {
  const [stored, setStored] = useState<StoredPrediction | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    const read = (): void => {
      setStored(recallPrediction());
      setChecked(true);
    };
    read();
    window.addEventListener(LAST_PREDICTION_EVENT, read);
    return () => window.removeEventListener(LAST_PREDICTION_EVENT, read);
  }, []);

  if (!checked || stored === null) {
    return (
      <EmptyState
        title="No prediction has been made in this browser tab yet"
        description={
          <>
            Score a recording on the{' '}
            <Link href="/predict/binary/" className="underline">
              binary prediction page
            </Link>{' '}
            (or either of the other two) and its explanation appears here. Uploads are never
            kept on the server, so the explanation travels with the response in this tab only.
          </>
        }
      />
    );
  }

  const { result } = stored;
  const explanation = result.explanation ?? null;
  const largest = Math.max(
    ...(explanation?.rows ?? []).map((row) => Math.abs(row.contribution ?? 0)),
    0,
  );

  return (
    <div className="space-y-4">
      <dl className={cn(TYPE_SCALE.caption, 'grid gap-x-6 gap-y-1 sm:grid-cols-2')}>
        <div className="flex gap-2">
          <dt className={SURFACE.muted}>Recording</dt>
          <dd className="font-mono">{result.source}</dd>
        </div>
        <div className="flex gap-2">
          <dt className={SURFACE.muted}>Task · model</dt>
          <dd>
            {result.task} · {result.model.model_id ?? 'unknown'}
          </dd>
        </div>
        <div className="flex gap-2">
          <dt className={SURFACE.muted}>Screening indication</dt>
          <dd className="font-semibold">{result.predicted_class}</dd>
        </div>
        <div className="flex gap-2">
          <dt className={SURFACE.muted}>Confidence</dt>
          <dd className="tabular-nums">
            {result.display.confidence}
            {result.low_confidence ? ' (low confidence)' : ''}
          </dd>
        </div>
      </dl>

      {explanation === null ? (
        <EmptyState
          title="This response carries no explanation"
          description="It was produced by an inference API older than Phase 117, which did not return one. Score the recording again."
        />
      ) : !explanation.available ? (
        <EmptyState title="No per-recording explanation for this model" description={explanation.reason} />
      ) : (
        <div className="overflow-x-auto">
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mb-2')}>
            {explanation.method}, in {explanation.units}. Positive terms push toward the positive
            class. The {explanation.n_shown} largest terms by size are shown.
          </p>
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className={cn(TYPE_SCALE.caption, 'border-b border-slate-300 dark:border-slate-700')}>
                <th scope="col" className="px-2 py-1.5">Feature</th>
                <th scope="col" className="px-2 py-1.5">Family</th>
                <th scope="col" className="px-2 py-1.5">Value</th>
                <th scope="col" className="px-2 py-1.5">Scaled</th>
                <th scope="col" className="px-2 py-1.5">Coefficient</th>
                <th scope="col" className="px-2 py-1.5">Contribution</th>
              </tr>
            </thead>
            <tbody>
              {explanation.rows.map((row) => {
                const size =
                  largest > 0 && row.contribution !== null ? Math.abs(row.contribution) / largest : 0;
                const positive = (row.contribution ?? 0) > 0;
                return (
                  <tr key={row.feature} className="border-b border-slate-200 dark:border-slate-800">
                    <td className={cn(TYPE_SCALE.caption, 'px-2 py-1 font-mono')}>{row.feature}</td>
                    <td className={cn(TYPE_SCALE.caption, 'px-2 py-1')}>{row.family}</td>
                    <td className={cn(TYPE_SCALE.caption, 'px-2 py-1 tabular-nums')}>{row.value_display}</td>
                    <td className={cn(TYPE_SCALE.caption, 'px-2 py-1 tabular-nums')}>{row.scaled_display}</td>
                    <td className={cn(TYPE_SCALE.caption, 'px-2 py-1 tabular-nums')}>
                      {row.coefficient_display}
                    </td>
                    <td className={cn(TYPE_SCALE.caption, 'px-2 py-1')}>
                      <span className="flex items-center gap-2">
                        <span
                          aria-hidden="true"
                          className={cn('h-2.5 rounded-sm', positive ? 'bg-rose-500' : 'bg-sky-600')}
                          style={{ width: String(size * 6) + 'rem' }}
                        />
                        <span className="tabular-nums">{row.contribution_display}</span>
                        <span className={SURFACE.subtle}>{row.direction}</span>
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>
        An explanation of a prediction is not a clinical explanation. Stored in this tab at {stored.saved}.
      </p>
    </div>
  );
}
