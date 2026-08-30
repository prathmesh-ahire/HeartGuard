'use client';

import { useEffect, useState } from 'react';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { ensemble } from '@/lib/generated';
import { EmptyState } from '@/components/ui/States';
import { SURFACE, TYPE_SCALE, needsOutlineOn, seriesColor } from '@/lib/tokens';

/**
 * SVM, Random Forest and Gradient Boosting fusing into the weighted vote
 * (T112.4).
 *
 * ## The honest version of this visualization
 *
 * The obvious way to draw a weighted vote is three bars of visibly different
 * lengths flowing into one. On this corpus that picture would be a lie:
 * **21 of the 25 outer folds chose weights identical to equal weighting**, and
 * the mean weights differ from 1/3 in the third decimal place. So the component
 * draws the equal-weight line across the bars, states the fold count in words,
 * and renders `ensemble.interpretation` beside the chart rather than leaving a
 * viewer to infer a large reweighting from bars that were scaled to look
 * interesting.
 *
 * ## Where the numbers come from
 *
 * `generated/ensemble.json`, read from SO-05's `final_weights.json` and
 * formatted in Python. `weight` positions a bar; `weight_display` is the only
 * thing rendered as text. The client neither computes nor rounds a number.
 *
 * The probabilities in the illustration are **not** model outputs -- they are
 * three fixed demonstration values, labelled as such in the UI, because no
 * per-record prediction is exported to this page. Their vote is computed in
 * Python and arrives here as a formatted string: this component does no
 * arithmetic on any number it displays.
 */

export function EnsembleVote({ className }: { className?: string }) {
  const reduced = useReducedMotion();
  const [flow, setFlow] = useState(reduced ? 1 : 0);

  useEffect(() => {
    if (reduced) {
      setFlow(1);
      return;
    }
    let frame = 0;
    const started = performance.now();
    const step = (now: number): void => {
      const t = Math.min(1, (now - started) / 900);
      setFlow(t);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [reduced]);

  if (!ensemble.available) {
    return (
      <EmptyState
        className={className}
        title="The searched weights have not been produced yet"
        description={ensemble.reason ?? 'SO-05 has not been run.'}
      />
    );
  }

  const members = ensemble.members;
  const equal = ensemble.equal_weight ?? 1 / Math.max(members.length, 1);
  // Bar geometry only. `weight` never reaches the page as text.
  const maxWeight = Math.max(...members.map((member) => member.weight), equal);
  const demo = ensemble.demonstration;

  return (
    <div className={cn(SURFACE.card, 'p-5', className)}>
      <h3 className={cn(TYPE_SCALE.h3)}>Weighted soft vote</h3>
      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>
        {ensemble.experiment} · {ensemble.objective} · {ensemble.n_folds} outer folds · seed{' '}
        {ensemble.seed}
      </p>

      <ul className="mt-5 space-y-4">
        {members.map((member, index) => {
          const fill = seriesColor(index);
          const outlined = needsOutlineOn(index, 'light');
          const width = (member.weight / maxWeight) * 100 * flow;
          return (
            <li key={member.model_id}>
              <div className="flex items-baseline justify-between gap-3">
                <span className={cn(TYPE_SCALE.body, 'font-medium')}>{member.name}</span>
                <span className={cn(TYPE_SCALE.caption, 'tabular-nums', SURFACE.muted)}>
                  weight {member.weight_display} ± {member.weight_std_display}
                </span>
              </div>
              <div
                className={cn('relative mt-1.5 h-4 w-full rounded', SURFACE.sunken)}
                role="img"
                aria-label={
                  member.name + ' carries weight ' + member.weight_display + ' in the vote'
                }
              >
                <div
                  className={cn('h-full rounded transition-none', outlined ? 'ring-1 ring-inset ring-slate-900/50 dark:ring-white/60' : '')}
                  style={{ width: width + '%', backgroundColor: fill }}
                />
                {/* The equal-weight line: the reference the search barely beat. */}
                <span
                  aria-hidden="true"
                  className="absolute inset-y-0 w-px bg-slate-900/60 dark:bg-white/60"
                  style={{ left: (equal / maxWeight) * 100 + '%' }}
                />
              </div>
            </li>
          );
        })}
      </ul>

      <p className={cn(TYPE_SCALE.micro, SURFACE.subtle, 'mt-3')}>
        The vertical line marks equal weighting ({ensemble.equal_weight_display}).{' '}
        {ensemble.folds_identical_display} outer folds chose weights identical to it.
      </p>

      {demo === null ? null : (
        <div className={cn(SURFACE.sunken, 'mt-5 p-4')}>
          <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>Illustration, not a prediction</p>
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>{demo.note}</p>
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
            {demo.inputs.map((input, index) => (
              <span key={input.model_id} className={cn(TYPE_SCALE.caption, 'tabular-nums')}>
                <span
                  aria-hidden="true"
                  className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
                  style={{ backgroundColor: seriesColor(index) }}
                />
                {input.short_name} {input.probability_display}
              </span>
            ))}
            <span
              aria-hidden="true"
              className={cn(TYPE_SCALE.caption, SURFACE.subtle, 'transition-opacity')}
              style={{ opacity: flow }}
            >
              →
            </span>
            <span
              className={cn(TYPE_SCALE.body, 'font-semibold tabular-nums transition-opacity')}
              style={{ opacity: flow }}
            >
              vote {demo.vote_display}
            </span>
          </div>
        </div>
      )}

      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-4')}>{ensemble.interpretation}</p>
      <p className={cn(TYPE_SCALE.micro, SURFACE.subtle, 'mt-2 font-mono')}>
        source {ensemble.source}
      </p>
    </div>
  );
}
