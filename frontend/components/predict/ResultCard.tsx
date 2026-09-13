'use client';

import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { SHOW_SCREENING_NOTICE } from '@/lib/flags';
import { SURFACE, TYPE_SCALE, seriesColor } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import type { GeneratedSample } from '@/lib/generated/types';
import type { PredictResult } from '@/lib/api';

/**
 * One prediction, rendered (T116.1, T116.2, T116.3).
 *
 * ## Every number here is a string the server already rounded
 *
 * `result.display.*` is formatted in Python by `tables.format_value`, the same
 * function that formatted every precomputed table on this site. This component
 * reads those strings. It does not round, does not compute a percentage and does
 * not decide what "confident" means — `low_confidence` arrives as a boolean from
 * the model layer, because a client that inferred it from a probability near the
 * middle would be a second implementation of a rule that already exists.
 *
 * ## The screening notice belongs to this component, behind one flag
 *
 * T116.2 asks for the screening notice on every result, so it is part of this
 * component rather than something a page remembers to add. A page that forgot it
 * would render a clean class name with no scope statement, which is the exact
 * shape of a diagnostic claim. It is hidden for the presentation by
 * `SHOW_SCREENING_NOTICE` (T127.2) and restored with it in T138.6.
 *
 * ## Low confidence is shown as loudly as the class
 *
 * When the top two probabilities are within the margin, the model has not
 * separated them, and the interface says that at the same size as the answer
 * rather than in a footnote.
 */

export function ProbabilityBars({
  result,
  classes,
}: {
  result: PredictResult;
  /** Declared class order for the task, so the bars do not reorder per result. */
  classes: string[];
}) {
  const ordered = classes.filter((name) => name in result.probabilities);
  const rest = Object.keys(result.probabilities).filter((name) => !ordered.includes(name));
  return (
    <ul className="mt-4 space-y-2">
      {[...ordered, ...rest].map((name, index) => {
        const value = result.probabilities[name];
        const width = typeof value === 'number' ? value * 100 : 0;
        const top = name === result.predicted_class;
        return (
          <li key={name}>
            <div className="flex items-baseline justify-between gap-3">
              <span className={cn('font-mono text-label-md uppercase', top ? 'text-accent-deep' : 'text-ink-2')}>
                {name}
              </span>
              <span className={cn('stat font-mono text-telemetry-sm', top ? 'text-ink' : 'text-ink-2')}>
                {result.display.probabilities[name] ?? 'n/a'}
                <span className={cn(SURFACE.muted, 'ml-2')}>
                  {result.display.probabilities_percent[name] ?? ''}
                </span>
              </span>
            </div>
            <div
              className="mt-1 h-1.5 w-full overflow-hidden rounded-full border border-line bg-sunken"
              role="img"
              aria-label={name + ' probability ' + (result.display.probabilities[name] ?? 'n/a')}
            >
              <div
                className="h-full rounded"
                style={{
                  width: width + '%',
                  backgroundColor: seriesColor(index),
                  opacity: top ? 1 : 0.55,
                }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export function ResultCard({
  result,
  classes,
  sample,
  className,
}: {
  result: PredictResult;
  classes: string[];
  /** Set when a built-in sample was scored, so its corpus label can be shown. */
  sample?: GeneratedSample | null;
  className?: string;
}) {
  const reference = sample?.reference ?? null;
  const trueLabel = sample?.labels?.[result.task] ?? null;

  // T121.5. Deliberately an early return rather than a banner above the usual
  // card: there is no class, no probability and no margin to show, and a
  // layout that left the empty shapes in place would read as a result of zero.
  if (result.scorable === false) {
    return (
      <section
        className={cn(
          'rounded-xl border-2 border-amber-400 p-4 dark:border-amber-600',
          'bg-amber-50 dark:bg-amber-950/40',
          className,
        )}
        aria-label="Prediction result"
      >
        <p className="font-mono text-label-sm uppercase text-amber-900 dark:text-amber-100">
          Screening indication
        </p>
        <p className={cn(TYPE_SCALE.h1, 'mt-1 text-amber-900 dark:text-amber-100')}>
          Not scored
        </p>
        <p
          role="alert"
          className={cn(TYPE_SCALE.body, 'mt-4 max-w-prose text-amber-900 dark:text-amber-100')}
        >
          {result.not_scorable_reason ??
            'This recording could not be scored, so no screening indication was produced.'}
        </p>
        <dl className={cn(TYPE_SCALE.caption, 'mt-4 grid gap-x-6 gap-y-1 sm:grid-cols-2')}>
          <div className="flex justify-between gap-3">
            <dt className={SURFACE.muted}>Features requested</dt>
            <dd className="stat font-mono text-ink">{result.display.n_features}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className={SURFACE.muted}>Features not computable</dt>
            <dd className="stat font-mono text-ink">{result.display.n_missing_features}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className={SURFACE.muted}>Duration</dt>
            <dd className="stat font-mono text-ink">{result.display.duration_seconds ?? 'n/a'}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className={SURFACE.muted}>Task</dt>
            <dd>{result.task}</dd>
          </div>
        </dl>
        {SHOW_SCREENING_NOTICE ? (
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-4 max-w-prose')}>
            {result.disclaimer}
          </p>
        ) : null}
      </section>
    );
  }

  return (
    <section
      className={cn(
        'overflow-hidden rounded-xl border border-line bg-panel shadow-panel',
        className,
      )}
      aria-label="Prediction result"
    >
      <div aria-hidden="true" className="h-0.5 w-full bg-accent" />
      <header className="flex flex-wrap items-center justify-between gap-3 p-4 pb-0">
        <div>
          <p className="label-micro">Screening indication</p>
          <p className={cn(TYPE_SCALE.h1, 'mt-1 text-ink')}>{result.predicted_class}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {result.low_confidence ? (
            <Badge tone="warn">Low confidence</Badge>
          ) : (
            <Badge tone="neutral">Margin {result.display.margin}</Badge>
          )}
          {trueLabel !== null ? (
            <Badge tone={trueLabel === result.predicted_class ? 'good' : 'danger'}>
              Corpus label: {trueLabel}
            </Badge>
          ) : null}
        </div>
      </header>

      {result.low_confidence ? (
        <p
          role="alert"
          className={cn(
            TYPE_SCALE.body,
            'mt-4 rounded border-2 border-amber-400 bg-amber-50 p-3 text-amber-900',
            'dark:border-amber-600 dark:bg-amber-950/40 dark:text-amber-100',
          )}
        >
          The gap between the top two classes is {result.display.margin}, below the{' '}
          {result.display.low_confidence_margin} margin. The model has not separated them,
          so this indication should not be read as a finding.{' '}
          {prediction.low_confidence.note}
        </p>
      ) : null}

      <ProbabilityBars result={result} classes={classes} />

      <dl className="mx-4 mt-4 grid gap-x-4 gap-y-1.5 border-t border-line pt-3 text-body-sm sm:grid-cols-2">
        <div className="flex justify-between gap-3">
          <dt className="label-micro">Confidence</dt>
          <dd className="stat font-mono text-ink">{result.display.confidence}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="label-micro">Operating point</dt>
          <dd className="stat font-mono text-ink">{result.display.operating_threshold ?? 'argmax'}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="label-micro">Duration</dt>
          <dd className="stat font-mono text-ink">{result.display.duration_seconds ?? 'n/a'}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="label-micro">Features used</dt>
          <dd className="stat font-mono text-ink">{result.display.n_features}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="label-micro">Features with no value</dt>
          <dd className="stat font-mono text-ink">{result.display.n_missing_features}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="label-micro">Total inference time</dt>
          <dd className="stat font-mono text-ink">{result.display.timings_seconds.total ?? 'n/a'}</dd>
        </div>
      </dl>

      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mx-4 mt-3')}>
        {result.operating_point_note}
      </p>

      {result.model.note ? (
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mx-4 mt-2')}>
          Model {result.model.model_id}: {result.model.note}
        </p>
      ) : null}

      {result.warnings.length > 0 ? (
        <ul
          className={cn(
            TYPE_SCALE.caption,
            'mx-4 mt-3 list-disc space-y-1 rounded-lg border border-amber-300 bg-amber-50 p-3 pl-7',
            'text-amber-900 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100',
          )}
        >
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}

      {reference !== null ? (
        <details className="mx-4 mt-3 rounded-lg border border-line bg-sunken p-3">
          <summary className="cursor-pointer font-mono text-label-md uppercase text-accent-strong marker:text-accent">
            What {reference.experiment} stored for this recording
          </summary>
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>
            {prediction.reference.note}
          </p>
          <ul className={cn(TYPE_SCALE.caption, 'mt-3 space-y-1')}>
            {reference.fold_labels.map((fold, index) => (
              <li key={fold} className="flex justify-between gap-3">
                <span className="font-mono">{fold}</span>
                <span className="stat font-mono">
                  {reference.probabilities_display[index]}{' '}
                  <span className={SURFACE.muted}>{reference.predicted_classes[index]}</span>
                </span>
              </li>
            ))}
            <li className="flex justify-between gap-3 border-t border-line pt-1 font-medium">
              <span>mean over {reference.n_repeats} repeats</span>
              <span className="stat font-mono">{reference.mean_probability_display}</span>
            </li>
          </ul>
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>
            {reference.folds_agree
              ? 'All repeats agreed on the class.'
              : 'The repeats did NOT all agree on the class: this recording sits near the boundary and its prediction changed with the training split.'}{' '}
            Model {reference.model_id}, positive class {reference.positive_class}.
          </p>
        </details>
      ) : null}

      {SHOW_SCREENING_NOTICE ? (
        <p
          className={cn(
            TYPE_SCALE.caption,
            'mt-4 border-t border-accent-line bg-accent-soft p-3 text-accent-deep',
          )}
        >
          {result.disclaimer}
        </p>
      ) : null}
    </section>
  );
}
