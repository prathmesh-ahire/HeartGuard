'use client';

import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
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
 * ## The screening banner is not conditional
 *
 * T116.2 asks for the screening notice on every result, so it is part of this
 * component rather than something a page remembers to add. A page that forgot it
 * would render a clean class name with no scope statement, which is the exact
 * shape of a diagnostic claim.
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
              <span className={cn(TYPE_SCALE.body, top && 'font-semibold')}>{name}</span>
              <span className={cn(TYPE_SCALE.caption, 'tabular-nums', top && 'font-semibold')}>
                {result.display.probabilities[name] ?? 'n/a'}
                <span className={cn(SURFACE.muted, 'ml-2')}>
                  {result.display.probabilities_percent[name] ?? ''}
                </span>
              </span>
            </div>
            <div
              className="mt-1 h-2 w-full overflow-hidden rounded bg-slate-200 dark:bg-slate-800"
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

  return (
    <section
      className={cn('rounded-lg border border-slate-200 p-5 dark:border-slate-800', className)}
      aria-label="Prediction result"
    >
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>Screening indication</p>
          <p className={cn(TYPE_SCALE.h2, 'mt-0.5')}>{result.predicted_class}</p>
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

      <dl className="mt-5 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
        <div className="flex justify-between gap-3">
          <dt className={SURFACE.muted}>Confidence</dt>
          <dd className="tabular-nums">{result.display.confidence}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className={SURFACE.muted}>Operating point</dt>
          <dd className="tabular-nums">{result.display.operating_threshold ?? 'argmax'}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className={SURFACE.muted}>Duration</dt>
          <dd className="tabular-nums">{result.display.duration_seconds ?? 'n/a'}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className={SURFACE.muted}>Features used</dt>
          <dd className="tabular-nums">{result.display.n_features}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className={SURFACE.muted}>Features with no value</dt>
          <dd className="tabular-nums">{result.display.n_missing_features}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className={SURFACE.muted}>Total inference time</dt>
          <dd className="tabular-nums">{result.display.timings_seconds.total ?? 'n/a'}</dd>
        </div>
      </dl>

      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-4')}>
        {result.operating_point_note}
      </p>

      {result.model.note ? (
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>
          Model {result.model.model_id}: {result.model.note}
        </p>
      ) : null}

      {result.warnings.length > 0 ? (
        <ul
          className={cn(
            TYPE_SCALE.caption,
            'mt-4 list-disc space-y-1 rounded border border-amber-300 bg-amber-50 p-3 pl-7',
            'text-amber-900 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100',
          )}
        >
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}

      {reference !== null ? (
        <details className="mt-5 rounded border border-slate-200 p-3 dark:border-slate-800">
          <summary className={cn(TYPE_SCALE.body, 'cursor-pointer font-medium')}>
            What {reference.experiment} stored for this recording
          </summary>
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>
            {prediction.reference.note}
          </p>
          <ul className={cn(TYPE_SCALE.caption, 'mt-3 space-y-1')}>
            {reference.fold_labels.map((fold, index) => (
              <li key={fold} className="flex justify-between gap-3">
                <span className="font-mono">{fold}</span>
                <span className="tabular-nums">
                  {reference.probabilities_display[index]}{' '}
                  <span className={SURFACE.muted}>{reference.predicted_classes[index]}</span>
                </span>
              </li>
            ))}
            <li className="flex justify-between gap-3 border-t border-slate-200 pt-1 font-medium dark:border-slate-800">
              <span>mean over {reference.n_repeats} repeats</span>
              <span className="tabular-nums">{reference.mean_probability_display}</span>
            </li>
          </ul>
          <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>
            {reference.folds_agree
              ? 'All repeats agreed on the class.'
              : 'The repeats did NOT all agree on the class: this recording sits near the boundary and its prediction changed with the training split.'}{' '}
            Source: <span className="font-mono">{reference.source}</span>, model{' '}
            {reference.model_id}, positive class {reference.positive_class}.
          </p>
        </details>
      ) : null}

      <p
        className={cn(
          TYPE_SCALE.caption,
          'mt-5 rounded border border-sky-300 bg-sky-50 p-3 text-sky-900',
          'dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-100',
        )}
      >
        {result.disclaimer}
      </p>
    </section>
  );
}
