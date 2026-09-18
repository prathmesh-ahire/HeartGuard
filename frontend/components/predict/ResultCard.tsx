'use client';

import { useState } from 'react';

import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button, ButtonLink } from '@/components/ui/Button';
import { Stagger, StaggerItem } from '@/components/motion/Reveal';
import { ConfidenceGauge } from '@/components/predict/ConfidenceGauge';
import { SHOW_SCREENING_NOTICE } from '@/lib/flags';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import type { GeneratedSample } from '@/lib/generated/types';
import type { PredictResult } from '@/lib/api';

/**
 * One prediction, rendered (T116.1-T116.3, rebuilt in T130.5 and T130.6).
 *
 * ## Every number here is a string the server already rounded
 *
 * `result.display.*` is formatted in Python by `tables.format_value`, the same
 * function that formatted every precomputed table on this site. This component
 * reads those strings. It does not round, does not compute a percentage and does
 * not decide what "confident" means — `low_confidence` arrives as a boolean from
 * the model layer. Numeric fields only size the gauge arc and the bars.
 *
 * ## The result says where it went
 *
 * T130.6: a successful result is saved to History by the API, which answers
 * with the row's id, so the card links to that row. When it was not saved --
 * an unscorable recording, or a History store that failed -- the card says so
 * with the server's note rather than offering a link to nothing.
 *
 * ## The screening notice belongs to this component, behind one flag
 *
 * Hidden for the presentation by `SHOW_SCREENING_NOTICE` (T127.2) and restored
 * with it in T138.6.
 *
 * ## Low confidence is shown as loudly as the class
 *
 * When the top two probabilities are within the margin, the model has not
 * separated them, and the card says that directly under the answer.
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
    <ul className="mt-3 space-y-2.5">
      {[...ordered, ...rest].map((name) => {
        const value = result.probabilities[name];
        const width = typeof value === 'number' ? value * 100 : 0;
        const top = name === result.predicted_class;
        return (
          <li key={name}>
            <div className="flex items-baseline justify-between gap-3">
              <span className={cn('text-label-md uppercase', top ? 'text-accent-deep' : 'text-ink-2')}>
                {name}
              </span>
              <span className={cn('stat font-mono text-telemetry-sm', top ? 'text-ink' : 'text-ink-2')}>
                {result.display.probabilities[name] ?? 'n/a'}
                <span className={cn(SURFACE.muted, 'ml-2')}>
                  {result.display.probabilities_percent[name] ?? ''}
                </span>
              </span>
            </div>
            {/* T141.4: two colours across the whole list, not one per class --
                the predicted class in the accent, every other category in one
                shared neutral, so the eye reads "this one" against "the rest"
                rather than a rainbow with no legend. */}
            <div
              className="mt-1 h-2 w-full overflow-hidden rounded-full border border-line bg-sunken"
              role="img"
              aria-label={name + ' probability ' + (result.display.probabilities[name] ?? 'n/a')}
            >
              <div
                className={cn('h-full rounded-full', top ? 'bg-accent' : 'bg-ink-3')}
                style={{ width: width + '%' }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

/** T130.6: where the result went, and its report. */
function ResultActions({
  result,
  onDownloadReport,
}: {
  result: PredictResult;
  onDownloadReport?: () => Promise<void>;
}) {
  const [report, setReport] = useState<'idle' | 'working' | 'failed'>('idle');
  const [reportError, setReportError] = useState('');
  const saved = result.history_saved === true && typeof result.history_id === 'string';

  return (
    <div className="border-t border-line p-5">
      <div className="flex flex-wrap items-center gap-2">
        {saved ? (
          <>
            <Badge tone="good" dot>
              Saved to History
            </Badge>
            <ButtonLink
              href={'/history/?record=' + encodeURIComponent(result.history_id as string)}
              icon="history"
            >
              View in History
            </ButtonLink>
          </>
        ) : (
          <p role="status" className={cn(TYPE_SCALE.caption, 'text-warn')}>
            {result.history_note ?? 'This result was not saved to History.'}
          </p>
        )}
        {onDownloadReport !== undefined ? (
          <Button
            icon="reports"
            disabled={report === 'working'}
            onClick={() => {
              setReport('working');
              onDownloadReport()
                .then(() => setReport('idle'))
                .catch((error: unknown) => {
                  setReportError(error instanceof Error ? error.message : String(error));
                  setReport('failed');
                });
            }}
          >
            {report === 'working' ? 'Preparing report…' : 'Download report'}
          </Button>
        ) : null}
      </div>
      {report === 'failed' ? (
        <p
          role="alert"
          className={cn(TYPE_SCALE.caption, 'mt-3 rounded border border-danger-line bg-danger-soft p-2 text-danger')}
        >
          The report was not produced. {reportError}
        </p>
      ) : null}
    </div>
  );
}

export function ResultCard({
  result,
  classes,
  taskTitle,
  sample,
  onDownloadReport,
  className,
}: {
  result: PredictResult;
  classes: string[];
  /** The task's title from the generated payload. */
  taskTitle?: string;
  /** Set when a built-in sample was scored, so its corpus label can be shown. */
  sample?: GeneratedSample | null;
  /** Offered only where the page can rebuild the same recording for the report. */
  onDownloadReport?: () => Promise<void>;
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
        className={cn('overflow-hidden rounded-xl border-2 border-warn-line bg-warn-soft', className)}
        aria-label="Prediction result"
      >
        <div className="p-5">
          <p className="text-label-sm uppercase text-warn">{taskTitle ?? result.task}</p>
          <p className={cn(TYPE_SCALE.h1, 'mt-1 text-warn')}>Not scored</p>
          <p role="alert" className={cn(TYPE_SCALE.body, 'mt-4 max-w-prose text-warn')}>
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
          </dl>
          {SHOW_SCREENING_NOTICE ? (
            <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-4 max-w-prose')}>{result.disclaimer}</p>
          ) : null}
        </div>
        <ResultActions result={result} />
      </section>
    );
  }

  return (
    <section
      className={cn('overflow-hidden rounded-xl border border-line bg-panel shadow-panel', className)}
      aria-label="Prediction result"
    >
      <div aria-hidden="true" className={cn('h-1 w-full', result.low_confidence ? 'bg-warn' : 'bg-accent')} />
      <Stagger>
        <StaggerItem>
          <header className="flex flex-wrap items-center gap-5 p-5">
            <ConfidenceGauge
              value={result.confidence}
              display={result.display.confidence}
              low={result.low_confidence}
            />
            <div className="min-w-0 flex-1">
              <p className="label-micro">{taskTitle ?? result.task}</p>
              <p className={cn(TYPE_SCALE.h1, 'mt-1 capitalize text-ink')} data-testid="result-class">
                {result.predicted_class}
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                {result.low_confidence ? (
                  <Badge tone="warn">Low confidence</Badge>
                ) : (
                  <Badge tone="neutral">Margin {result.display.margin}</Badge>
                )}
                {trueLabel !== null ? (
                  <Badge tone={trueLabel === result.predicted_class ? 'good' : 'danger'}>
                    Dataset label: {trueLabel}
                  </Badge>
                ) : null}
              </div>
            </div>
          </header>
        </StaggerItem>

        {result.low_confidence ? (
          <StaggerItem>
            <p
              role="alert"
              className={cn(TYPE_SCALE.body, 'mx-5 rounded-lg border-2 border-warn-line bg-warn-soft p-3 text-warn')}
            >
              The top two results are only {result.display.margin} apart, less than the{' '}
              {result.display.low_confidence_margin} the model needs to tell them apart. Treat this
              result as uncertain. {prediction.low_confidence.note}
            </p>
          </StaggerItem>
        ) : null}

        <StaggerItem className="px-5 pt-4">
          <p className="label-micro">Probability of each category</p>
          <ProbabilityBars result={result} classes={classes} />
        </StaggerItem>

        {result.explanation?.available === true && result.explanation.rows.length > 0 ? (
          <StaggerItem className="px-5 pt-4">
            <p className="label-micro">What drove this result</p>
            <ul className="mt-2 space-y-1.5">
              {result.explanation.rows.slice(0, 3).map((row) => (
                <li
                  key={row.feature}
                  className={cn(TYPE_SCALE.caption, 'flex items-center justify-between gap-3')}
                >
                  <span className="truncate font-mono text-ink-2">{row.feature}</span>
                  <span className={cn(SURFACE.muted, 'shrink-0')}>{row.direction}</span>
                </li>
              ))}
            </ul>
            <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2 max-w-prose')}>
              The largest factors the model weighed for this recording, from its{' '}
              {result.explanation.method}. This says what the model responded to — it is not a
              diagnosis and not a medical explanation.
            </p>
          </StaggerItem>
        ) : null}

        {result.warnings.length > 0 ? (
          <StaggerItem>
            <ul
              className={cn(
                TYPE_SCALE.caption,
                'mx-5 mt-4 list-disc space-y-1 rounded-lg border border-warn-line bg-warn-soft p-3 pl-7 text-warn',
              )}
            >
              {result.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </StaggerItem>
        ) : null}

        <StaggerItem className="px-5 pb-5">
          <details className="mt-4 rounded-lg border border-line bg-sunken p-3">
            <summary className="cursor-pointer text-label-md uppercase text-accent-strong marker:text-accent">
              Details
            </summary>
            <dl className="mt-3 grid gap-x-4 gap-y-1.5 text-body-sm sm:grid-cols-2">
              <div className="flex justify-between gap-3">
                <dt className="label-micro">Model</dt>
                <dd className="stat truncate font-mono text-ink">
                  {result.model.estimator_class ?? result.model.model_id ?? 'n/a'}
                </dd>
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
                <dt className="label-micro">Analysis time</dt>
                <dd className="stat font-mono text-ink">{result.display.timings_seconds.total ?? 'n/a'}</dd>
              </div>
            </dl>
            <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-3')}>{result.operating_point_note}</p>
            {result.model.note ? (
              <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>{result.model.note}</p>
            ) : null}
          </details>

          {reference !== null ? (
            <details className="mt-3 rounded-lg border border-line bg-sunken p-3">
              <summary className="cursor-pointer text-label-md uppercase text-accent-strong marker:text-accent">
                What cross-validation recorded for this sample
              </summary>
              <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>{prediction.reference.note}</p>
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
                  ? 'All repeats agreed on the category.'
                  : 'The repeats did NOT all agree: this recording sits near the boundary and its result changed with the training split.'}
              </p>
            </details>
          ) : null}
        </StaggerItem>

        <StaggerItem>
          <ResultActions result={result} onDownloadReport={onDownloadReport} />
        </StaggerItem>

        {SHOW_SCREENING_NOTICE ? (
          <p className={cn(TYPE_SCALE.caption, 'border-t border-accent-line bg-accent-soft p-3 text-accent-deep')}>
            {result.disclaimer}
          </p>
        ) : null}
      </Stagger>
    </section>
  );
}
