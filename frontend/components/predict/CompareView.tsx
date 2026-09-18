'use client';

import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { EmptyState } from '@/components/ui/States';
import { WaveformPlayer } from '@/components/audio/WaveformPlayer';
import { TYPE_SCALE, seriesColor } from '@/lib/tokens';
import type { PredictResult } from '@/lib/api';

/**
 * Two recordings side by side: waveform, result and probabilities (T131.4).
 *
 * Built on a neutral `CompareItem` rather than on `PredictResult`, because
 * Phase 132 opens the same view from two History rows, which have results but
 * no audio (History never keeps a recording). A side without audio says so.
 *
 * ## Nothing is subtracted
 *
 * The view puts the two results next to each other and computes no difference
 * between them: a "difference in confidence" would be a number the browser made.
 * Every value shown is a display string from the server; numeric fields only
 * size the bars.
 *
 * ## Different tasks are shown, and called different
 *
 * Two results from different label spaces can be opened together, but their
 * categories are not comparable (research rule 4), so the view says that instead
 * of lining their bars up as if they were.
 */

export interface CompareItem {
  key: string;
  label: string;
  /** The recording to draw and play, or null when its audio was not kept. */
  source: File | string | null;
  task: string;
  taskTitle: string;
  classes: readonly string[];
  result: string;
  confidenceDisplay: string;
  lowConfidence: boolean;
  /** Bar geometry only. */
  probabilities: Record<string, number | null>;
  probabilitiesDisplay: Record<string, string>;
}

export function compareItemFromPrediction({
  key,
  label,
  source,
  result,
  taskTitle,
  classes,
}: {
  key: string;
  label: string;
  source: File | string | null;
  result: PredictResult;
  taskTitle: string;
  classes: readonly string[];
}): CompareItem {
  return {
    key,
    label,
    source,
    task: result.task,
    taskTitle,
    classes,
    result: result.predicted_class,
    confidenceDisplay: result.display.confidence,
    lowConfidence: result.low_confidence,
    probabilities: result.probabilities,
    probabilitiesDisplay: result.display.probabilities,
  };
}

function Side({ item }: { item: CompareItem }) {
  const names = [
    ...item.classes.filter((name) => name in item.probabilities),
    ...Object.keys(item.probabilities).filter((name) => !item.classes.includes(name)),
  ];
  return (
    <article aria-label={'Compared recording ' + item.label} className="min-w-0 space-y-4">
      {item.source !== null ? (
        <WaveformPlayer source={item.source} label={item.label} />
      ) : (
        <EmptyState
          icon="waveform"
          title="Audio not kept"
          description="History keeps each result, never the recording, so there is no waveform to show."
        />
      )}
      <div className="rounded-xl border border-line bg-panel p-4">
        <p className="label-micro">{item.taskTitle}</p>
        <p className={cn(TYPE_SCALE.h2, 'mt-1 capitalize text-ink')} data-testid="compare-result">
          {item.result}
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Badge tone="neutral">
            Confidence <span data-testid="compare-confidence">{item.confidenceDisplay}</span>
          </Badge>
          {item.lowConfidence ? <Badge tone="warn">Low confidence</Badge> : null}
        </div>
        <p className="label-micro mt-4">Probability of each category</p>
        <ul className="mt-2 space-y-2">
          {names.map((name, index) => {
            const value = item.probabilities[name];
            const top = name === item.result;
            return (
              <li key={name}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className={cn('text-label-md uppercase', top ? 'text-accent-deep' : 'text-ink-2')}>
                    {name}
                  </span>
                  <span className="stat font-mono text-telemetry-sm text-ink">
                    {item.probabilitiesDisplay[name] ?? 'n/a'}
                  </span>
                </div>
                <div className="mt-1 h-2 w-full overflow-hidden rounded-full border border-line bg-sunken">
                  <div
                    className={cn('h-full rounded-full', !top && 'opacity-60')}
                    style={{
                      width: (typeof value === 'number' ? value * 100 : 0) + '%',
                      backgroundColor: seriesColor(index),
                    }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </article>
  );
}

export function CompareView({
  items,
  onClose,
  className,
}: {
  items: readonly [CompareItem, CompareItem];
  onClose?: () => void;
  className?: string;
}) {
  const [left, right] = items;
  const sameTask = left.task === right.task;
  return (
    <section aria-label="Comparison" className={cn('rounded-xl border border-line bg-sunken p-4', className)}>
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className={cn(TYPE_SCALE.h3, 'text-ink')}>Side by side</h3>
          {sameTask ? (
            <Badge tone={left.result === right.result ? 'good' : 'warn'}>
              {left.result === right.result ? 'Same result' : 'Different results'}
            </Badge>
          ) : null}
        </div>
        {onClose !== undefined ? (
          <Button tone="ghost" size="sm" onClick={onClose}>
            Close comparison
          </Button>
        ) : null}
      </header>
      {!sameTask ? (
        <p role="note" className={cn(TYPE_SCALE.caption, 'mt-3 rounded border border-warn-line bg-warn-soft p-2 text-warn')}>
          These results come from different sets of categories ({left.taskTitle} and {right.taskTitle}). Read each
          against its own categories; they are not comparable with each other.
        </p>
      ) : null}
      <div className="mt-4 grid gap-6 md:grid-cols-2">
        <Side item={left} />
        <Side item={right} />
      </div>
    </section>
  );
}
