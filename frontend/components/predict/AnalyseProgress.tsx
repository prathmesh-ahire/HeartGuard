'use client';

import { useEffect, useState } from 'react';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * The busy state for one analysis run (T141.3), replacing a plain spinner.
 *
 * `POST /predict` answers once, at the end -- there is no server-sent progress
 * to render honestly. What IS real is the pipeline's own stage order, which
 * `src/inference/predictor.py` already names and times (`validate`,
 * `load_model`, `preprocess`, `extract`, `predict`, returned as
 * `timings_seconds` keys once a result lands). This walks those same five
 * stages, in that order, on a fixed client-side clock: the labels are real
 * pipeline stages, but the pace is a guess and the bar's width is never
 * presented as a duration or a percentage the pipeline itself reported --
 * nothing here is a metric, it is a loading affordance, same as the
 * indeterminate sweep `FileUpload` already uses for its own upload phase.
 */

const STAGES = [
  'Validating the recording',
  'Loading the deployed model',
  'Cleaning and resampling the signal',
  'Extracting acoustic features',
  'Scoring with the model',
] as const;

const STEP_MS = 650;
// Held below 100 for as long as a stage clock is running: only the real
// response arriving (this component unmounting) means the work is actually
// done, and the bar must never claim that before it is true.
const HELD_MAX_PERCENT = 96;

export function AnalyseProgress({ className }: { className?: string }) {
  const reduced = useReducedMotion();
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (reduced) return;
    const id = window.setInterval(() => {
      setStep((current) => (current + 1 < STAGES.length ? current + 1 : current));
    }, STEP_MS);
    return () => window.clearInterval(id);
  }, [reduced]);

  const percent = reduced
    ? HELD_MAX_PERCENT
    : Math.round(((step + 1) / STAGES.length) * HELD_MAX_PERCENT);

  return (
    <div
      role="status"
      aria-live="polite"
      aria-busy="true"
      className={cn(SURFACE.sunken, 'p-5', className)}
    >
      <span className="sr-only">Analysing the recording</span>
      <p aria-hidden="true" className={cn(TYPE_SCALE.body, 'font-medium text-ink')}>
        Analysing the recording…
      </p>
      <div
        aria-hidden="true"
        className="mt-4 h-1.5 w-full overflow-hidden rounded bg-panel"
      >
        <div
          className="h-full rounded bg-accent transition-[width] ease-out"
          style={{ width: percent + '%', transitionDuration: reduced ? '0ms' : '400ms' }}
        />
      </div>
      <ul aria-hidden="true" className="mt-4 space-y-1.5">
        {STAGES.map((label, index) => {
          const done = reduced || index < step;
          const current = !reduced && index === step;
          return (
            <li
              key={label}
              className={cn(
                TYPE_SCALE.caption,
                'flex items-center gap-2',
                done ? 'text-ink-2' : current ? 'font-medium text-accent-deep' : SURFACE.subtle,
              )}
            >
              <span>{done ? '✓' : current ? '…' : '·'}</span>
              {label}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
