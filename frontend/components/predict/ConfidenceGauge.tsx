'use client';

import { useEffect, useState } from 'react';

import { cn } from '@/lib/cn';

/**
 * The confidence ring on a result (T130.5).
 *
 * The number in the middle is `display` -- the server's string. `value` only
 * decides how far the arc is drawn, which is geometry, the same rule the
 * probability bars follow: numeric fields position marks, display strings are
 * text. The arc grows from empty once on mount; under reduced motion the
 * transition is off and it simply appears at its length.
 */

const RADIUS = 42;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function ConfidenceGauge({
  value,
  display,
  low,
  className,
}: {
  value: number | null;
  display: string;
  /** Drawn in the warning colour when the model has not separated the top two classes. */
  low: boolean;
  className?: string;
}) {
  const [drawn, setDrawn] = useState(0);

  useEffect(() => {
    const frame = requestAnimationFrame(() => setDrawn(value ?? 0));
    return () => cancelAnimationFrame(frame);
  }, [value]);

  const filled = Math.max(0, Math.min(1, drawn));

  return (
    <figure
      role="img"
      aria-label={'Confidence ' + display}
      className={cn('relative h-28 w-28 shrink-0', className)}
    >
      <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90" aria-hidden="true">
        <circle cx="50" cy="50" r={RADIUS} fill="none" strokeWidth="8" className="stroke-sunken" />
        <circle
          cx="50"
          cy="50"
          r={RADIUS}
          fill="none"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={CIRCUMFERENCE * (1 - filled)}
          className={cn(
            low ? 'stroke-warn' : 'stroke-accent',
            'motion-safe:transition-[stroke-dashoffset] motion-safe:duration-700 motion-safe:ease-out',
          )}
        />
      </svg>
      <span className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="stat font-mono text-headline-lg text-ink">{display}</span>
        <span className="label-micro">confidence</span>
      </span>
    </figure>
  );
}
