import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { AnimatedCounter } from '@/components/ui/AnimatedCounter';

/**
 * One headline figure, in the shape of an instrument readout: a status rule
 * across the top, the label in micro type, the value in tabular monospace with
 * its unit trailing, and the file it came from in the footer.
 *
 * `display` is required and is the only thing rendered at rest -- the string
 * Python formatted under the thesis rounding rules. `value` is optional and
 * only enables the count-up animation, which is why it is accepted for counts
 * and left off for metrics.
 *
 * `source` is not decoration. A figure on a dashboard with no visible
 * provenance is exactly what this project exists not to produce, so the tile
 * has a slot for the file it came from and the pages fill it.
 */
export function StatTile({
  label,
  display,
  value = null,
  unit,
  source,
  hint,
  className,
  animate = false,
  /** Draws the top rule in the accent rather than the neutral hairline. */
  marked = false,
}: {
  label: string;
  display: string;
  value?: number | null;
  unit?: string;
  source?: string;
  hint?: ReactNode;
  className?: string;
  animate?: boolean;
  marked?: boolean;
}) {
  return (
    <div className={cn(SURFACE.card, 'overflow-hidden', className)}>
      <div
        aria-hidden="true"
        className={cn('h-0.5 w-full', marked ? 'bg-accent' : 'bg-line')}
      />
      <div className="p-4">
        <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>{label}</p>
        <p className="mt-2 flex items-baseline gap-1.5">
          {animate && value !== null ? (
            <AnimatedCounter value={value} display={display} />
          ) : (
            <span className={cn(TYPE_SCALE.stat, 'text-ink')}>{display}</span>
          )}
          {unit ? (
            <span className={cn('font-mono text-label-md uppercase', SURFACE.subtle)}>{unit}</span>
          ) : null}
        </p>
        {hint ? <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>{hint}</p> : null}
        {source ? (
          <p
            className={cn(
              TYPE_SCALE.caption,
              SURFACE.subtle,
              'mt-3 break-all border-t border-line pt-2 font-mono',
            )}
          >
            {source}
          </p>
        ) : null}
      </div>
    </div>
  );
}
