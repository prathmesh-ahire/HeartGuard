import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { AnimatedCounter } from '@/components/ui/AnimatedCounter';

/**
 * One headline figure.
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
}: {
  label: string;
  display: string;
  value?: number | null;
  unit?: string;
  source?: string;
  hint?: ReactNode;
  className?: string;
  animate?: boolean;
}) {
  return (
    <div className={cn(SURFACE.card, 'p-4', className)}>
      <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>{label}</p>
      <p className="mt-2 flex items-baseline gap-1.5">
        {animate && value !== null ? (
          <AnimatedCounter value={value} display={display} />
        ) : (
          <span className={TYPE_SCALE.stat}>{display}</span>
        )}
        {unit ? <span className={cn(TYPE_SCALE.caption, SURFACE.subtle)}>{unit}</span> : null}
      </p>
      {hint ? <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>{hint}</p> : null}
      {source ? (
        <p className={cn(TYPE_SCALE.caption, SURFACE.subtle, 'mt-2 font-mono break-all')}>
          {source}
        </p>
      ) : null}
    </div>
  );
}
