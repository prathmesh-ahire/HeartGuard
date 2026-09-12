import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { STATUS, type StatusTone } from '@/lib/tokens';

export function Badge({
  children,
  tone = 'neutral',
  className,
  /** A leading status dot. `pulse` marks a live, changing state. */
  dot = false,
  pulse = false,
}: {
  children: ReactNode;
  tone?: StatusTone;
  className?: string;
  dot?: boolean;
  pulse?: boolean;
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5',
        'font-mono text-label-sm uppercase',
        STATUS[tone],
        className,
      )}
    >
      {dot || pulse ? (
        <span
          aria-hidden="true"
          className={cn(
            'h-1.5 w-1.5 shrink-0 rounded-full bg-current',
            pulse && 'animate-pulse-subtle',
          )}
        />
      ) : null}
      {children}
    </span>
  );
}
