import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { STATUS, type StatusTone } from '@/lib/tokens';

export function Badge({
  children,
  tone = 'neutral',
  className,
}: {
  children: ReactNode;
  tone?: StatusTone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium',
        STATUS[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
