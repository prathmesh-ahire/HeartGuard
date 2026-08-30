import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

export function SectionHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
  level = 2,
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
  level?: 1 | 2 | 3;
}) {
  const Heading = (level === 1 ? 'h1' : level === 2 ? 'h2' : 'h3') as 'h1' | 'h2' | 'h3';
  const size = level === 1 ? TYPE_SCALE.h1 : level === 2 ? TYPE_SCALE.h2 : TYPE_SCALE.h3;

  return (
    <div className={cn('flex flex-wrap items-start justify-between gap-4', className)}>
      <div className="max-w-3xl">
        {eyebrow ? (
          <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>{eyebrow}</p>
        ) : null}
        <Heading className={cn(size, eyebrow && 'mt-1')}>{title}</Heading>
        {description ? (
          <div className={cn(TYPE_SCALE.body, SURFACE.muted, 'mt-2')}>{description}</div>
        ) : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}
