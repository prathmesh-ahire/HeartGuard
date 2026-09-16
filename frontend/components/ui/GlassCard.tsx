import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { SURFACE } from '@/lib/tokens';

/**
 * A surface.
 *
 * `glass` is translucent and belongs over the 3D hero; `solid` is the default
 * everywhere else, because a chart read through a blur is a chart misread.
 *
 * The optional header is the workstation panel head: an eyebrow in instrument
 * type on the left, status or controls on the right, a hairline between it and
 * the body. It is a slot rather than a separate component so a panel cannot
 * end up with a header that scrolls away from its own body.
 */
export function GlassCard({
  children,
  className,
  bodyClassName,
  variant = 'solid',
  as: Component = 'div',
  eyebrow,
  title,
  meta,
  /** A 2px accent rule across the top of the panel. Marks the active unit. */
  marked = false,
  /** Removes body padding, for a panel whose body is a table or a canvas. */
  flush = false,
  /** For `as="section"`/`"aside"`: an accessible name, since `title` may not be plain text. */
  ariaLabel,
}: {
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  variant?: 'solid' | 'glass' | 'sunken';
  as?: 'div' | 'section' | 'article' | 'aside';
  eyebrow?: string;
  title?: ReactNode;
  meta?: ReactNode;
  marked?: boolean;
  flush?: boolean;
  ariaLabel?: string;
}) {
  const surface =
    variant === 'glass' ? SURFACE.glass : variant === 'sunken' ? SURFACE.sunken : SURFACE.card;
  const hasHeader = eyebrow !== undefined || title !== undefined || meta !== undefined;

  return (
    <Component aria-label={ariaLabel} className={cn(surface, 'overflow-hidden', className)}>
      {marked ? <div aria-hidden="true" className="h-0.5 w-full bg-accent" /> : null}

      {hasHeader ? (
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
          <div className="min-w-0">
            {eyebrow ? <p className="label-micro">{eyebrow}</p> : null}
            {title ? (
              <div className={cn('text-headline-sm text-ink', eyebrow && 'mt-0.5')}>{title}</div>
            ) : null}
          </div>
          {meta ? (
            <div className="flex shrink-0 items-center gap-2 font-mono text-label-sm uppercase text-ink-3">
              {meta}
            </div>
          ) : null}
        </div>
      ) : null}

      <div className={cn(flush ? '' : 'p-4', bodyClassName)}>{children}</div>
    </Component>
  );
}
