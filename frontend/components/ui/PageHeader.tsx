import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';

/**
 * The page title area (T128.2, T128.4).
 *
 * Every page opens the same way: an optional eyebrow, what this page is for in
 * one title and one lede, and the page's **one** primary action. The layout
 * rule is one primary action per page, so `primaryAction` takes a single
 * control and anything else goes in `actions`, which renders before it as
 * secondary controls. The `note` slot is where a population caveat or a scope
 * limit goes.
 */
export function PageHeader({
  title,
  eyebrow,
  lede,
  note,
  primaryAction,
  actions,
  className,
  level = 1,
}: {
  title: string;
  eyebrow?: string;
  lede?: ReactNode;
  note?: ReactNode;
  /** The one thing this page is for. A single control. */
  primaryAction?: ReactNode;
  /** Secondary controls, rendered before the primary action. */
  actions?: ReactNode;
  className?: string;
  /** 2 when the header opens a section inside a page that already has an h1. */
  level?: 1 | 2;
}) {
  const Heading = level === 1 ? 'h1' : 'h2';
  const hasActions = primaryAction !== undefined || actions !== undefined;

  return (
    <header
      className={cn(
        'flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between',
        className,
      )}
    >
      <div className="max-w-reading">
        {eyebrow ? <p className="font-mono text-label-md uppercase text-accent">{eyebrow}</p> : null}
        <Heading
          className={cn(level === 1 ? 'text-title' : 'text-headline-lg', 'text-ink', eyebrow && 'mt-2')}
        >
          {title}
        </Heading>
        {lede ? <div className="mt-3 text-lede text-ink-2">{lede}</div> : null}
        {note ? (
          <div className="mt-4 flex gap-2 border-l-2 border-accent-line pl-3 text-body-md text-ink-3">
            {note}
          </div>
        ) : null}
      </div>
      {hasActions ? (
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          {actions}
          {primaryAction}
        </div>
      ) : null}
    </header>
  );
}
