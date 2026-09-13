import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';

/**
 * The head of a document page.
 *
 * Every page opens the same way -- what this page is, then the one caveat the
 * reader has to carry into it -- so it is one component rather than an h1 hand
 * written thirteen times. The `note` slot is where a population caveat or a
 * scope limit goes, and it is styled as an instrument annotation rather than
 * as body text so it does not read as optional.
 */
export function PageHeader({
  title,
  lede,
  note,
  actions,
  className,
  level = 1,
}: {
  title: string;
  lede?: ReactNode;
  note?: ReactNode;
  actions?: ReactNode;
  className?: string;
  /** 2 when the header opens a section inside a page that already has an h1. */
  level?: 1 | 2;
}) {
  const Heading = level === 1 ? 'h1' : 'h2';
  return (
    <header
      className={cn(
        'flex flex-wrap items-start justify-between gap-4 border-b border-line pb-4',
        className,
      )}
    >
      <div className="max-w-3xl">
        <Heading className="text-headline-lg text-ink">{title}</Heading>
        {lede ? <div className="mt-2 text-body-lg text-ink-2">{lede}</div> : null}
        {note ? (
          <div className="mt-3 flex gap-2 border-l-2 border-accent-line pl-3 text-body-sm text-ink-3">
            {note}
          </div>
        ) : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </header>
  );
}
