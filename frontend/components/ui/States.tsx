import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * Loading, empty and error (T111.4).
 *
 * The rule that matters here is the third one: **a failed request must render a
 * visible error, never an empty chart.** An empty chart is indistinguishable
 * from a real result of zero, and the reader has no way to tell that anything
 * went wrong. So `ErrorState` is loud, says what failed, and offers a retry --
 * and `EmptyState` is explicitly a different component with different wording,
 * so "no data" and "the request failed" can never be confused for each other.
 */

export function LoadingState({
  label = 'Loading',
  className,
  rows = 3,
}: {
  label?: string;
  className?: string;
  rows?: number;
}) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-busy="true"
      className={cn(SURFACE.sunken, 'p-5', className)}
    >
      <span className="sr-only">{label}</span>
      <div className="space-y-3" aria-hidden="true">
        {Array.from({ length: rows }).map((_, index) => (
          <div
            key={index}
            className="h-3 animate-pulse rounded bg-slate-200 dark:bg-slate-800"
            style={{ width: `${100 - index * 12}%` }}
          />
        ))}
      </div>
      <p className={cn(TYPE_SCALE.caption, SURFACE.subtle, 'mt-4')}>{label}…</p>
    </div>
  );
}

export function EmptyState({
  title = 'Nothing to show',
  description,
  action,
  className,
}: {
  title?: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'rounded-lg border border-dashed border-slate-300 p-6 text-center dark:border-slate-700',
        className,
      )}
    >
      <p className={cn(TYPE_SCALE.h3)}>{title}</p>
      {description ? (
        <div className={cn(TYPE_SCALE.body, SURFACE.muted, 'mx-auto mt-2 max-w-prose')}>
          {description}
        </div>
      ) : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

export function ErrorState({
  title = 'Something failed',
  detail,
  onRetry,
  className,
}: {
  title?: string;
  detail?: ReactNode;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        'rounded-lg border-2 border-rose-400 bg-rose-50 p-5',
        'dark:border-rose-700 dark:bg-rose-950/40',
        className,
      )}
    >
      <p className={cn(TYPE_SCALE.h3, 'text-rose-800 dark:text-rose-200')}>{title}</p>
      {detail ? (
        <div
          className={cn(
            TYPE_SCALE.body,
            'mt-2 break-words text-rose-800/90 dark:text-rose-200/90',
          )}
        >
          {detail}
        </div>
      ) : null}
      <p className={cn(TYPE_SCALE.caption, 'mt-3 text-rose-700 dark:text-rose-300')}>
        No result is shown above, because there is no result — this is a failure, not a
        value of zero.
      </p>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 rounded border border-rose-500 px-3 py-1.5 text-sm font-medium text-rose-800 hover:bg-rose-100 dark:border-rose-600 dark:text-rose-200 dark:hover:bg-rose-900/40"
        >
          Try again
        </button>
      ) : null}
    </div>
  );
}
