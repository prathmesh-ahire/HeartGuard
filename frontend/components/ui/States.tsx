import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { Icon, type IconName } from '@/components/ui/Icon';
import { SkeletonText } from '@/components/ui/Skeleton';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * Loading, empty and error (T111.4, refreshed in T128.6).
 *
 * The rule that matters here is the third one: **a failed request must render a
 * visible error, never an empty chart.** An empty chart is indistinguishable
 * from a real result of zero, and the reader has no way to tell that anything
 * went wrong. So `ErrorState` is loud, says what failed, and offers a retry --
 * and `EmptyState` is explicitly a different component with different wording,
 * so "no data" and "the request failed" can never be confused for each other.
 *
 * The page-specific presets built on these live in `PageStates.tsx`.
 */

/** A region that is waiting. Announced once; its skeletons are silent. */
export function LoadingRegion({
  label,
  children,
  className,
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div role="status" aria-live="polite" aria-busy="true" className={className}>
      <span className="sr-only">{label}</span>
      {children}
    </div>
  );
}

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
      <SkeletonText lines={rows} />
      <p className={cn(TYPE_SCALE.caption, SURFACE.subtle, 'mt-4')} aria-hidden="true">
        {label}…
      </p>
    </div>
  );
}

export function EmptyState({
  title = 'Nothing to show',
  description,
  action,
  icon,
  className,
}: {
  title?: string;
  description?: ReactNode;
  action?: ReactNode;
  icon?: IconName;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'rounded-xl border border-dashed border-line bg-panel/60 px-6 py-10 text-center',
        className,
      )}
    >
      {icon ? (
        <span className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-accent-soft text-accent">
          <Icon name={icon} className="h-5 w-5" strokeWidth={1.8} />
        </span>
      ) : null}
      <p className={cn(TYPE_SCALE.h2, 'text-ink')}>{title}</p>
      {description ? (
        <div className={cn(TYPE_SCALE.lead, SURFACE.muted, 'mx-auto mt-2 max-w-reading')}>
          {description}
        </div>
      ) : null}
      {action ? <div className="mt-6 flex justify-center">{action}</div> : null}
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
        'flex gap-3 rounded-xl border-2 border-danger-line bg-danger-soft p-5',
        className,
      )}
    >
      {/* An icon as well as a colour: with a wine accent on every control,
          hue alone cannot be what tells a failure from ordinary chrome. */}
      <Icon name="limitations" className="mt-0.5 h-5 w-5 shrink-0 text-danger" strokeWidth={1.8} />
      <div className="min-w-0">
        <p className={cn(TYPE_SCALE.h3, 'text-danger')}>{title}</p>
        {detail ? (
          <div className={cn(TYPE_SCALE.body, 'mt-1.5 break-words text-danger')}>{detail}</div>
        ) : null}
        <p className={cn(TYPE_SCALE.caption, 'mt-2 text-danger')}>
          No result is shown above, because there is no result — this is a failure, not a
          value of zero.
        </p>
        {onRetry ? (
          <button
            type="button"
            onClick={onRetry}
            className="mt-3 rounded-lg border border-danger-line px-3 py-1.5 text-label-md uppercase text-danger hover:bg-danger-line/20"
          >
            Try again
          </button>
        ) : null}
      </div>
    </div>
  );
}
