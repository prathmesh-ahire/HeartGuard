import { cn } from '@/lib/cn';

/**
 * Skeleton loaders (T128.5): the shape of what is coming, in place of it.
 *
 * A skeleton is the SHAPE of a list, a row of tiles or a chart -- never a
 * number, a bar height taken from data, or anything that could be read as a
 * result. The pulse is behind `motion-safe:`, so under reduced motion the
 * blocks are still.
 *
 * Skeletons are `aria-hidden`. The announcement belongs to the region that
 * contains them (`LoadingRegion` in `States.tsx`), so a screen reader hears
 * "Loading your history" once rather than a dozen empty shapes.
 */
export function Skeleton({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      data-skeleton=""
      className={cn('block rounded-md bg-accent-soft/70 motion-safe:animate-skeleton', className)}
    />
  );
}

const LINE_WIDTHS = ['w-full', 'w-11/12', 'w-4/5', 'w-2/3', 'w-3/4'] as const;

export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <span aria-hidden="true" className={cn('block space-y-2.5', className)}>
      {Array.from({ length: lines }, (_, index) => (
        <Skeleton key={index} className={cn('h-3', LINE_WIDTHS[index % LINE_WIDTHS.length])} />
      ))}
    </span>
  );
}

export function SkeletonTiles({ count = 4, className }: { count?: number; className?: string }) {
  return (
    <div aria-hidden="true" className={cn('grid gap-4 sm:grid-cols-2 lg:grid-cols-4', className)}>
      {Array.from({ length: count }, (_, index) => (
        <div key={index} className="rounded-xl border border-line bg-panel p-5">
          <Skeleton className="h-2.5 w-1/2" />
          <Skeleton className="mt-4 h-7 w-2/3" />
          <Skeleton className="mt-3 h-2.5 w-4/5" />
        </div>
      ))}
    </div>
  );
}

export function SkeletonList({ rows = 5, className }: { rows?: number; className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={cn('divide-y divide-line rounded-xl border border-line bg-panel', className)}
    >
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="flex items-center gap-4 px-5 py-4">
          <Skeleton className="h-9 w-9 shrink-0 rounded-lg" />
          <div className="min-w-0 flex-1">
            <Skeleton className={cn('h-3', LINE_WIDTHS[(index + 2) % LINE_WIDTHS.length])} />
            <Skeleton className="mt-2 h-2.5 w-1/3" />
          </div>
          <Skeleton className="h-5 w-16 shrink-0 rounded-full" />
        </div>
      ))}
    </div>
  );
}

/** Fixed, decorative bar heights: a silhouette, not a distribution. */
const BAR_HEIGHTS = ['h-1/3', 'h-1/2', 'h-2/3', 'h-2/5', 'h-3/4', 'h-1/2', 'h-3/5'] as const;

export function SkeletonChart({ className }: { className?: string }) {
  return (
    <div aria-hidden="true" className={cn('rounded-xl border border-line bg-panel p-5', className)}>
      <Skeleton className="h-3 w-1/3" />
      <div className="mt-6 flex h-40 items-end gap-3">
        {BAR_HEIGHTS.map((height, index) => (
          <Skeleton key={index} className={cn('flex-1 rounded-b-none', height)} />
        ))}
      </div>
    </div>
  );
}
