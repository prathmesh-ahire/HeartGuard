import { Button, ButtonLink } from '@/components/ui/Button';
import { SkeletonChart, SkeletonList, SkeletonTiles } from '@/components/ui/Skeleton';
import { EmptyState, ErrorState, LoadingRegion } from '@/components/ui/States';

/**
 * The empty, loading and error states of the new pages (T128.6), written once
 * so Phases 130-134 use them instead of each inventing its own.
 *
 * Three rules, all from the states they build on:
 *
 * * **Empty** says why there is nothing and offers the one action that fixes
 *   it. A fresh install is not an error.
 * * **Loading** is the shape of what is coming, announced once.
 * * **Error** is an error: the `role="alert"` panel, never an empty chart and
 *   never a table with no rows, which would read as a real result of zero.
 *
 * No preset contains a number.
 */

export function HistoryEmpty() {
  return (
    <EmptyState
      icon="history"
      title="No analyses yet"
      description="Every recording you analyse is kept here with its result, so you can come back to it, add notes and compare two side by side."
      action={
        <ButtonLink href="/" tone="primary" icon="pulse">
          Analyse a recording
        </ButtonLink>
      }
    />
  );
}

export function NoMatches({ onClear }: { onClear: () => void }) {
  return (
    <EmptyState
      icon="features"
      title="Nothing matches these filters"
      description="Your history has analyses, but none of them fit the search and filters you chose."
      action={
        <Button onClick={onClear} icon="close">
          Clear filters
        </Button>
      }
    />
  );
}

export function InsightsEmpty() {
  return (
    <EmptyState
      icon="insights"
      title="No trends to show yet"
      description="Insights are drawn from the analyses in your history. Analyse a few recordings and the trends appear here."
      action={
        <ButtonLink href="/" tone="primary" icon="pulse">
          Analyse a recording
        </ButtonLink>
      }
    />
  );
}

export function ServiceUnavailable({ onRetry }: { onRetry?: () => void }) {
  return (
    <ErrorState
      title="The analysis service is not responding"
      detail="The page loaded, but the local analysis service did not answer. Start the service and try again."
      onRetry={onRetry}
    />
  );
}

export function HistoryLoading() {
  return (
    <LoadingRegion label="Loading your history">
      <SkeletonList />
    </LoadingRegion>
  );
}

export function InsightsLoading() {
  return (
    <LoadingRegion label="Loading insights" className="space-y-6">
      <SkeletonTiles />
      <div className="grid gap-6 lg:grid-cols-2">
        <SkeletonChart />
        <SkeletonChart />
      </div>
    </LoadingRegion>
  );
}
