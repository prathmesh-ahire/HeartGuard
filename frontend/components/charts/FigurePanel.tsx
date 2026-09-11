import { cn } from '@/lib/cn';
import { FigureDownload } from '@/components/charts/FigureDownload';
import { EmptyState } from '@/components/ui/States';
import { figure as generatedFigure } from '@/lib/generated/figures';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * A G-figure shown as its canonical 300 dpi matplotlib PNG, with its caption.
 *
 * Used where the figure is multi-panel (G29-G35 are two panels each) and an
 * interactive re-drawing would be a second rendering of the same numbers that
 * could disagree with the print version. The interactive part of those pages
 * is the table beside it, which carries the same values as display strings.
 * A figure that has not been generated renders its absence, never an empty box.
 */
export function FigurePanel({ figureId, className }: { figureId: string; className?: string }) {
  const meta = generatedFigure(figureId);
  if (meta === undefined || meta.png === null) {
    return (
      <EmptyState
        className={className}
        title={figureId + ' has not been generated'}
        description="It is produced by the Part VIII figure scripts, not by this page."
      />
    );
  }
  return (
    <figure className={cn('space-y-2', className)}>
      {/* eslint-disable-next-line @next/next/no-img-element -- a static export has no image optimizer */}
      <img
        src={meta.png}
        alt={meta.id + ': ' + meta.title}
        loading="lazy"
        className="w-full rounded border border-slate-200 bg-white dark:border-slate-800"
      />
      <figcaption className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
        <span className="font-semibold">
          {meta.id} — {meta.title}.
        </span>{' '}
        {meta.caption}
      </figcaption>
      <FigureDownload figureId={figureId} />
    </figure>
  );
}
