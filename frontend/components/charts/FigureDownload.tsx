'use client';

import { cn } from '@/lib/cn';
import { figure as generatedFigure } from '@/lib/generated/figures';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * "Download the 300 dpi figure", beside every chart (T113.4).
 *
 * The link serves the **canonical matplotlib PNG** -- the same file that goes
 * into the thesis and the paper -- not a rasterised screenshot of the browser
 * chart. That distinction is the whole point of the task. An ECharts canvas
 * exported at devicePixelRatio is 96 dpi of a different renderer with different
 * fonts, and two figures of the same data that do not match is precisely the
 * discrepancy a reviewer notices.
 *
 * A missing PNG renders as a stated absence rather than a dead link: an anchor
 * pointing at a 404 tells the reader nothing about why.
 */

export function FigureDownload({
  figureId,
  className,
}: {
  figureId: string;
  className?: string;
}) {
  const figure = generatedFigure(figureId);

  if (figure === undefined || figure.png === null) {
    return (
      <p className={cn(TYPE_SCALE.caption, SURFACE.subtle, className)}>
        No print version of figure {figureId} is available.
      </p>
    );
  }

  return (
    <div className={cn('flex flex-wrap items-center gap-x-3 gap-y-1', className)}>
      <a
        href={figure.png}
        download
        className={cn(
          'inline-flex items-center gap-1.5 rounded border px-2.5 py-1',
          'border-line text-label-md uppercase text-ink-2',
          'hover:border-accent-line hover:text-accent-strong',
        )}
      >
        <span aria-hidden="true">↓</span>
        Download {figureId} at {figure.dpi} dpi
      </a>
      <span className={cn(TYPE_SCALE.caption, SURFACE.subtle)}>
        the print figure, identical to the one in the thesis
      </span>
    </div>
  );
}
