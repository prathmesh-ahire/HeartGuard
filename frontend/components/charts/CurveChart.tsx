'use client';

import { useTheme } from 'next-themes';
import { useEffect, useMemo, useState } from 'react';

import { EChart, chartBase, type EChartsOption } from '@/components/charts/EChart';
import { EmptyState } from '@/components/ui/States';
import { seriesColor } from '@/lib/tokens';

/**
 * A cross-validated curve: the mean across folds, with a one-SD band (T115.4).
 *
 * ## The band is not decoration
 *
 * A mean ROC drawn without its spread reads as a single measurement. It is not:
 * it is the average of 25 curves computed inside 25 different folds, and where
 * those folds disagree the band is wide. `curves.py` computes both; this
 * component draws what it is handed and derives nothing.
 *
 * The band is drawn as a stacked transparent base plus a visible ribbon, which
 * is how ECharts expresses an area between two lines. `mean - sd` is clamped at
 * zero because a rate below zero is not a thing a reader should be shown; the
 * clamp is a drawing bound and the underlying numbers are untouched in the
 * payload.
 */

function useDark(): boolean {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted && resolvedTheme === 'dark';
}

export function CurveChart({
  x,
  mean,
  sd,
  xName,
  yName,
  label,
  caption,
  diagonal = false,
  className,
  height = 300,
}: {
  x: number[];
  mean: number[];
  sd: number[];
  xName: string;
  yName: string;
  label: string;
  caption?: string;
  /** The chance line, for ROC only. PR's baseline is prevalence, not 0.5. */
  diagonal?: boolean;
  className?: string;
  height?: number;
}) {
  const dark = useDark();

  const option = useMemo<EChartsOption>(() => {
    const base = chartBase(dark);
    const lower = mean.map((value, index) => Math.max(0, value - (sd[index] ?? 0)));
    // The band is expressed as a transparent base plus a visible ribbon, which
    // is how ECharts draws an area between two lines. Both bounds are clamped
    // to [0, 1] for drawing only; the payload's numbers are untouched.
    const width = mean.map(
      (value, index) => Math.min(1, value + (sd[index] ?? 0)) - (lower[index] ?? 0),
    );

    return {
      ...base,
      tooltip: { ...(base.tooltip as object), trigger: 'axis' },
      grid: { left: 56, right: 20, top: 16, bottom: 44 },
      xAxis: {
        ...(base.xAxis as object),
        type: 'category',
        data: x.map((value) => value.toFixed(2)),
        name: xName,
        nameLocation: 'middle',
        nameGap: 28,
      },
      yAxis: {
        ...(base.yAxis as object),
        type: 'value',
        min: 0,
        max: 1,
        name: yName,
      },
      series: [
        {
          name: 'lower',
          type: 'line',
          stack: 'band',
          symbol: 'none',
          lineStyle: { opacity: 0 },
          data: lower,
          silent: true,
          tooltip: { show: false },
        },
        {
          name: '+/- 1 SD across folds',
          type: 'line',
          stack: 'band',
          symbol: 'none',
          lineStyle: { opacity: 0 },
          areaStyle: { color: seriesColor(0), opacity: 0.18 },
          data: width,
          silent: true,
        },
        {
          name: 'mean across folds',
          type: 'line',
          symbol: 'none',
          lineStyle: { width: 2, color: seriesColor(0) },
          itemStyle: { color: seriesColor(0) },
          data: mean,
        },
        ...(diagonal
          ? [
              {
                name: 'chance',
                type: 'line',
                symbol: 'none',
                lineStyle: { type: 'dashed', width: 1, color: dark ? '#94a3b8' : '#64748b' },
                data: x,
                silent: true,
              },
            ]
          : []),
      ],
    };
  }, [dark, diagonal, mean, sd, x, xName, yName]);

  if (x.length === 0 || mean.length === 0) {
    return (
      <EmptyState
        className={className}
        title="No curve to draw"
        description="No curve points were exported for this selection."
      />
    );
  }

  return (
    <div className={className}>
      <EChart option={option} ariaLabel={label} caption={caption} height={height} />
    </div>
  );
}
