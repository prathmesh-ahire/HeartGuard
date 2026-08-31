'use client';

import { useTheme } from 'next-themes';
import { useEffect, useMemo, useState } from 'react';

import { EChart, chartBase, type EChartsOption } from '@/components/charts/EChart';
import { EmptyState } from '@/components/ui/States';
import { needsOutlineOn, seriesColor } from '@/lib/tokens';

/**
 * A waveform, or several overlaid (T114.4).
 *
 * Takes arrays of numbers that Python already produced and draws them. It does
 * not filter, normalise, resample, decimate or smooth: every series handed to
 * it is a state the preprocessing pipeline computed, strided to a display
 * budget in `src/reporting/signals.py`. A browser-side filter would be a second
 * implementation of the pipeline's most load-bearing step, and the reader would
 * believe the one on screen.
 *
 * Points are drawn without symbols and with `sampling: 'lttb'` for rendering
 * only — that is ECharts choosing which of the given points to paint at the
 * current pixel width, not a transformation of the data behind them.
 */

function useDark(): boolean {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted && resolvedTheme === 'dark';
}

export interface SignalSeries {
  name: string;
  values: number[];
  colorIndex?: number;
}

export function SignalChart({
  time,
  series,
  label,
  caption,
  height = 260,
  yName = 'Amplitude',
}: {
  time: number[];
  series: SignalSeries[];
  label: string;
  caption?: string;
  height?: number;
  yName?: string;
}) {
  const dark = useDark();

  const option = useMemo<EChartsOption>(() => {
    const base = chartBase(dark);
    return {
      ...base,
      tooltip: { ...(base.tooltip as object), trigger: 'axis' },
      legend: series.length > 1 ? { top: 0, data: series.map((item) => item.name) } : undefined,
      grid: { left: 56, right: 16, top: series.length > 1 ? 34 : 12, bottom: 40 },
      xAxis: {
        ...(base.xAxis as object),
        type: 'value',
        name: 'Time (s)',
        nameLocation: 'middle',
        nameGap: 26,
        min: time.length > 0 ? time[0] : 0,
        max: time.length > 0 ? time[time.length - 1] : 1,
      },
      yAxis: { ...(base.yAxis as object), type: 'value', name: yName, scale: true },
      series: series.map((item, index) => {
        const colorIndex = item.colorIndex ?? index;
        return {
          name: item.name,
          type: 'line',
          showSymbol: false,
          sampling: 'lttb',
          lineStyle: {
            width: needsOutlineOn(colorIndex, dark ? 'dark' : 'light') ? 1.6 : 1.1,
            color: seriesColor(colorIndex),
          },
          itemStyle: { color: seriesColor(colorIndex) },
          data: item.values.map((value, position) => [time[position], value]),
        };
      }),
    };
  }, [dark, series, time, yName]);

  if (time.length === 0 || series.length === 0) {
    return (
      <EmptyState
        title="No signal to draw"
        description="No precomputed example was exported for this record."
      />
    );
  }

  return <EChart option={option} ariaLabel={label} caption={caption} height={height} />;
}
