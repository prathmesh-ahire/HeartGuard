'use client';

import { useEffect, useRef, useState } from 'react';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { theme as generatedTheme } from '@/lib/generated';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * The one place ECharts is constructed (T113.1).
 *
 * ## Lazy, like three.js and for the same reason
 *
 * ECharts is roughly 300 kB minified. It is loaded through `await
 * import('echarts')` inside an effect, so it is not in any route's first-load
 * bundle and `scripts/20_check_bundle_budget.py` fails the build if it ever
 * gets there. An effect also means it never runs during the static export,
 * where there is no DOM to attach a canvas to.
 *
 * ## The theme is the figure palette
 *
 * Colours come from `generated/theme.json`, which `plot_style.py` writes. A
 * browser chart and the 300 dpi PNG beside it therefore colour series 3 the
 * same way by construction. Where a series colour fails contrast against the
 * page, the mark is stroked -- `needs_outline_on`, read from the measurement
 * rather than restated here.
 *
 * ## What this component will not do
 *
 * It takes an ECharts option object and renders it. It does not compute,
 * derive, aggregate or round anything. Every number reaching it came from
 * `generated/`, formatted in Python; the chart positions marks from the numeric
 * `values` array and prints text from `display`.
 */

export type EChartsOption = Record<string, unknown>;

export interface EChartProps {
  option: EChartsOption;
  /** Accessible description. A chart with no text alternative is a picture. */
  ariaLabel: string;
  className?: string;
  height?: number;
  /** Rendered under the chart, beside the download action. */
  caption?: string;
}

export function EChart({ option, ariaLabel, className, height = 320, caption }: EChartProps) {
  const container = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  const [status, setStatus] = useState<'loading' | 'ready' | 'failed'>('loading');
  const [message, setMessage] = useState('');

  useEffect(() => {
    const node = container.current;
    if (node === null) return;

    let disposed = false;
    let chart: { setOption: (o: unknown) => void; resize: () => void; dispose: () => void } | null =
      null;
    let observer: ResizeObserver | null = null;

    void (async () => {
      try {
        const echarts = await import('echarts');
        if (disposed) return;
        chart = echarts.init(node, undefined, { renderer: 'canvas' });
        chart.setOption({
          ...option,
          // Animation is decoration; someone who asked for it to stop gets a
          // chart that simply appears, which is also faster.
          animation: !reduced,
          textStyle: { fontFamily: 'inherit' },
        });
        observer = new ResizeObserver(() => chart?.resize());
        observer.observe(node);
        setStatus('ready');
      } catch (error) {
        if (disposed) return;
        setMessage(error instanceof Error ? error.message : String(error));
        setStatus('failed');
      }
    })();

    return () => {
      disposed = true;
      observer?.disconnect();
      chart?.dispose();
    };
  }, [option, reduced]);

  return (
    <figure className={cn('m-0', className)}>
      <div
        ref={container}
        role="img"
        aria-label={ariaLabel}
        style={{ height: height + 'px', width: '100%' }}
        className={cn(status === 'failed' ? 'hidden' : '')}
      />
      {status === 'failed' ? (
        <div
          role="alert"
          className={cn(
            'rounded border border-rose-300 bg-rose-50 p-4 dark:border-rose-800 dark:bg-rose-950/50',
            TYPE_SCALE.caption,
          )}
        >
          <p className="font-medium">This chart could not be drawn.</p>
          <p className="mt-1">
            The charting library failed to load, so the marks are missing — this is a failure, not
            an empty result. {message}
          </p>
        </div>
      ) : null}
      {caption === undefined ? null : (
        <figcaption className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>{caption}</figcaption>
      )}
    </figure>
  );
}

/**
 * Axis, grid and tooltip styling shared by every chart, in both themes.
 *
 * Returned rather than exported as a constant because it reads the resolved
 * theme: the axis of a chart on a dark page has to be light, and a single
 * static object cannot be both.
 */
export function chartBase(dark: boolean): EChartsOption {
  const ink = dark ? '#e2e8f0' : '#0f172a';
  const muted = dark ? '#94a3b8' : '#475569';
  const grid = dark ? '#1e293b' : '#e2e8f0';
  return {
    backgroundColor: 'transparent',
    grid: { left: 56, right: 20, top: 28, bottom: 44, containLabel: true },
    textStyle: { color: ink },
    tooltip: {
      trigger: 'item',
      backgroundColor: dark ? '#0f172a' : '#ffffff',
      borderColor: grid,
      textStyle: { color: ink },
    },
    legend: { textStyle: { color: muted }, top: 0 },
    xAxis: {
      axisLine: { lineStyle: { color: muted } },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: grid } },
    },
    yAxis: {
      axisLine: { lineStyle: { color: muted } },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: grid } },
    },
  };
}

/** The colormaps the matplotlib figures use, so a heatmap matches its PNG. */
export const COLORMAPS = generatedTheme.colormaps;
