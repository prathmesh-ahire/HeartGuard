'use client';

import { useMemo, useState } from 'react';

import { EChart, chartBase, type EChartsOption } from '@/components/charts/EChart';
import { EmptyState } from '@/components/ui/States';
import { cn } from '@/lib/cn';
import { seriesColor } from '@/lib/tokens';
import { useTheme } from 'next-themes';
import { useEffect } from 'react';
import type { GeneratedFramePayload, GeneratedOptimization } from '@/lib/generated/types';

/**
 * The two views the optimization page needs (T115.5).
 *
 * `ConvergencePanel` draws one line per outer fold, because that is what the
 * search produced: the search runs inside each training fold and there are
 * therefore several independent searches, not several attempts at one.
 *
 * `FrameTable` renders a passthrough CSV payload. It is the one generated shape
 * whose columns are not enumerated in TypeScript — search runs write different
 * columns — so the component reads `columns[].display` and nothing else. The
 * `values` arrays are present in the payload for charts and are deliberately
 * not touched here: a table renders strings.
 */

type Run = GeneratedOptimization['runs'][number];

function useDark(): boolean {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted && resolvedTheme === 'dark';
}

export function ConvergencePanel({ run }: { run: Run }) {
  const dark = useDark();
  const convergence = run.convergence;

  const option = useMemo<EChartsOption>(() => {
    const base = chartBase(dark);
    const series = convergence?.series ?? [];
    return {
      ...base,
      tooltip: { ...(base.tooltip as object), trigger: 'axis' },
      legend: { top: 0, type: 'scroll' },
      grid: { left: 56, right: 20, top: 34, bottom: 44 },
      xAxis: {
        ...(base.xAxis as object),
        type: 'value',
        name: convergence?.x_label ?? 'trial',
        nameLocation: 'middle',
        nameGap: 26,
      },
      yAxis: {
        ...(base.yAxis as object),
        type: 'value',
        name: convergence?.y_label ?? 'score',
        scale: true,
      },
      series: series.map((item, index: number) => ({
        name: item.label,
        type: 'line',
        step: 'end',
        symbol: 'none',
        lineStyle: { width: 1.4, color: seriesColor(index) },
        itemStyle: { color: seriesColor(index) },
        data: item.x.map((value, position) => [value, item.y[position]]),
      })),
    };
  }, [convergence, dark]);

  return (
    <article className="rounded-lg border border-slate-200 p-5 dark:border-slate-800">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-medium">
          <span className="font-mono text-sky-700 dark:text-sky-400">{run.run_id}</span>{' '}
          {run.title}
        </h3>
        {!run.available ? (
          <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-800 dark:bg-amber-950/50 dark:text-amber-300">
            not run
          </span>
        ) : null}
      </header>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{run.description}</p>

      {convergence?.available ? (
        <>
          <EChart
            className="mt-4"
            option={option}
            ariaLabel={'Convergence for ' + run.run_id}
            height={260}
            caption={
              'One trace per ' +
              (convergence.n_series === 1 ? 'search' : 'outer fold and model') +
              '. ' +
              String(convergence.n_series) +
              ' traces, best-so-far against ' +
              String(convergence.x_label) +
              '.'
            }
          />
        </>
      ) : (
        <EmptyState
          className="mt-4"
          title="No convergence trace for this run"
          description={
            convergence?.reason ??
            run.reason ??
            'This run records its result without a per-trial trace.'
          }
        />
      )}
    </article>
  );
}

export function FrameTable({
  frame,
  className,
  maxRows = 60,
}: {
  frame: GeneratedFramePayload;
  className?: string;
  maxRows?: number;
}) {
  const [expanded, setExpanded] = useState(false);

  if (!frame.available) {
    return (
      <EmptyState
        className={className}
        title="Not produced"
        description={frame.reason ?? 'This search run has not written its output.'}
      />
    );
  }

  const shown = expanded ? frame.n_rows : Math.min(maxRows, frame.n_rows);

  return (
    <div className={className}>
      <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-widest text-slate-500 dark:bg-slate-900/60">
            <tr>
              {frame.columns.map((column) => (
                <th key={column.name} scope="col" className="whitespace-nowrap px-3 py-2">
                  {column.name.replace(/_/g, ' ')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: shown }, (_unused, row) => (
              <tr key={row} className="border-t border-slate-100 dark:border-slate-800">
                {frame.columns.map((column) => (
                  <td
                    key={column.name}
                    className={cn(
                      'whitespace-nowrap px-3 py-1.5',
                      column.values === null ? '' : 'tabular-nums',
                    )}
                  >
                    {column.display[row]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-500">
        {shown} of {frame.n_rows} rows from {frame.source}.{' '}
        {frame.n_rows > maxRows ? (
          <button
            type="button"
            onClick={() => setExpanded((current) => !current)}
            className="underline"
          >
            {expanded ? 'Show fewer' : 'Show all rows'}
          </button>
        ) : null}
      </p>
    </div>
  );
}
