'use client';

import { useMemo, useState } from 'react';

import { EChart, chartBase, type EChartsOption } from '@/components/charts/EChart';
import { EmptyState } from '@/components/ui/States';
import { cn } from '@/lib/cn';
import { userFacing } from '@/lib/internal';
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
    <article className="overflow-hidden rounded-xl border border-line bg-panel shadow-panel">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
        <h3 className="flex flex-wrap items-baseline gap-2 text-headline-sm text-ink">
          <span className="rounded border border-accent-line bg-accent-soft px-1.5 py-0.5 text-label-md uppercase text-accent-deep">
            {run.run_id}
          </span>
          {run.title}
        </h3>
        {!run.available ? (
          <span className="rounded border border-warn-line bg-warn-soft px-1.5 py-0.5 text-label-sm uppercase text-warn">
            not run
          </span>
        ) : null}
      </header>
      <p className="px-4 pt-3 text-body-md text-ink-2">{run.description}</p>

      {convergence?.available ? (
        <>
          <EChart
            className="telemetry-grid mt-3 p-3"
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
          className="m-4"
          title="No convergence trace for this run"
          description={
            userFacing(convergence?.reason) ??
            userFacing(run.reason) ??
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
      <div className="overflow-x-auto rounded-lg border border-line bg-panel">
        <table className="min-w-full border-collapse text-left text-body-sm">
          <thead className="border-b-2 border-line bg-sunken text-label-sm uppercase text-ink-3">
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
              <tr key={row} className="border-t border-line hover:bg-accent-soft/40">
                {frame.columns.map((column) => (
                  <td
                    key={column.name}
                    className={cn(
                      'whitespace-nowrap px-3 py-1.5',
                      column.values === null ? '' : 'stat',
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
      <p className="mt-2 font-mono text-body-sm text-ink-3">
        {shown} of {frame.n_rows} rows.{' '}
        {frame.n_rows > maxRows ? (
          <button
            type="button"
            onClick={() => setExpanded((current) => !current)}
            className="text-accent-strong underline decoration-dotted underline-offset-2 hover:decoration-solid"
          >
            {expanded ? 'Show fewer' : 'Show all rows'}
          </button>
        ) : null}
      </p>
    </div>
  );
}
