'use client';

import { useTheme } from 'next-themes';
import { useEffect, useMemo, useState } from 'react';

import { EChart, chartBase, type EChartsOption } from '@/components/charts/EChart';
import { EmptyState } from '@/components/ui/States';
import { needsOutlineOn, seriesColor } from '@/lib/tokens';
import {
  displayColumn,
  hasRows,
  numericColumn,
  points,
  type Source,
} from '@/components/charts/types';

/**
 * The six chart types T113.2 names: ROC, PR, confusion-matrix heatmap, grouped
 * bars, scatter and calibration curve.
 *
 * Every one of them takes a **generated table or figure** and nothing else. A
 * chart with no source renders `EmptyState` saying which experiment has not
 * been run — never an empty axis, because an empty axis is indistinguishable
 * from a real result of zero and the reader cannot tell which they are looking
 * at.
 *
 * ROC, PR, confusion and calibration are wired to `outputs/06_binary_results`
 * and `outputs/07_multiclass_results`, which the exporter reads only under
 * `--include-results`. Until those experiments settle they render the absence.
 * That is the honest state of the dashboard tonight, not a placeholder.
 */

/** Resolved theme, for axis colours. False during SSR and the first render. */
function useDark(): boolean {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted && resolvedTheme === 'dark';
}

function Absent({ what, why }: { what: string; why: string }) {
  return <EmptyState title={what} description={why} />;
}

const NO_RESULTS =
  'The results directories are excluded from this export while the experiments are ' +
  'still being written. Re-run scripts/17_export_frontend_data.py with --include-results ' +
  'once they have settled.';

// ---------------------------------------------------------------------------
// ROC and PR: the same shape, different axes and different reference line
// ---------------------------------------------------------------------------

interface CurveProps {
  source?: Source;
  xColumn: string;
  yColumn: string;
  label: string;
  caption?: string;
  height?: number;
}

function curveOption(
  dark: boolean,
  data: [number, number][],
  xName: string,
  yName: string,
  chance: [number, number][] | null,
  colorIndex: number,
): EChartsOption {
  const base = chartBase(dark);
  return {
    ...base,
    tooltip: { ...(base.tooltip as object), trigger: 'axis' },
    xAxis: { ...(base.xAxis as object), type: 'value', min: 0, max: 1, name: xName },
    yAxis: { ...(base.yAxis as object), type: 'value', min: 0, max: 1, name: yName },
    series: [
      ...(chance === null
        ? []
        : [
            {
              name: 'chance',
              type: 'line',
              data: chance,
              showSymbol: false,
              lineStyle: { type: 'dashed', width: 1, color: dark ? '#64748b' : '#94a3b8' },
              silent: true,
            },
          ]),
      {
        name: yName,
        type: 'line',
        data,
        showSymbol: false,
        lineStyle: { width: 2, color: seriesColor(colorIndex) },
        // The stroke rule from theme.json: a fill too close to the page ground
        // needs an outline. A line at 2 px is the stroke.
        itemStyle: { color: seriesColor(colorIndex) },
      },
    ],
  };
}

export function RocCurve({ source, xColumn, yColumn, label, caption, height }: CurveProps) {
  const dark = useDark();
  const data = points(numericColumn(source, xColumn), numericColumn(source, yColumn));
  const option = useMemo(
    () =>
      curveOption(
        dark,
        data,
        'False positive rate',
        'True positive rate',
        [
          [0, 0],
          [1, 1],
        ],
        0,
      ),
    [dark, data],
  );
  if (!hasRows(source) || data.length === 0) {
    return <Absent what="No ROC curve yet" why={NO_RESULTS} />;
  }
  return <EChart option={option} ariaLabel={label} caption={caption} height={height} />;
}

export function PrCurve({ source, xColumn, yColumn, label, caption, height }: CurveProps) {
  const dark = useDark();
  const data = points(numericColumn(source, xColumn), numericColumn(source, yColumn));
  const option = useMemo(
    () => curveOption(dark, data, 'Recall', 'Precision', null, 1),
    [dark, data],
  );
  if (!hasRows(source) || data.length === 0) {
    return <Absent what="No precision-recall curve yet" why={NO_RESULTS} />;
  }
  return <EChart option={option} ariaLabel={label} caption={caption} height={height} />;
}

// ---------------------------------------------------------------------------
// Confusion matrix
// ---------------------------------------------------------------------------

export function ConfusionMatrix({
  source,
  trueColumn,
  predictedColumn,
  countColumn,
  label,
  caption,
  height = 360,
}: {
  source?: Source;
  trueColumn: string;
  predictedColumn: string;
  countColumn: string;
  label: string;
  caption?: string;
  height?: number;
}) {
  const dark = useDark();
  const trueLabels = displayColumn(source, trueColumn);
  const predictedLabels = displayColumn(source, predictedColumn);
  const counts = numericColumn(source, countColumn);

  const rows = useMemo(() => Array.from(new Set(trueLabels)), [trueLabels]);
  const columns = useMemo(() => Array.from(new Set(predictedLabels)), [predictedLabels]);

  const cells = useMemo(
    () =>
      counts
        .map((count, index) => {
          const x = columns.indexOf(predictedLabels[index] ?? '');
          const y = rows.indexOf(trueLabels[index] ?? '');
          return x < 0 || y < 0 || count === null ? null : [x, y, count];
        })
        .filter((cell): cell is number[] => cell !== null),
    [counts, columns, rows, predictedLabels, trueLabels],
  );

  const option = useMemo<EChartsOption>(() => {
    const base = chartBase(dark);
    const max = cells.reduce((best, cell) => Math.max(best, cell[2] ?? 0), 0);
    return {
      ...base,
      tooltip: { ...(base.tooltip as object), position: 'top' },
      grid: { left: 90, right: 24, top: 24, bottom: 70, containLabel: true },
      xAxis: { type: 'category', data: columns, name: 'Predicted', axisLabel: { rotate: 30 } },
      yAxis: { type: 'category', data: rows, name: 'Actual' },
      visualMap: {
        min: 0,
        max: max || 1,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        textStyle: { color: dark ? '#94a3b8' : '#475569' },
      },
      series: [
        {
          type: 'heatmap',
          data: cells,
          label: { show: true, color: dark ? '#e2e8f0' : '#0f172a' },
          emphasis: { itemStyle: { borderColor: dark ? '#e2e8f0' : '#0f172a', borderWidth: 1 } },
        },
      ],
    };
  }, [cells, columns, dark, rows]);

  if (!hasRows(source) || cells.length === 0) {
    return <Absent what="No confusion matrix yet" why={NO_RESULTS} />;
  }
  return <EChart option={option} ariaLabel={label} caption={caption} height={height} />;
}

// ---------------------------------------------------------------------------
// Grouped bars
// ---------------------------------------------------------------------------

export function GroupedBars({
  source,
  categoryColumn,
  valueColumns,
  label,
  caption,
  height,
  horizontal = false,
}: {
  source?: Source;
  categoryColumn: string;
  valueColumns: string[];
  label: string;
  caption?: string;
  height?: number;
  horizontal?: boolean;
}) {
  const dark = useDark();
  const categories = displayColumn(source, categoryColumn);

  const option = useMemo<EChartsOption>(() => {
    const base = chartBase(dark);
    const categoryAxis = { type: 'category', data: categories };
    const valueAxis = { type: 'value' };
    return {
      ...base,
      tooltip: { ...(base.tooltip as object), trigger: 'axis' },
      xAxis: horizontal
        ? { ...(base.xAxis as object), ...valueAxis }
        : { ...(base.xAxis as object), ...categoryAxis, axisLabel: { rotate: 30 } },
      yAxis: horizontal
        ? { ...(base.yAxis as object), ...categoryAxis }
        : { ...(base.yAxis as object), ...valueAxis },
      series: valueColumns.map((name, index) => ({
        name,
        type: 'bar',
        data: numericColumn(source, name),
        itemStyle: {
          color: seriesColor(index),
          // theme.json says this fill disappears into the page. Stroke it.
          borderColor: needsOutlineOn(index, dark ? 'dark' : 'light')
            ? dark
              ? '#e2e8f0'
              : '#0f172a'
            : 'transparent',
          borderWidth: needsOutlineOn(index, dark ? 'dark' : 'light') ? 1 : 0,
        },
      })),
    };
  }, [categories, dark, horizontal, source, valueColumns]);

  if (!hasRows(source)) {
    return <Absent what="No data for this chart" why="Its source table has not been exported." />;
  }
  return <EChart option={option} ariaLabel={label} caption={caption} height={height} />;
}

// ---------------------------------------------------------------------------
// Scatter
// ---------------------------------------------------------------------------

export function ScatterPlot({
  source,
  xColumn,
  yColumn,
  xName,
  yName,
  label,
  caption,
  height,
}: {
  source?: Source;
  xColumn: string;
  yColumn: string;
  xName: string;
  yName: string;
  label: string;
  caption?: string;
  height?: number;
}) {
  const dark = useDark();
  const data = points(numericColumn(source, xColumn), numericColumn(source, yColumn));
  const outlined = needsOutlineOn(2, dark ? 'dark' : 'light');

  const option = useMemo<EChartsOption>(() => {
    const base = chartBase(dark);
    return {
      ...base,
      xAxis: { ...(base.xAxis as object), type: 'value', name: xName, scale: true },
      yAxis: { ...(base.yAxis as object), type: 'value', name: yName, scale: true },
      series: [
        {
          type: 'scatter',
          data,
          symbolSize: 7,
          itemStyle: {
            color: seriesColor(2),
            opacity: 0.85,
            borderColor: outlined ? (dark ? '#e2e8f0' : '#0f172a') : 'transparent',
            borderWidth: outlined ? 1 : 0,
          },
        },
      ],
    };
  }, [dark, data, outlined, xName, yName]);

  if (!hasRows(source) || data.length === 0) {
    return <Absent what="No data for this scatter" why="Its source table has not been exported." />;
  }
  return <EChart option={option} ariaLabel={label} caption={caption} height={height} />;
}

// ---------------------------------------------------------------------------
// Calibration curve
// ---------------------------------------------------------------------------

export function CalibrationCurve({
  source,
  predictedColumn,
  observedColumn,
  label,
  caption,
  height,
}: {
  source?: Source;
  predictedColumn: string;
  observedColumn: string;
  label: string;
  caption?: string;
  height?: number;
}) {
  const dark = useDark();
  const data = points(numericColumn(source, predictedColumn), numericColumn(source, observedColumn));

  const option = useMemo<EChartsOption>(() => {
    const base = chartBase(dark);
    return {
      ...base,
      tooltip: { ...(base.tooltip as object), trigger: 'axis' },
      xAxis: { ...(base.xAxis as object), type: 'value', min: 0, max: 1, name: 'Predicted' },
      yAxis: { ...(base.yAxis as object), type: 'value', min: 0, max: 1, name: 'Observed' },
      series: [
        {
          name: 'perfect calibration',
          type: 'line',
          data: [
            [0, 0],
            [1, 1],
          ],
          showSymbol: false,
          silent: true,
          lineStyle: { type: 'dashed', width: 1, color: dark ? '#64748b' : '#94a3b8' },
        },
        {
          name: 'observed',
          type: 'line',
          data,
          symbolSize: 7,
          lineStyle: { width: 2, color: seriesColor(3) },
          itemStyle: { color: seriesColor(3) },
        },
      ],
    };
  }, [dark, data]);

  if (!hasRows(source) || data.length === 0) {
    return <Absent what="No calibration curve yet" why={NO_RESULTS} />;
  }
  return <EChart option={option} ariaLabel={label} caption={caption} height={height} />;
}
