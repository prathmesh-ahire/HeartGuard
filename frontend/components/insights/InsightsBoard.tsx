'use client';

import { useCallback, useEffect, useMemo, useId, useState, type ReactNode } from 'react';
import { useTheme } from 'next-themes';
import { LazyMotion, m } from 'framer-motion';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { SHOW_SCREENING_NOTICE } from '@/lib/flags';
import { tokenColour } from '@/lib/cssTokens';
import { CARD_HOVER, PRESS_SPRING } from '@/lib/motion';
import { seriesColor, SURFACE, TYPE_SCALE, LAYOUT } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import { ButtonLink } from '@/components/ui/Button';
import { StatTile } from '@/components/ui/StatTile';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { ErrorState, EmptyState } from '@/components/ui/States';
import { InsightsLoading, ServiceUnavailable } from '@/components/ui/PageStates';
import { EChart, chartBase, type EChartsOption } from '@/components/charts/EChart';
import {
  ApiError,
  getInsights,
  type InsightsData,
  type InsightTrendPoint,
  type InsightConfidenceBin,
  type InsightResultSplit,
} from '@/lib/api';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TREND_WINDOW_OPTIONS = [7, 14, 30, 90] as const;
type TrendWindow = (typeof TREND_WINDOW_OPTIONS)[number];

const loadFeatures = () => import('@/components/motion/features').then((mod) => mod.default);

/**
 * This page's cards react the same way the Analyse page's Step 1 check cards
 * do (2026-09-18, replacing a first, CSS-only hover that read as too weak):
 * the same `CARD_HOVER` scale-and-lift on the same `PRESS_SPRING`, not a
 * separate hand-tuned motion. No `whileTap` though, unlike Step 1's own
 * cards -- these are not buttons, and a press state would claim a click
 * these cards do not have.
 */
function InteractiveCard({ className, children }: { className?: string; children: ReactNode }) {
  const reduced = useReducedMotion();
  return (
    <m.div className={cn('rounded-2xl', className)} whileHover={reduced ? undefined : CARD_HOVER} transition={PRESS_SPRING}>
      {children}
    </m.div>
  );
}

const TASKS = prediction.tasks;

/** Browser UTC offset in minutes (UTC+05:30 ? 330). */
function tzOffset(): number {
  return -new Date().getTimezoneOffset();
}

/** Format YYYY-MM-DD as "Sep 15". */
function shortDate(iso: string): string {
  const parts = iso.split('-').map(Number);
  const [yr, mo, dy] = parts as [number, number, number];
  if (!yr || !mo || !dy) return iso;
  return new Date(yr, mo - 1, dy).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
}

function useDark(): boolean {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted && resolvedTheme === 'dark';
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

function TrendChart({ points, days }: { points: InsightTrendPoint[]; days: number }) {
  const dark = useDark();
  const step = Math.max(1, Math.ceil(days / 7));
  const option = useMemo<EChartsOption>(
    () => ({
      ...chartBase(dark),
      tooltip: {
        ...(chartBase(dark).tooltip as object),
        trigger: 'axis',
        formatter: (params: unknown) => {
          const arr = params as { name: string; value: number }[];
          const p = arr[0];
          return p ? `${shortDate(p.name)}: ${String(p.value)}` : '';
        },
      },
      xAxis: {
        ...(chartBase(dark).xAxis as object),
        type: 'category',
        data: points.map((p) => p.date),
        axisLabel: {
          color: tokenColour('ink-3'),
          interval: step - 1,
          formatter: (v: string) => shortDate(v),
          rotate: 30,
        },
      },
      yAxis: {
        ...(chartBase(dark).yAxis as object),
        type: 'value',
        minInterval: 1,
        name: 'Analyses',
      },
      series: [
        {
          name: 'Analyses',
          type: 'bar',
          data: points.map((p) => p.count),
          itemStyle: { color: seriesColor(0), borderRadius: [3, 3, 0, 0] },
        },
      ],
    }),
    [dark, points, step],
  );
  return (
    <EChart
      option={option}
      ariaLabel={`Daily analyses over the last ${String(days)} days`}
      height={240}
    />
  );
}

function ConfidenceChart({ bins }: { bins: InsightConfidenceBin[] }) {
  const dark = useDark();
  const option = useMemo<EChartsOption>(
    () => ({
      ...chartBase(dark),
      tooltip: {
        ...(chartBase(dark).tooltip as object),
        trigger: 'axis',
        formatter: (params: unknown) => {
          const arr = params as { name: string; value: number }[];
          const p = arr[0];
          return p ? `${p.name}: ${String(p.value)}` : '';
        },
      },
      xAxis: {
        ...(chartBase(dark).xAxis as object),
        type: 'category',
        data: bins.map((b) => b.label),
        name: 'Confidence',
        axisLabel: { color: tokenColour('ink-3'), rotate: 30 },
      },
      yAxis: {
        ...(chartBase(dark).yAxis as object),
        type: 'value',
        minInterval: 1,
        name: 'Count',
      },
      series: [
        {
          name: 'Recordings',
          type: 'bar',
          data: bins.map((b) => b.count),
          itemStyle: { color: seriesColor(1), borderRadius: [3, 3, 0, 0] },
        },
      ],
    }),
    [dark, bins],
  );
  return (
    <EChart
      option={option}
      ariaLabel="Confidence score distribution across all analyses"
      height={240}
    />
  );
}

function OutcomeSplitChart({ splits }: { splits: InsightResultSplit[] }) {
  const dark = useDark();
  const allClasses = useMemo(
    () => Array.from(new Set(splits.flatMap((s) => s.classes.map((c) => c.class)))),
    [splits],
  );
  const option = useMemo<EChartsOption>(
    () => ({
      ...chartBase(dark),
      tooltip: {
        ...(chartBase(dark).tooltip as object),
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      legend: {
        ...(chartBase(dark).legend as object),
        data: allClasses,
        top: 0,
      },
      grid: { left: 140, right: 20, top: 36, bottom: 28, containLabel: false },
      xAxis: {
        ...(chartBase(dark).xAxis as object),
        type: 'value',
        max: 100,
        axisLabel: {
          color: tokenColour('ink-3'),
          formatter: (v: number) => `${String(v)}%`,
        },
      },
      yAxis: {
        ...(chartBase(dark).yAxis as object),
        type: 'category',
        data: splits.map((s) => s.title),
        axisLabel: {
          color: tokenColour('ink-3'),
          width: 130,
          overflow: 'truncate',
        },
      },
      series: allClasses.map((cls, idx) => ({
        name: cls,
        type: 'bar',
        stack: 'total',
        data: splits.map((s) => {
          const entry = s.classes.find((c) => c.class === cls);
          return entry?.percent ?? 0;
        }),
        itemStyle: { color: seriesColor(idx) },
      })),
    }),
    [dark, splits, allClasses],
  );
  const height = Math.max(180, splits.length * 52 + 64);
  return (
    <EChart option={option} ariaLabel="Outcome distribution by check type" height={height} />
  );
}

// ---------------------------------------------------------------------------
// Model reliability (T133.4) — text only, from generated/prediction
// ---------------------------------------------------------------------------

function ReliabilitySummary() {
  return (
    <section aria-label="Model reliability summary" className={LAYOUT.section}>
      <SectionHeader title="Model reliability" />
      {SHOW_SCREENING_NOTICE ? (
        <p className={cn(TYPE_SCALE.body, SURFACE.muted)}>{prediction.disclaimer}</p>
      ) : null}
      <div className="space-y-3">
        {TASKS.map((task) => (
          <InteractiveCard key={task.task} className={cn(SURFACE.card, 'p-4')}>
            <p className={cn(TYPE_SCALE.caption, 'font-semibold text-ink')}>{task.title}</p>
            <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>{task.description}</p>
          </InteractiveCard>
        ))}
      </div>
      <div className={cn(SURFACE.sunken, 'rounded-2xl p-4 space-y-1')}>
        <p className={cn(TYPE_SCALE.caption, 'font-semibold text-ink')}>Low-confidence flag</p>
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>{prediction.low_confidence.note}</p>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Filters (T133.5) — drive every chart together
// ---------------------------------------------------------------------------

const INPUT_CLS =
  'rounded-lg border border-line bg-panel px-3 py-1.5 text-body-sm text-ink';

function FiltersBar({
  task,
  days,
  onTask,
  onDays,
}: {
  task: string;
  days: TrendWindow;
  onTask: (t: string) => void;
  onDays: (d: TrendWindow) => void;
}) {
  const baseId = useId();
  return (
    <section
      aria-label="Insights filters"
      className="flex flex-wrap items-end gap-4 rounded-2xl border border-line bg-sunken p-4"
    >
      <div>
        <label htmlFor={baseId + 'task'} className="label-micro mb-1 block">
          Check type
        </label>
        <select
          id={baseId + 'task'}
          value={task}
          onChange={(e) => onTask(e.target.value)}
          className={INPUT_CLS}
        >
          <option value="">All checks</option>
          {TASKS.map((t) => (
            <option key={t.task} value={t.task}>
              {t.title}
            </option>
          ))}
        </select>
      </div>
      <div>
        <label htmlFor={baseId + 'days'} className="label-micro mb-1 block">
          Trend window
        </label>
        <select
          id={baseId + 'days'}
          value={days}
          onChange={(e) => onDays(Number(e.target.value) as TrendWindow)}
          className={INPUT_CLS}
        >
          {TREND_WINDOW_OPTIONS.map((d) => (
            <option key={d} value={d}>
              Last {d} days
            </option>
          ))}
        </select>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// InsightsBoard — main export
// ---------------------------------------------------------------------------

type LoadState =
  | { status: 'loading' }
  | { status: 'ready'; data: InsightsData }
  | { status: 'failed'; error: ApiError | Error };

/**
 * Phase 133 — Insights page (T133.1–T133.6).
 *
 * All numeric text comes from the API's display strings; the component only
 * positions chart marks from raw integer counts.
 */
export function InsightsBoard() {
  const [task, setTask] = useState('');
  const [days, setDays] = useState<TrendWindow>(30);
  const [load, setLoad] = useState<LoadState>({ status: 'loading' });
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let active = true;
    setLoad((prev) => (prev.status === 'ready' ? prev : { status: 'loading' }));
    getInsights({ task: task || undefined, days, tzOffsetMinutes: tzOffset() })
      .then((data) => {
        if (active) setLoad({ status: 'ready', data });
      })
      .catch((err: unknown) => {
        if (active)
          setLoad({
            status: 'failed',
            error: err instanceof Error ? err : new Error(String(err)),
          });
      });
    return () => {
      active = false;
    };
  }, [task, days, reload]);

  const retry = useCallback(() => setReload((v) => v + 1), []);

  const onTask = useCallback((t: string) => {
    setTask(t);
    setLoad({ status: 'loading' });
  }, []);

  const onDays = useCallback((d: TrendWindow) => {
    setDays(d);
    setLoad({ status: 'loading' });
  }, []);

  const data = load.status === 'ready' ? load.data : null;
  const isEmpty = data !== null && data.total === 0 && !task;

  return (
    <LazyMotion features={loadFeatures} strict>
    <div className="space-y-10">
      {/* T133.5: Filters drive every chart on the page together. */}
      <FiltersBar task={task} days={days} onTask={onTask} onDays={onDays} />

      {/* Service notice from TinyDB store */}
      {data?.notice ? (
        <p
          role="note"
          className={cn(
            TYPE_SCALE.caption,
            'rounded border border-warn-line bg-warn-soft p-3 text-warn',
          )}
        >
          {data.notice}
        </p>
      ) : null}

      {load.status === 'loading' ? <InsightsLoading /> : null}

      {load.status === 'failed' ? (
        load.error instanceof ApiError && load.error.offline ? (
          <ServiceUnavailable onRetry={retry} />
        ) : (
          <ErrorState
            title="Insights could not be loaded"
            detail={load.error.message}
            onRetry={retry}
          />
        )
      ) : null}

      {/* T133.6: No-history empty state — clear prompt, not empty charts. */}
      {load.status === 'ready' && isEmpty ? (
        <EmptyState
          icon="insights"
          title="No trends to show yet"
          description="Insights are drawn from the analyses in your History. Analyse a few recordings and the trends appear here."
          action={
            <ButtonLink href="/" tone="primary" icon="pulse">
              Analyse a recording
            </ButtonLink>
          }
        />
      ) : null}

      {/* Filtered result empty but history exists */}
      {load.status === 'ready' && !isEmpty && data !== null && data.total === 0 ? (
        <EmptyState
          icon="features"
          title="No analyses match this filter"
          description="There are analyses in History but none used this check type. Try a different filter."
        />
      ) : null}

      {/* T133.1: Summary tiles */}
      {data !== null && data.total > 0 ? (
        <section aria-label="Summary" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatTile
            label="Total analyses"
            display={data.total_display}
            value={data.total}
            animate
            marked
          />
          <StatTile
            label={`Analyses — last ${String(days)} days`}
            display={data.trend.in_window_display}
            value={data.trend.in_window}
            animate
          />
          {(() => {
            const topSplit = data.result_split[0];
            const topClass = topSplit?.classes.reduce<(typeof topSplit.classes)[number] | null>(
              (best, c) => (c.count > (best?.count ?? -1) ? c : best),
              null,
            );
            return topClass && topSplit ? (
              <StatTile
                label={`Most common result — ${topSplit.title}`}
                display={topClass.class}
                hint={`${topClass.percent_display} of ${topSplit.title}`}
              />
            ) : null;
          })()}
          <StatTile
            label="Low-confidence analyses"
            display={data.low_confidence.percent_display}
            hint={`${data.low_confidence.count_display} of ${data.total_display}`}
          />
        </section>
      ) : null}

      {/* T133.2: Trend chart */}
      {data !== null && data.total > 0 ? (
        <section aria-label="Trend over time" className={LAYOUT.section}>
          <SectionHeader title={`Analyses over the last ${String(days)} days`} />
          {data.trend.points.some((p) => p.count > 0) ? (
            <InteractiveCard className={cn(SURFACE.card, 'p-4')}>
              <TrendChart points={data.trend.points} days={days} />
            </InteractiveCard>
          ) : (
            <div className={cn(SURFACE.sunken, 'p-6')}>
              <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
                No analyses in this window. Widen the trend window or analyse more recordings.
              </p>
            </div>
          )}
        </section>
      ) : null}

      {/* T133.3: Confidence distribution + outcome split */}
      {data !== null && data.total > 0 ? (
        <div className="grid gap-6 lg:grid-cols-2">
          <section aria-label="Results by check type" className={LAYOUT.section}>
            <SectionHeader title="Results by check type" />
            {data.result_split.length > 0 ? (
              <InteractiveCard className={cn(SURFACE.card, 'p-4')}>
                <OutcomeSplitChart splits={data.result_split} />
              </InteractiveCard>
            ) : (
              <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'p-4')}>
                No result breakdown available for the current filter.
              </p>
            )}
          </section>

          <section aria-label="Confidence distribution" className={LAYOUT.section}>
            <SectionHeader title="Confidence distribution" />
            {data.confidence_distribution.n > 0 ? (
              <InteractiveCard className={cn(SURFACE.card, 'p-4')}>
                <ConfidenceChart bins={data.confidence_distribution.bins} />
              </InteractiveCard>
            ) : (
              <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'p-4')}>
                No confidence data yet.
              </p>
            )}
          </section>
        </div>
      ) : null}

      {/* T133.4: Model reliability — read from generated/, never typed */}
      <ReliabilitySummary />
    </div>
    </LazyMotion>
  );
}
