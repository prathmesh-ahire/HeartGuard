'use client';

import { useMemo, useState } from 'react';

import { CurveChart } from '@/components/charts/CurveChart';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/cn';
import { Disclosure } from '@/components/ui/Disclosure';
import { FilterPills } from '@/components/ui/FilterPills';
import { EmptyState } from '@/components/ui/States';
import { experiments as generated } from '@/lib/generated/experiments';

/**
 * The leaderboard, the model selector, and the curve and confusion viewers
 * (T115.3, T115.4; leaderboard, task pills and numbered charts added T142).
 *
 * ## Sorting is a view, not a computation
 *
 * Rows sort on the numeric `mean` a payload already carries. The cell that
 * renders is always `display`, the string Python formatted; the number exists so
 * a comparator has something to compare. Sorting never changes what a cell says.
 *
 * ## Direction is data, not an assumption
 *
 * `higher_is_better` travels with each metric because it is false for two of
 * them — Brier score and calibration error — and a table that sorted every
 * column descending would present the worst-calibrated model as the best. The
 * arrow and the ordering both read that flag.
 *
 * ## "Best" is a rank, not a claim (T142.1)
 *
 * Rank 1 is whichever row the current sort already put first — the table's own
 * ordering, just labelled. Never "SOTA": nothing here was benchmarked against a
 * published result, so the badge says "Best" (best among these declared runs,
 * by the metric currently sorted) and nothing stronger.
 *
 * ## Task first, run second (T142.2, T142.6)
 *
 * The old flat "Experiment" dropdown mixed three binary re-runs with one
 * PASCAL A and one PASCAL B run in one list, so switching "experiment" and
 * switching "label space" looked like the same action. `FilterPills` now
 * switches the task (a pill per distinct label space among the available
 * experiments, sliding highlight, no `layoutId`) and a plain `<select>` only
 * appears underneath it when that task has more than one declared run to
 * choose between — which is exactly when a second control earns its place.
 * The rest (curves, confusion matrix, per-class cards) is the same live
 * component reacting to the new value, not a static figure switched by CSS.
 *
 * ## The confusion matrix is not normalised here
 *
 * Counts render as counts. A percentage computed in the browser would be a
 * client-side metric, which this project forbids, and the row totals are
 * visible anyway. The support note explains why the totals are five times the
 * corpus: the matrix sums element-wise over a repeated 5x5 map.
 *
 * ## Per-class cards, not a raw table (T142.3)
 *
 * Recall/precision/F1 render as small proportional bars sized from the same
 * `mean` (0..1) the old table's `display` string was formatted from — the bar
 * is geometry only, the printed number is still the server's string.
 *
 * ## Numbered so the tab reads as complete (T142.4)
 *
 * Per-class, ROC, precision-recall and the confusion matrix are the four
 * visual sections this component can show for one model; each is labelled
 * "Graph N of 4" so a reader who has seen three of them knows there is
 * exactly one left, not an unknown number.
 */

type SortState = { metric: string; descending: boolean };

const TASK_LABELS: Record<string, string> = {
  binary: 'Binary',
  pascal_a: 'PASCAL A',
  pascal_b: 'PASCAL B',
  murmur: 'Murmur',
  outcome: 'Outcome',
};

function taskLabel(task: string): string {
  return TASK_LABELS[task] ?? task;
}

const GRAPH_TOTAL = 4;

type PerClassMetric = { label?: string; mean?: number; display?: string };
type PerClassRow = Record<string, unknown> & {
  class?: string;
  support_display?: string;
};

export function ModelComparison() {
  const available = generated.experiments.filter((item) => item.available);

  const tasks = useMemo(() => {
    const seen = new Set<string>();
    const ordered: string[] = [];
    for (const item of available) {
      if (!seen.has(item.task)) {
        seen.add(item.task);
        ordered.push(item.task);
      }
    }
    return ordered;
  }, [available]);

  const [task, setTask] = useState(tasks[0] ?? '');
  const experimentsForTask = useMemo(
    () => available.filter((item) => item.task === task),
    [available, task],
  );
  const [expId, setExpId] = useState(experimentsForTask[0]?.exp_id ?? '');

  const experiment = useMemo(
    () => experimentsForTask.find((item) => item.exp_id === expId) ?? experimentsForTask[0],
    [experimentsForTask, expId],
  );

  const metrics = experiment?.metrics ?? [];
  const [sort, setSort] = useState<SortState>({ metric: 'sensitivity', descending: true });
  const [modelId, setModelId] = useState<string>('');

  const sortKey = metrics.some((item) => item.name === sort.metric)
    ? sort.metric
    : (metrics[0]?.name ?? '');
  const sortMetric = metrics.find((item) => item.name === sortKey);

  const models = useMemo(() => {
    const rows = [...(experiment?.models ?? [])];
    rows.sort((a, b) => {
      const left = a.metrics[sortKey]?.mean ?? Number.NEGATIVE_INFINITY;
      const right = b.metrics[sortKey]?.mean ?? Number.NEGATIVE_INFINITY;
      return sort.descending ? right - left : left - right;
    });
    return rows;
  }, [experiment, sortKey, sort.descending]);

  const selectedModel = modelId !== '' ? modelId : (models[0]?.model_id ?? '');

  if (experiment === undefined) {
    return (
      <EmptyState
        title="No experiment has produced results yet"
        description="Every declared experiment reported its own reason; see the list below."
      />
    );
  }

  const confusion = experiment.confusion;
  const curves = experiment.curves;
  const curveModel = (curves?.models ?? []).find(
    (item) => (item as { model_id?: string }).model_id === selectedModel,
  ) as
    | Record<string, { x: number[]; x_display: string[]; mean: number[]; sd: number[] } | undefined>
    | undefined;
  const perClassRows = (models.find((m) => m.model_id === selectedModel)?.per_class ??
    []) as PerClassRow[];

  return (
    <div>
      {/* ---------------------------------------------------------------- */}
      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-line bg-panel px-4 py-3 shadow-panel">
        <div className="flex flex-col gap-1.5">
          <span className="label-micro">Task</span>
          <FilterPills
            ariaLabel="Which label space"
            options={tasks.map((item) => ({ id: item, label: taskLabel(item) }))}
            value={task}
            onChange={(next) => {
              setTask(next);
              const first = available.find((item) => item.task === next);
              setExpId(first?.exp_id ?? '');
              setModelId('');
            }}
          />
        </div>

        {experimentsForTask.length > 1 ? (
          <label className="flex flex-col gap-1.5">
            <span className="label-micro">Run</span>
            <select
              value={experiment.exp_id}
              onChange={(event) => {
                setExpId(event.target.value);
                setModelId('');
              }}
              className="rounded border border-line bg-sunken px-2 py-1.5 text-label-md text-ink focus:border-accent focus:outline-none"
            >
              {experimentsForTask.map((item) => (
                <option key={item.exp_id} value={item.exp_id}>
                  {item.exp_id} — {item.title}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        <label className="flex flex-col gap-1.5">
          <span className="label-micro">Model</span>
          <select
            value={selectedModel}
            onChange={(event) => setModelId(event.target.value)}
            className="rounded border border-line bg-sunken px-2 py-1.5 text-label-md text-ink focus:border-accent focus:outline-none"
          >
            {models.map((item) => (
              <option key={item.model_id} value={item.model_id}>
                {item.model_id}
              </option>
            ))}
          </select>
        </label>

        <p className="text-xs text-ink-3">
          {experiment.tuned === true
            ? 'Hyperparameters searched inside each training fold.'
            : experiment.tuned === false
              ? 'Configured defaults; no search runs inside this experiment.'
              : 'The run did not record whether a search ran.'}
        </p>
      </div>

      <p className="mt-3 max-w-3xl text-sm text-ink-2">
        {experiment.description}
      </p>
      {experiment.caveat ? (
        <p className="mt-2 max-w-3xl rounded border-l-2 border-warn-line bg-warn-soft py-2 pl-3 text-sm text-warn">
          {experiment.caveat}
        </p>
      ) : null}

      {/* ---------------------------------------------------------------- */}
      <div className="mt-6 overflow-x-auto rounded-lg border border-line">
        <table className="min-w-full text-left text-sm">
          <thead className="border-b-2 border-line bg-sunken text-label-sm uppercase text-ink-3">
            <tr>
              <th scope="col" className="px-3 py-2 text-center">
                Rank
              </th>
              <th scope="col" className="px-3 py-2">
                Model
              </th>
              {metrics.map((metric) => (
                <th key={metric.name} scope="col" className="px-3 py-2">
                  <button
                    type="button"
                    onClick={() =>
                      setSort((current) =>
                        current.metric === metric.name
                          ? { metric: metric.name, descending: !current.descending }
                          : { metric: metric.name, descending: metric.higher_is_better },
                      )
                    }
                    className="whitespace-nowrap uppercase tracking-widest hover:underline"
                  >
                    {metric.label}
                    {sort.metric === metric.name ? (sort.descending ? ' ↓' : ' ↑') : ''}
                    {metric.higher_is_better ? '' : ' (lower is better)'}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((row, index) => {
              const rank = index + 1;
              const best = rank === 1;
              return (
                <tr
                  key={row.model_id}
                  className={cn(
                    'border-t border-line',
                    best ? 'bg-good-soft/50' : row.model_id === selectedModel ? 'bg-accent-soft/60' : undefined,
                  )}
                >
                  <td className="px-3 py-1.5 text-center font-mono tabular-nums text-ink-3">{rank}</td>
                  <td className="px-3 py-1.5 font-medium">
                    <span className="flex items-center gap-2">
                      {row.model_id}
                      {best ? <Badge tone="good">Best</Badge> : null}
                    </span>
                  </td>
                  {metrics.map((metric) => (
                    <td key={metric.name} className="whitespace-nowrap px-3 py-1.5 tabular-nums">
                      {row.metrics[metric.name]?.display ?? 'n/a'}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-ink-3">
        Mean +/- standard deviation across {models[0]?.n_folds_display ?? 'the'} folds. Ranked by{' '}
        {sortMetric?.label ?? 'the sorted column'}; sorting reorders rows, it never changes what a
        cell says.
      </p>

      {/* ---------------------------------------------------------------- */}
      <Disclosure
        className="mt-8"
        summary={<>Per-class breakdown, ROC / precision-recall curves and confusion matrix for {selectedModel}</>}
      >
      {perClassRows.length > 0 ? (
        <section>
          <h3 className="label-micro">
            Graph 1 of {GRAPH_TOTAL} — Per class, for {selectedModel}
          </h3>
          <ul className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {perClassRows.map((row) => {
              const cls = String(row.class ?? '');
              const metricCell = (key: string) => row[key] as PerClassMetric | undefined;
              return (
                <li key={cls} className="rounded-xl border border-line bg-panel p-4 shadow-panel">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-label-md uppercase text-ink">{cls}</span>
                    <span className="text-xs text-ink-3">support {row.support_display ?? 'n/a'}</span>
                  </div>
                  <ul className="mt-3 space-y-2">
                    {(['recall', 'precision', 'f1'] as const).map((key) => {
                      const entry = metricCell(key);
                      const width = typeof entry?.mean === 'number' ? entry.mean * 100 : 0;
                      return (
                        <li key={key}>
                          <div className="flex items-baseline justify-between text-xs text-ink-2">
                            <span>{entry?.label ?? key}</span>
                            <span className="font-mono tabular-nums">{entry?.display ?? 'n/a'}</span>
                          </div>
                          <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full border border-line bg-sunken">
                            <div className="h-full rounded-full bg-accent" style={{ width: width + '%' }} />
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                </li>
              );
            })}
          </ul>
          <p className="mt-2 text-xs text-ink-3">
            A macro average says nothing about the thinnest class. Support is on every card so
            the weight behind each number is visible.
          </p>
        </section>
      ) : null}

      {/* ---------------------------------------------------------------- */}
      <section className="mt-10 grid gap-8 lg:grid-cols-2">
        <div>
          <h3 className="label-micro">
            Graph 2 of {GRAPH_TOTAL} — ROC, {selectedModel}
          </h3>
          {curves?.available && curveModel?.roc ? (
            <CurveChart
              className="mt-3"
              x={curveModel.roc.x}
              xDisplay={curveModel.roc.x_display}
              mean={curveModel.roc.mean}
              sd={curveModel.roc.sd}
              xName="False positive rate"
              yName="True positive rate"
              diagonal
              label={'Mean ROC across folds for ' + selectedModel}
            />
          ) : (
            <EmptyState
              className="mt-3"
              title="No ROC for this selection"
              description={curves?.reason ?? 'This experiment produced no curve points.'}
            />
          )}
        </div>
        <div>
          <h3 className="label-micro">
            Graph 3 of {GRAPH_TOTAL} — Precision-recall, {selectedModel}
          </h3>
          {curves?.available && curveModel?.pr ? (
            <CurveChart
              className="mt-3"
              x={curveModel.pr.x}
              xDisplay={curveModel.pr.x_display}
              mean={curveModel.pr.mean}
              sd={curveModel.pr.sd}
              xName="Recall"
              yName="Precision"
              label={'Mean precision-recall across folds for ' + selectedModel}
            />
          ) : (
            <EmptyState
              className="mt-3"
              title="No precision-recall curve for this selection"
              description={curves?.reason ?? 'This experiment produced no curve points.'}
            />
          )}
        </div>
      </section>
      {curves?.available ? (
        <p className="mt-3 text-xs text-ink-3">
          {curves.aggregation_note}
        </p>
      ) : null}

      {/* ---------------------------------------------------------------- */}
      <section className="mt-10">
        <h3 className="label-micro">
          Graph 4 of {GRAPH_TOTAL} — Confusion matrix, {selectedModel}
        </h3>
        {confusion?.available && confusion.models?.[selectedModel] ? (
          <>
            <div className="mt-3 inline-block overflow-hidden rounded-lg border border-line">
              <table className="text-sm">
                <thead className="border-b-2 border-line bg-sunken text-label-sm uppercase text-ink-3">
                  <tr>
                    <th className="px-3 py-2 text-left">True \ Predicted</th>
                    {(confusion.class_names ?? []).map((name) => (
                      <th key={name} className="px-4 py-2">
                        {name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {confusion.models[selectedModel].total.map((row, index) => (
                    <tr
                      key={(confusion.class_names ?? [])[index] ?? index}
                      className="border-t border-line"
                    >
                      <th scope="row" className="px-3 py-2 text-left font-medium">
                        {(confusion.class_names ?? [])[index] ?? index}
                      </th>
                      {row.map((cell, column) => (
                        <td
                          key={column}
                          className={
                            index === column
                              ? 'px-4 py-2 text-center tabular-nums font-semibold'
                              : 'px-4 py-2 text-center tabular-nums text-ink-2'
                          }
                        >
                          {cell.toLocaleString('en-US')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 max-w-3xl text-xs text-ink-3">
              {confusion.note}
            </p>
          </>
        ) : (
          <EmptyState
            className="mt-3"
            title="No confusion matrix for this selection"
            description={confusion?.reason ?? 'This run recorded no confusion matrices.'}
          />
        )}
      </section>
      </Disclosure>
    </div>
  );
}
