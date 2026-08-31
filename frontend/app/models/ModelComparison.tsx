'use client';

import { useMemo, useState } from 'react';

import { CurveChart } from '@/components/charts/CurveChart';
import { EmptyState } from '@/components/ui/States';
import { experiments as generated } from '@/lib/generated/experiments';

/**
 * The metric table, the model selector, and the curve and confusion viewers
 * (T115.3, T115.4).
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
 * ## The confusion matrix is not normalised here
 *
 * Counts render as counts. A percentage computed in the browser would be a
 * client-side metric, which this project forbids, and the row totals are
 * visible anyway. The support note explains why the totals are five times the
 * corpus: the matrix sums element-wise over a repeated 5x5 map.
 */

type SortState = { metric: string; descending: boolean };

export function ModelComparison() {
  const available = generated.experiments.filter((item) => item.available);
  const [expId, setExpId] = useState(available[0]?.exp_id ?? '');

  const experiment = useMemo(
    () => available.find((item) => item.exp_id === expId) ?? available[0],
    [available, expId],
  );

  const metrics = experiment?.metrics ?? [];
  const [sort, setSort] = useState<SortState>({ metric: 'sensitivity', descending: true });
  const [modelId, setModelId] = useState<string>('');

  const models = useMemo(() => {
    const rows = [...(experiment?.models ?? [])];
    const known = metrics.some((item) => item.name === sort.metric);
    const key = known ? sort.metric : (metrics[0]?.name ?? '');
    rows.sort((a, b) => {
      const left = a.metrics[key]?.mean ?? Number.NEGATIVE_INFINITY;
      const right = b.metrics[key]?.mean ?? Number.NEGATIVE_INFINITY;
      return sort.descending ? right - left : left - right;
    });
    return rows;
  }, [experiment, metrics, sort]);

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

  return (
    <div>
      {/* ---------------------------------------------------------------- */}
      <div className="flex flex-wrap items-end gap-4 rounded-lg border border-slate-200 p-4 dark:border-slate-800">
        <label className="flex flex-col gap-1 text-xs">
          <span className="uppercase tracking-widest text-slate-500">Experiment</span>
          <select
            value={experiment.exp_id}
            onChange={(event) => {
              setExpId(event.target.value);
              setModelId('');
            }}
            className="rounded border border-slate-300 bg-white px-2 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900"
          >
            {available.map((item) => (
              <option key={item.exp_id} value={item.exp_id}>
                {item.exp_id} — {item.title}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs">
          <span className="uppercase tracking-widest text-slate-500">Model</span>
          <select
            value={selectedModel}
            onChange={(event) => setModelId(event.target.value)}
            className="rounded border border-slate-300 bg-white px-2 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900"
          >
            {models.map((item) => (
              <option key={item.model_id} value={item.model_id}>
                {item.model_id}
              </option>
            ))}
          </select>
        </label>

        <p className="text-xs text-slate-500">
          {experiment.tuned === true
            ? 'Hyperparameters searched inside each training fold.'
            : experiment.tuned === false
              ? 'Configured defaults; no search runs inside this experiment.'
              : 'The run did not record whether a search ran.'}
        </p>
      </div>

      <p className="mt-3 max-w-3xl text-sm text-slate-600 dark:text-slate-400">
        {experiment.description}
      </p>
      {experiment.caveat ? (
        <p className="mt-2 max-w-3xl rounded border-l-2 border-amber-400 bg-amber-50/60 py-2 pl-3 text-sm text-amber-900 dark:bg-amber-950/30 dark:text-amber-200">
          {experiment.caveat}
        </p>
      ) : null}

      {/* ---------------------------------------------------------------- */}
      <div className="mt-6 overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-widest text-slate-500 dark:bg-slate-900/60">
            <tr>
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
            {models.map((row) => (
              <tr
                key={row.model_id}
                className={
                  row.model_id === selectedModel
                    ? 'border-t border-slate-100 bg-sky-50/60 dark:border-slate-800 dark:bg-sky-950/30'
                    : 'border-t border-slate-100 dark:border-slate-800'
                }
              >
                <td className="px-3 py-1.5 font-medium">{row.model_id}</td>
                {metrics.map((metric) => (
                  <td key={metric.name} className="whitespace-nowrap px-3 py-1.5 tabular-nums">
                    {row.metrics[metric.name]?.display ?? 'n/a'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-500">
        Mean +/- standard deviation across {models[0]?.n_folds_display ?? 'the'} folds, read
        from {experiment.directory}/aggregate_metrics.csv. Sorting reorders rows; it never
        changes what a cell says.
      </p>

      {/* ---------------------------------------------------------------- */}
      {(experiment.models?.[0]?.per_class ?? []).length > 0 ? (
        <section className="mt-8">
          <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
            Per class, for {selectedModel}
          </h3>
          <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase tracking-widest text-slate-500 dark:bg-slate-900/60">
                <tr>
                  <th className="px-3 py-2">Class</th>
                  <th className="px-3 py-2">Recall</th>
                  <th className="px-3 py-2">Precision</th>
                  <th className="px-3 py-2">F1</th>
                  <th className="px-3 py-2">Mean support</th>
                </tr>
              </thead>
              <tbody>
                {(models.find((m) => m.model_id === selectedModel)?.per_class ?? []).map(
                  (row) => {
                    const cell = (key: string) =>
                      (row[key] as { display?: string } | undefined)?.display ?? 'n/a';
                    return (
                      <tr
                        key={String(row.class)}
                        className="border-t border-slate-100 dark:border-slate-800"
                      >
                        <td className="px-3 py-1.5">{String(row.class)}</td>
                        <td className="px-3 py-1.5 tabular-nums">{cell('recall')}</td>
                        <td className="px-3 py-1.5 tabular-nums">{cell('precision')}</td>
                        <td className="px-3 py-1.5 tabular-nums">{cell('f1')}</td>
                        <td className="px-3 py-1.5 tabular-nums">
                          {String(row.support_display ?? 'n/a')}
                        </td>
                      </tr>
                    );
                  },
                )}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-500">
            A macro average says nothing about the thinnest class. Support is on every row
            so the weight behind each number is visible.
          </p>
        </section>
      ) : null}

      {/* ---------------------------------------------------------------- */}
      <section className="mt-10 grid gap-8 lg:grid-cols-2">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
            ROC — {selectedModel}
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
          <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
            Precision-recall — {selectedModel}
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
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-500">
          {curves.aggregation_note}
        </p>
      ) : null}

      {/* ---------------------------------------------------------------- */}
      <section className="mt-10">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
          Confusion matrix — {selectedModel}
        </h3>
        {confusion?.available && confusion.models?.[selectedModel] ? (
          <>
            <div className="mt-3 inline-block overflow-hidden rounded-lg border border-slate-200 dark:border-slate-800">
              <table className="text-sm">
                <thead className="bg-slate-50 text-xs uppercase tracking-widest text-slate-500 dark:bg-slate-900/60">
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
                      className="border-t border-slate-100 dark:border-slate-800"
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
                              : 'px-4 py-2 text-center tabular-nums text-slate-600 dark:text-slate-400'
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
            <p className="mt-2 max-w-3xl text-xs text-slate-500 dark:text-slate-500">
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
    </div>
  );
}
