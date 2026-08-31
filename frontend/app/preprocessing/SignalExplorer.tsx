'use client';

import { useMemo, useState } from 'react';

import { SignalChart } from '@/components/charts/SignalChart';
import { EmptyState } from '@/components/ui/States';
import { preprocessingExamples as examples } from '@/lib/generated/signals';

/**
 * The record picker and the filter / normalization toggles (T114.4, T114.5).
 *
 * ## Every state on screen was computed in Python
 *
 * The toggles do not filter anything. They select between four series that
 * `src/reporting/signals.py` produced by calling the pipeline's own
 * `filter_signal` and `normalize_signal` — the same functions the corpus was
 * preprocessed with. A Butterworth filter written in TypeScript would be a
 * second answer to the pipeline's most consequential step, and the reader would
 * believe the one on screen because it is the one they can see.
 *
 * ## Why raw and processed are drawn on separate charts by default
 *
 * A raw PhysioNet waveform and a z-scored one differ by three orders of
 * magnitude. Overlaying them on a shared axis flattens the raw trace into a
 * line at zero, which reads as "the filter removed the signal". The overlay is
 * available, and it is off by default; when it is on, the axis is shared and the
 * caption says so.
 *
 * Quality indicators are the record's own measurements from
 * `signal_quality_flags.csv`, formatted in Python. None of them is scored,
 * graded or turned into "good"/"poor" — a threshold that renames a number as a
 * verdict is a clinical-sounding call this dashboard has no basis for making.
 */

const STATE_LABELS: Record<string, string> = {
  raw: 'Raw',
  filtered: 'Band-pass filtered',
  normalized: 'Normalized',
  filtered_normalized: 'Filtered and normalized',
};

/** The label for a state key, never `undefined`: an unlabelled chart is worse
 *  than a plainly named one, and the export declares exactly these four keys. */
function stateLabel(key: string): string {
  return STATE_LABELS[key] ?? key;
}

export function SignalExplorer() {
  const [recordKey, setRecordKey] = useState(examples.records[0]?.key ?? '');
  const [useFilter, setUseFilter] = useState(true);
  const [useNormalize, setUseNormalize] = useState(true);
  const [overlay, setOverlay] = useState(false);

  const record = useMemo(
    () => examples.records.find((item) => item.key === recordKey) ?? examples.records[0],
    [recordKey],
  );

  if (!examples.available || record === undefined) {
    return (
      <EmptyState
        title="No preprocessing example was exported"
        description={
          examples.reason ??
          'Run scripts/17_export_frontend_data.py on a machine holding the corpus, or commit outputs/02_preprocessing/preprocessing_examples.csv.'
        }
      />
    );
  }

  const selected =
    useFilter && useNormalize
      ? 'filtered_normalized'
      : useFilter
        ? 'filtered'
        : useNormalize
          ? 'normalized'
          : 'raw';

  const selectedSeries = record.series[selected] ?? [];
  const rawSeries = record.series.raw ?? [];

  return (
    <div>
      {/* ---------------------------------------------------------------- */}
      <div className="flex flex-wrap items-end gap-4 rounded-lg border border-slate-200 p-4 dark:border-slate-800">
        <label className="flex flex-col gap-1 text-xs">
          <span className="uppercase tracking-widest text-slate-500">Recording</span>
          <select
            value={record.key}
            onChange={(event) => setRecordKey(event.target.value)}
            className="rounded border border-slate-300 bg-white px-2 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900"
          >
            {examples.records.map((item) => (
              <option key={item.key} value={item.key}>
                {item.title}
              </option>
            ))}
          </select>
        </label>

        <fieldset className="flex items-center gap-4">
          <legend className="sr-only">Preprocessing stages</legend>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useFilter}
              onChange={(event) => setUseFilter(event.target.checked)}
              className="h-4 w-4"
            />
            Band-pass filter
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useNormalize}
              onChange={(event) => setUseNormalize(event.target.checked)}
              className="h-4 w-4"
            />
            Normalize
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={overlay}
              onChange={(event) => setOverlay(event.target.checked)}
              className="h-4 w-4"
            />
            Overlay raw
          </label>
        </fieldset>
      </div>

      <p className="mt-3 max-w-3xl text-sm text-slate-600 dark:text-slate-400">{record.note}</p>

      {/* ---------------------------------------------------------------- */}
      {overlay ? (
        <div className="mt-6">
          <SignalChart
            time={record.time_sec}
            series={[
              { name: stateLabel('raw'), values: rawSeries, colorIndex: 0 },
              { name: stateLabel(selected), values: selectedSeries, colorIndex: 1 },
            ]}
            label={'Raw and ' + stateLabel(selected).toLowerCase() + ' waveform, shared axis'}
            caption="One shared amplitude axis. A raw waveform and a normalized one differ by orders of magnitude, so on this axis the smaller of the two is close to flat — that is the axis, not the signal."
            height={320}
          />
        </div>
      ) : (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <SignalChart
            time={record.time_sec}
            series={[{ name: stateLabel('raw'), values: rawSeries, colorIndex: 0 }]}
            label="Raw waveform"
            caption="As read from the file, after mono collapse and resampling to the working rate."
          />
          <SignalChart
            time={record.time_sec}
            series={[{ name: stateLabel(selected), values: selectedSeries, colorIndex: 1 }]}
            label={stateLabel(selected) + ' waveform'}
            caption={
              selected === 'raw'
                ? 'Both stages off: this is the same series as the panel beside it.'
                : 'Each axis is scaled to its own series, so both traces are visible.'
            }
          />
        </div>
      )}

      {/* ---------------------------------------------------------------- */}
      <section className="mt-8">
        <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">
          Quality indicators for this recording
        </h3>
        <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {record.quality.map((item) => (
            <div
              key={item.name}
              className="rounded border border-slate-200 p-3 dark:border-slate-800"
            >
              <dt className="text-xs text-slate-500">{item.label}</dt>
              <dd className="mt-0.5 tabular-nums font-medium">{item.display}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-500">
          Measured during the preprocessing run and read from
          outputs/02_preprocessing/signal_quality_flags.csv. Reported as measurements,
          never graded: no threshold here turns a number into a verdict.
        </p>
      </section>

      {/* ---------------------------------------------------------------- */}
      <p className="mt-6 text-xs text-slate-500 dark:text-slate-500">
        Record <span className="font-mono">{record.record_uid}</span> · native{' '}
        {record.native_fs} Hz, shown at the working rate of {record.fs} Hz ·{' '}
        {examples.window_seconds} s window · {record.n_points} points, every{' '}
        {record.stride}
        <sup>th</sup> sample. {examples.note}
      </p>
    </div>
  );
}
