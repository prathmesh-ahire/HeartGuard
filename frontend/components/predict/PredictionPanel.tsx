'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States';
import { FileUpload, type UploadPhase } from '@/components/ui/FileUpload';
import { ResultCard } from '@/components/predict/ResultCard';
import { WaveformPreview } from '@/components/predict/WaveformPreview';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import { rememberPrediction } from '@/lib/lastPrediction';
import type { GeneratedSample, GeneratedTaskSpec } from '@/lib/generated/types';
import {
  ApiError,
  predictFile,
  predictSample,
  sampleAudioUrl,
  samples as fetchSamples,
  tasks as fetchTasks,
  type PredictResult,
  type SampleStatus,
  type TaskStatus,
} from '@/lib/api';

/**
 * Upload or pick a sample, score it, show the result (T116.1, T116.5, T116.6).
 *
 * One component behind all three prediction pages. The pages differ only in
 * which label spaces they offer, and building three of these would guarantee
 * that the low-confidence warning or the disclaimer eventually appeared on two
 * of them.
 *
 * ## Availability is a runtime question, asked at runtime
 *
 * `generated/prediction.json` records whether each task had a saved model when
 * the export ran. That is a build-time fact and it goes stale the moment a model
 * is saved, so the panel re-reads `GET /tasks` on mount and prefers the live
 * answer. A task with no model renders the reason the API gives — not an empty
 * form that fails on submit, and not a silent absence.
 *
 * ## The samples are the operator's own corpus, not shipped audio
 *
 * `GET /samples` says which pinned recordings this process can reach. None are
 * committed: the PhysioNet and PASCAL copies carry no licence file, so the audio
 * stays in the read-only `dataset/` folder it came from. Where the corpus is
 * absent the sample list explains that rather than disappearing.
 *
 * ## Nothing is rounded here
 *
 * Every number rendered downstream comes from `result.display.*`, formatted in
 * Python by the same function that formatted the precomputed tables.
 */

export interface PanelTask {
  task: string;
  /** Short label for the task selector. Longer text comes from the payload. */
  label: string;
}

function taskSpec(name: string): GeneratedTaskSpec | undefined {
  return prediction.tasks.find((entry) => entry.task === name);
}

function samplesFor(name: string): GeneratedSample[] {
  return prediction.samples.filter((entry) => entry.tasks.includes(name));
}

export function PredictionPanel({
  offered,
  className,
}: {
  /** The label spaces this page offers. Never merged into one selector. */
  offered: readonly PanelTask[];
  className?: string;
}) {
  const [task, setTask] = useState(offered[0]?.task ?? 'binary');
  const [live, setLive] = useState<TaskStatus[] | null>(null);
  const [servable, setServable] = useState<SampleStatus[] | null>(null);
  const [probeFailed, setProbeFailed] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [chosenSample, setChosenSample] = useState<string | null>(null);
  const [phase, setPhase] = useState<UploadPhase>('idle');
  const [result, setResult] = useState<PredictResult | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let disposed = false;
    void (async () => {
      try {
        const [taskRows, sampleRows] = await Promise.all([fetchTasks(), fetchSamples()]);
        if (disposed) return;
        setLive(taskRows);
        setServable(sampleRows);
        setProbeFailed(null);
      } catch (error) {
        if (disposed) return;
        setProbeFailed(error instanceof ApiError ? error.message : String(error));
      }
    })();
    return () => {
      disposed = true;
    };
  }, []);

  const spec = taskSpec(task);
  const liveStatus = live?.find((entry) => entry.task === task) ?? null;
  const available = liveStatus?.available ?? spec?.model_available_at_export ?? false;
  const unavailableReason =
    liveStatus?.reason ?? spec?.reason_at_export ?? 'No model is deployed for this task.';

  const pageSamples = useMemo(() => samplesFor(task), [task]);
  const servableIds = useMemo(
    () => new Set((servable ?? []).filter((row) => row.available).map((row) => row.sample_id)),
    [servable],
  );
  const activeSample = pageSamples.find((entry) => entry.sample_id === chosenSample) ?? null;

  const reset = useCallback(() => {
    setResult(null);
    setFailure(null);
    setPhase('idle');
  }, []);

  const run = useCallback(async () => {
    setBusy(true);
    setFailure(null);
    setResult(null);
    setPhase('uploading');
    try {
      const outcome =
        chosenSample !== null
          ? await predictSample(chosenSample, task)
          : file !== null
            ? await predictFile(file, task)
            : null;
      if (outcome === null) {
        setFailure('Choose a sample or a file first.');
        setPhase('error');
        return;
      }
      setResult(outcome);
      // T117.2: hand the response to /explainability, which renders its
      // `explanation` block as the API formatted it.
      rememberPrediction(outcome);
      setPhase('done');
    } catch (error) {
      setFailure(error instanceof ApiError ? error.message : String(error));
      setPhase('error');
    } finally {
      setBusy(false);
    }
  }, [chosenSample, file, task]);

  return (
    <div className={className}>
      {offered.length > 1 ? (
        <fieldset className="mb-6">
          <legend className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mb-2')}>
            Label space — these are separate tasks with separate models and are never merged
          </legend>
          <div className="flex flex-wrap gap-2">
            {offered.map((entry) => (
              <button
                key={entry.task}
                type="button"
                onClick={() => {
                  setTask(entry.task);
                  setChosenSample(null);
                  reset();
                }}
                aria-pressed={entry.task === task}
                className={cn(
                  'rounded border px-3 py-1.5 text-sm font-medium',
                  entry.task === task
                    ? 'border-sky-600 bg-sky-600 text-white dark:border-sky-500 dark:bg-sky-500'
                    : 'border-slate-300 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800',
                )}
              >
                {entry.label}
              </button>
            ))}
          </div>
        </fieldset>
      ) : null}

      {spec !== undefined ? (
        <p className={cn(TYPE_SCALE.body, SURFACE.muted, 'mb-6 max-w-prose')}>
          <span className="font-medium text-slate-900 dark:text-slate-100">{spec.title}.</span>{' '}
          {spec.description} Classes: {spec.classes.join(', ')}.
        </p>
      ) : null}

      {!available ? (
        <EmptyState
          title="No model is deployed for this task yet"
          description={
            <>
              <p>{unavailableReason}</p>
              <p className="mt-2">
                The page is complete and will score recordings as soon as{' '}
                <span className="font-mono">{spec?.model_dir ?? task}</span> holds a saved
                model. Nothing here fabricates a result in the meantime.
              </p>
            </>
          }
        />
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <div>
            <h3 className={cn(TYPE_SCALE.h3, 'mb-3')}>1. Choose a recording</h3>

            {pageSamples.length > 0 ? (
              <div className="mb-5">
                <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mb-2')}>
                  Built-in dataset samples. No corpus audio is committed to the repository:
                  these are served by the local inference service from its own read-only copy.
                </p>
                <ul className="space-y-2">
                  {pageSamples.map((entry) => {
                    const reachable = servable === null || servableIds.has(entry.sample_id);
                    const active = entry.sample_id === chosenSample;
                    return (
                      <li key={entry.sample_id}>
                        <button
                          type="button"
                          disabled={!reachable || busy}
                          aria-pressed={active}
                          onClick={() => {
                            setChosenSample(active ? null : entry.sample_id);
                            setFile(null);
                            reset();
                          }}
                          className={cn(
                            'w-full rounded border px-3 py-2 text-left text-sm',
                            active
                              ? 'border-sky-600 bg-sky-50 dark:border-sky-500 dark:bg-sky-950/40'
                              : 'border-slate-300 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800',
                            !reachable && 'cursor-not-allowed opacity-55',
                          )}
                        >
                          <span className="flex flex-wrap items-baseline justify-between gap-2">
                            <span className="font-mono">{entry.record_uid}</span>
                            <span className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
                              {entry.dataset_name} · {entry.duration_display}
                            </span>
                          </span>
                          <span className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1 block')}>
                            {entry.labels[task] !== null && entry.labels[task] !== undefined ? (
                              <>Corpus label: {entry.labels[task]}. </>
                            ) : null}
                            {reachable
                              ? entry.selection
                              : 'Not on this machine — dataset/ is read-only input and is never committed.'}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ) : null}

            <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mb-2')}>
              …or upload your own. {prediction.upload.duration_note} Accepted:{' '}
              {prediction.upload.accepted_suffixes.join(', ')}, between{' '}
              {prediction.upload.min_duration_display} and{' '}
              {prediction.upload.max_duration_display}.
            </p>
            <FileUpload
              onFile={(chosen) => {
                setFile(chosen);
                setChosenSample(null);
                reset();
              }}
              phase={phase}
              fileName={file?.name ?? null}
              disabled={busy}
            />

            <button
              type="button"
              onClick={() => void run()}
              disabled={busy || (file === null && chosenSample === null)}
              className={cn(
                'mt-4 w-full rounded px-4 py-2 text-sm font-semibold text-white',
                'bg-sky-700 hover:bg-sky-800 disabled:opacity-50 dark:bg-sky-600 dark:hover:bg-sky-500',
              )}
            >
              {busy ? 'Scoring…' : '2. Run the screening model'}
            </button>

            {probeFailed !== null ? (
              <p
                className={cn(
                  TYPE_SCALE.caption,
                  'mt-3 rounded border border-amber-300 bg-amber-50 p-2 text-amber-900',
                  'dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100',
                )}
              >
                {probeFailed}
              </p>
            ) : null}
          </div>

          <div>
            <h3 className={cn(TYPE_SCALE.h3, 'mb-3')}>3. Result</h3>
            <WaveformPreview
              className="mb-4"
              source={
                file ?? (chosenSample !== null ? sampleAudioUrl(chosenSample) : null)
              }
              label={file?.name ?? activeSample?.record_uid ?? 'recording'}
            />
            {busy ? <LoadingState label="Preprocessing, extracting 138 features and scoring" /> : null}
            {failure !== null ? (
              <ErrorState
                title="No prediction was produced"
                detail={failure}
                onRetry={() => void run()}
              />
            ) : null}
            {result !== null && !busy ? (
              <ResultCard
                result={result}
                classes={spec?.classes ?? []}
                sample={activeSample}
              />
            ) : null}
            {result === null && !busy && failure === null ? (
              <EmptyState
                title="Nothing scored yet"
                description="Pick a recording on the left and run the model. Nothing is shown here until a real prediction comes back."
              />
            ) : null}
          </div>
        </div>
      )}

      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-8')}>
        <Badge tone="neutral">Operating point</Badge> {prediction.operating_point.note}
      </p>
    </div>
  );
}
