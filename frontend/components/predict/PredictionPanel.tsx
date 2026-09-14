'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import { cn } from '@/lib/cn';
import { GlassCard } from '@/components/ui/GlassCard';
import { Button } from '@/components/ui/Button';
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States';
import { FileUpload, type UploadPhase } from '@/components/ui/FileUpload';
import { MicRecorder } from '@/components/audio/MicRecorder';
import { WaveformPlayer } from '@/components/audio/WaveformPlayer';
import { BatchPanel } from '@/components/predict/BatchPanel';
import { CheckSelector, type AnalyseCheck } from '@/components/predict/CheckSelector';
import { ResultCard } from '@/components/predict/ResultCard';
import { TYPE_SCALE } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import { rememberPrediction } from '@/lib/lastPrediction';
import type { GeneratedSample, GeneratedTaskSpec } from '@/lib/generated/types';
import {
  ApiError,
  predictFile,
  predictSample,
  sampleAudioUrl,
  sampleReport,
  samples as fetchSamples,
  saveDocument,
  tasks as fetchTasks,
  type PredictResult,
  type SampleStatus,
  type TaskStatus,
} from '@/lib/api';

/**
 * The Analyse workspace (T116.1, rebuilt in T130.1-T130.6).
 *
 * Choose a check, add a recording -- dropped, recorded or a built-in sample --
 * hear and scrub it, analyse it, and read the result beside it. One primary
 * action: "Analyse recording".
 *
 * ## One path for every recording
 *
 * A dropped file and a microphone recording both arrive through `acceptFile`,
 * as a validated WAV `File`. From there they are the same thing: the same
 * preview, the same `POST /predict`, the same History row, the same report.
 * Only a built-in sample differs, because its audio is on the server already.
 *
 * ## Availability is a runtime question, asked at runtime
 *
 * `generated/prediction.json` records whether each task had a saved model when
 * the export ran. That goes stale the moment a model is saved, so the panel
 * re-reads `GET /tasks` on mount and prefers the live answer.
 *
 * ## Nothing is rounded here
 *
 * Every number rendered downstream comes from `result.display.*`, formatted in
 * Python.
 */

function taskSpec(name: string): GeneratedTaskSpec | undefined {
  return prediction.tasks.find((entry) => entry.task === name);
}

function samplesFor(name: string): GeneratedSample[] {
  return prediction.samples.filter((entry) => entry.tasks.includes(name));
}

export function PredictionPanel({
  checks,
  className,
}: {
  /** The checks offered. Each task inside a check is still its own label space. */
  checks: readonly AnalyseCheck[];
  className?: string;
}) {
  const [task, setTask] = useState(checks[0]?.tasks[0]?.task ?? 'binary');
  const [live, setLive] = useState<TaskStatus[] | null>(null);
  const [servable, setServable] = useState<SampleStatus[] | null>(null);
  const [probeFailed, setProbeFailed] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [chosenSample, setChosenSample] = useState<string | null>(null);
  const [phase, setPhase] = useState<UploadPhase>('idle');
  const [result, setResult] = useState<PredictResult | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // T131: one recording, or a batch. A running batch locks the check and the mode.
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [batchBusy, setBatchBusy] = useState(false);

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

  /** T130.1 and T130.2: an upload and a recording both land here. */
  const acceptFile = useCallback(
    (chosen: File) => {
      setFile(chosen);
      setChosenSample(null);
      reset();
    },
    [reset],
  );

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
        setFailure('Choose a recording first.');
        setPhase('error');
        return;
      }
      setResult(outcome);
      // T117.2: hand the response to About the Model's Performance tab.
      rememberPrediction(outcome);
      setPhase('done');
    } catch (error) {
      setFailure(error instanceof ApiError ? error.message : String(error));
      setPhase('error');
    } finally {
      setBusy(false);
    }
  }, [chosenSample, file, task]);

  /** T130.6: the report is rebuilt by the API from the same recording. */
  const downloadReport = useCallback(async () => {
    const source =
      chosenSample !== null ? { sampleId: chosenSample } : file !== null ? { file } : null;
    if (source === null) throw new Error('The recording is no longer selected.');
    saveDocument(await sampleReport(task, source));
  }, [chosenSample, file, task]);

  const source = file ?? (chosenSample !== null ? sampleAudioUrl(chosenSample) : null);

  return (
    <div
      className={cn(
        'grid items-start gap-6',
        mode === 'single' && 'lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]',
        className,
      )}
    >
      <div className="min-w-0 space-y-6">
        <GlassCard eyebrow="Step 1" title="Choose a check">
          <CheckSelector
            checks={checks}
            task={task}
            disabled={busy || batchBusy}
            onChange={(next) => {
              setTask(next);
              setChosenSample(null);
              reset();
            }}
          />
          {spec !== undefined ? (
            <p className={cn(TYPE_SCALE.caption, 'mt-3 max-w-prose text-ink-3')}>
              {spec.description} Categories: {spec.classes.join(', ')}.
            </p>
          ) : null}
        </GlassCard>

        {!available ? (
          <EmptyState
            title="No model is deployed for this task yet"
            description={<p>{unavailableReason}</p>}
          />
        ) : (
          <>
          <div role="group" aria-label="How many recordings" className="flex flex-wrap gap-2">
            {(['single', 'batch'] as const).map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={mode === value}
                disabled={busy || batchBusy}
                onClick={() => setMode(value)}
                className={cn(
                  'rounded-lg border px-3 py-1.5 font-mono text-label-md uppercase transition-colors',
                  mode === value
                    ? 'border-accent bg-accent-soft text-accent-deep'
                    : 'border-line bg-panel text-ink-2 hover:border-accent-line',
                  (busy || batchBusy) && 'cursor-not-allowed opacity-60',
                )}
              >
                {value === 'single' ? 'One recording' : 'Several recordings'}
              </button>
            ))}
          </div>
          {mode === 'batch' ? (
            <GlassCard eyebrow="Step 2" title="Add several recordings">
              {/* Keyed by task: a batch's rows belong to the check they were scored for. */}
              <BatchPanel
                key={task}
                task={task}
                taskTitle={spec?.title ?? task}
                classes={spec?.classes ?? []}
                onRunningChange={setBatchBusy}
              />
            </GlassCard>
          ) : (
          <GlassCard eyebrow="Step 2" title="Add a recording">
            <FileUpload onFile={acceptFile} phase={phase} fileName={file?.name ?? null} disabled={busy} />

            <MicRecorder
              onFile={acceptFile}
              className="mt-4"
              maxSeconds={prediction.upload.max_duration_seconds}
              minSeconds={prediction.upload.min_duration_seconds}
              minDisplay={prediction.upload.min_duration_display}
              disabled={busy}
            />

            <p className={cn(TYPE_SCALE.caption, 'mt-4 text-ink-3')}>
              Accepted: {prediction.upload.accepted_suffixes.join(', ')}, between{' '}
              {prediction.upload.min_duration_display} and {prediction.upload.max_duration_display}.{' '}
              {prediction.upload.duration_note}
            </p>

            {pageSamples.length > 0 ? (
              <div className="mt-5">
                <p className="label-micro">Or try a sample from the datasets</p>
                <ul className="mt-2 grid gap-2 sm:grid-cols-2">
                  {pageSamples.map((entry) => {
                    const reachable = servable === null || servableIds.has(entry.sample_id);
                    const active = entry.sample_id === chosenSample;
                    const label = entry.labels[task];
                    return (
                      <li key={entry.sample_id}>
                        <button
                          type="button"
                          // Addressed by sample id (the capture plan and the
                          // browser tests), not by its visible text.
                          data-sample-id={entry.sample_id}
                          disabled={!reachable || busy}
                          aria-pressed={active}
                          title={entry.selection}
                          onClick={() => {
                            setChosenSample(active ? null : entry.sample_id);
                            setFile(null);
                            reset();
                          }}
                          className={cn(
                            'w-full rounded-lg border px-3 py-2 text-left transition-colors',
                            active
                              ? 'border-accent bg-accent-soft shadow-panel'
                              : 'border-line bg-sunken hover:border-accent-line',
                            !reachable && 'cursor-not-allowed opacity-55',
                          )}
                        >
                          <span className={cn(TYPE_SCALE.body, 'block font-medium capitalize text-ink')}>
                            {label ?? entry.dataset_name}
                          </span>
                          <span className={cn(TYPE_SCALE.caption, 'mt-0.5 block text-ink-3')}>
                            {reachable
                              ? entry.dataset_name + ' · ' + entry.duration_display
                              : 'The datasets are not on this machine.'}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ) : null}

            <WaveformPlayer
              className="mt-5"
              source={source}
              label={file?.name ?? (activeSample !== null ? activeSample.dataset_name + ' sample' : 'recording')}
            />

            <Button
              tone="primary"
              size="lg"
              icon="pulse"
              className="mt-5 w-full"
              onClick={() => void run()}
              disabled={busy || (file === null && chosenSample === null)}
            >
              {busy ? 'Analysing…' : 'Analyse recording'}
            </Button>

            {probeFailed !== null ? (
              <p className={cn(TYPE_SCALE.caption, 'mt-3 rounded border border-warn-line bg-warn-soft p-2 text-warn')}>
                {probeFailed}
              </p>
            ) : null}
          </GlassCard>
          )}
          </>
        )}
      </div>

      {mode === 'single' ? (
      <section aria-label="Result" className="space-y-3 lg:sticky lg:top-6">
        <h2 className={cn(TYPE_SCALE.h2, 'text-ink')}>Result</h2>
        {busy ? <LoadingState label="Cleaning the recording, measuring it and scoring it" /> : null}
        {failure !== null ? (
          <ErrorState title="No prediction was produced" detail={failure} onRetry={() => void run()} />
        ) : null}
        {result !== null && !busy ? (
          <ResultCard
            result={result}
            classes={spec?.classes ?? []}
            taskTitle={spec?.title}
            sample={activeSample}
            onDownloadReport={downloadReport}
          />
        ) : null}
        {result === null && !busy && failure === null ? (
          <EmptyState
            icon="pulse"
            title="Nothing scored yet"
            description="Add a recording and select Analyse recording. The result appears here."
          />
        ) : null}
      </section>
      ) : null}
    </div>
  );
}
