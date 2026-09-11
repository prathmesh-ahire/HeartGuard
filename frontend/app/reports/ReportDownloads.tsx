'use client';

import { useState } from 'react';

import { cn } from '@/lib/cn';
import { ErrorState } from '@/components/ui/States';
import {
  ApiError,
  experimentReport,
  objectivesReport,
  sampleReport,
  saveDocument,
  type GeneratedDocument,
} from '@/lib/api';
import type { GeneratedReports, GeneratedSample } from '@/lib/generated/types';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * Trigger and download the three reports (T117.3).
 *
 * Each button asks the running inference API to render a DOCX with the Phase
 * 107 generators and saves what comes back. A failure -- the API is not running,
 * the recording is unusable -- renders its reason as a visible error; a button
 * that silently does nothing is the one outcome this must never have.
 */

const BUTTON = cn(
  'rounded border px-3 py-1.5 text-sm font-medium',
  'border-slate-300 hover:bg-slate-50 disabled:opacity-50',
  'dark:border-slate-700 dark:hover:bg-slate-800',
);

const SELECT = cn(
  'rounded border px-2 py-1 text-sm',
  'border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900',
);

type Key = 'sample' | 'experiment' | 'objective';

export function ReportDownloads({
  reports,
  samples,
}: {
  reports: GeneratedReports;
  samples: GeneratedSample[];
}) {
  const [task, setTask] = useState(reports.sample.tasks[0]?.task ?? 'binary');
  const [sampleId, setSampleId] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [expId, setExpId] = useState(reports.experiments[0]?.exp_id ?? '');
  const [busy, setBusy] = useState<Key | null>(null);
  const [failure, setFailure] = useState<{ key: Key; message: string } | null>(null);
  const [saved, setSaved] = useState<{ key: Key; filename: string } | null>(null);

  const offered = samples.filter((sample) => sample.tasks.includes(task));

  const run = async (key: Key, request: () => Promise<GeneratedDocument>): Promise<void> => {
    setBusy(key);
    setFailure(null);
    setSaved(null);
    try {
      const document = await request();
      saveDocument(document);
      setSaved({ key, filename: document.filename });
    } catch (error) {
      setFailure({ key, message: error instanceof ApiError ? error.message : String(error) });
    } finally {
      setBusy(null);
    }
  };

  const status = (key: Key) => (
    <>
      {failure?.key === key ? (
        <ErrorState className="mt-3" title="The report was not produced" detail={failure.message} />
      ) : null}
      {saved?.key === key ? (
        <p className={cn(TYPE_SCALE.caption, 'mt-2 text-emerald-700 dark:text-emerald-400')}>
          Saved {saved.filename}.
        </p>
      ) : null}
    </>
  );

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <div className={cn(SURFACE.card, 'p-4')}>
        <p className={TYPE_SCALE.h3}>Recording report</p>
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>{reports.sample.contents}</p>
        <div className="mt-3 space-y-2">
          <label className={cn(TYPE_SCALE.caption, 'flex flex-col gap-1')}>
            <span className={SURFACE.muted}>Label space</span>
            <select
              className={SELECT}
              value={task}
              onChange={(event) => {
                setTask(event.target.value);
                setSampleId('');
              }}
            >
              {reports.sample.tasks.map((item) => (
                <option key={item.task} value={item.task}>
                  {item.title}
                </option>
              ))}
            </select>
          </label>
          <label className={cn(TYPE_SCALE.caption, 'flex flex-col gap-1')}>
            <span className={SURFACE.muted}>Built-in sample</span>
            <select className={SELECT} value={sampleId} onChange={(event) => setSampleId(event.target.value)}>
              <option value="">— or upload a file below —</option>
              {offered.map((sample) => (
                <option key={sample.sample_id} value={sample.sample_id}>
                  {sample.sample_id} ({sample.dataset_name})
                </option>
              ))}
            </select>
          </label>
          <label className={cn(TYPE_SCALE.caption, 'flex flex-col gap-1')}>
            <span className={SURFACE.muted}>WAV file</span>
            <input
              type="file"
              accept=".wav,audio/wav"
              disabled={sampleId !== ''}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <button
            type="button"
            className={BUTTON}
            disabled={busy !== null || (sampleId === '' && file === null)}
            onClick={() =>
              void run('sample', () =>
                sampleReport(task, sampleId !== '' ? { sampleId } : { file: file as File }),
              )
            }
          >
            {busy === 'sample' ? 'Scoring and rendering…' : 'Generate recording report'}
          </button>
        </div>
        {status('sample')}
      </div>

      <div className={cn(SURFACE.card, 'p-4')}>
        <p className={TYPE_SCALE.h3}>Experiment report</p>
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>
          One run summarised from the files it wrote: aggregate metrics in research rule 6&apos;s
          order, fold structure, run provenance and the digest of every file read.
        </p>
        <div className="mt-3 space-y-2">
          <label className={cn(TYPE_SCALE.caption, 'flex flex-col gap-1')}>
            <span className={SURFACE.muted}>Experiment run</span>
            <select className={SELECT} value={expId} onChange={(event) => setExpId(event.target.value)}>
              {reports.experiments.map((item) => (
                <option key={item.exp_id} value={item.exp_id}>
                  {item.exp_id === item.title ? item.exp_id : item.exp_id + ' — ' + item.title}
                </option>
              ))}
            </select>
          </label>
          <p className={cn(TYPE_SCALE.micro, SURFACE.subtle, 'font-mono normal-case tracking-normal')}>
            {reports.experiments.find((item) => item.exp_id === expId)?.directory}
          </p>
          <button
            type="button"
            className={BUTTON}
            disabled={busy !== null || expId === ''}
            onClick={() => void run('experiment', () => experimentReport(expId))}
          >
            {busy === 'experiment' ? 'Rendering…' : 'Generate experiment report'}
          </button>
        </div>
        {status('experiment')}
      </div>

      <div className={cn(SURFACE.card, 'p-4')}>
        <p className={TYPE_SCALE.h3}>Objective-coverage report</p>
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>
          {reports.objective.table_id}: every locked objective mapped to the modules, output files
          and tables that answer it, with its status and caveat. The same table is rendered below.
        </p>
        <div className="mt-3">
          {reports.objective.available ? (
            <button
              type="button"
              className={BUTTON}
              disabled={busy !== null}
              onClick={() => void run('objective', objectivesReport)}
            >
              {busy === 'objective' ? 'Fetching…' : 'Download objective-coverage report'}
            </button>
          ) : (
            <p className={cn(TYPE_SCALE.caption, 'text-amber-800 dark:text-amber-300')}>
              {reports.objective.reason}
            </p>
          )}
        </div>
        {status('objective')}
      </div>
    </div>
  );
}
