'use client';

import { useCallback, useMemo, useRef, useState } from 'react';

import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/States';
import { validateRecording } from '@/components/ui/FileUpload';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import { ApiError, predictFile, type PredictResult } from '@/lib/api';

/**
 * Score several recordings and export the table (T116.5).
 *
 * ## Sequential, not parallel
 *
 * Files are sent one at a time. The service is a single CPU-bound process doing
 * real preprocessing and feature extraction per recording, and firing twenty
 * requests at it does not make it faster — it makes every one of them slower and
 * the progress display meaningless. Sequential also means a failure is
 * attributable: row four failed, rows one to three are still valid results.
 *
 * ## A failed row is a row, not a gap
 *
 * A file the service refuses keeps its place in the table with its error, and it
 * is exported to the CSV with an empty prediction and the reason. Dropping it
 * would leave a table of twelve rows for twenty uploads with nothing saying why.
 *
 * ## The CSV carries display strings, not re-rounded numbers
 *
 * Every value written is `result.display.*` — already rounded in Python by the
 * same formatter as the precomputed tables. The browser does no arithmetic here;
 * it joins strings with commas.
 */

interface Row {
  name: string;
  status: 'pending' | 'done' | 'failed';
  result: PredictResult | null;
  error: string | null;
}

const COLUMNS = [
  'file',
  'task',
  'predicted_class',
  'confidence',
  'margin',
  'low_confidence',
  'duration_seconds',
  'n_missing_features',
  'error',
] as const;

function escapeCell(value: string): string {
  return /[",\n]/.test(value) ? '"' + value.replace(/"/g, '""') + '"' : value;
}

function toCsv(rows: Row[], task: string): string {
  const lines = [COLUMNS.join(',')];
  for (const row of rows) {
    const result = row.result;
    lines.push(
      [
        row.name,
        task,
        result?.predicted_class ?? '',
        result?.display.confidence ?? '',
        result?.display.margin ?? '',
        result === null ? '' : String(result.low_confidence),
        result?.display.duration_seconds ?? '',
        result?.display.n_missing_features ?? '',
        row.error ?? '',
      ]
        .map((cell) => escapeCell(String(cell)))
        .join(','),
    );
  }
  return lines.join('\n') + '\n';
}

export function BatchPanel({ task, className }: { task: string; className?: string }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [running, setRunning] = useState(false);
  const [position, setPosition] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  const done = useMemo(() => rows.filter((row) => row.status === 'done').length, [rows]);
  const failed = useMemo(() => rows.filter((row) => row.status === 'failed').length, [rows]);

  const start = useCallback(
    async (files: FileList) => {
      const chosen = Array.from(files);
      const initial: Row[] = chosen.map((file) => {
        const problem = validateRecording(file);
        return {
          name: file.name,
          status: problem === null ? 'pending' : 'failed',
          result: null,
          error: problem,
        };
      });
      setRows(initial);
      setRunning(true);
      setPosition(0);

      for (let index = 0; index < chosen.length; index += 1) {
        setPosition(index + 1);
        if (initial[index]?.status === 'failed') continue;
        const file = chosen[index];
        if (file === undefined) continue;
        try {
          const result = await predictFile(file, task);
          setRows((current) =>
            current.map((row, position) =>
              position === index ? { ...row, status: 'done', result, error: null } : row,
            ),
          );
        } catch (error) {
          const message = error instanceof ApiError ? error.message : String(error);
          setRows((current) =>
            current.map((row, position) =>
              position === index ? { ...row, status: 'failed', result: null, error: message } : row,
            ),
          );
        }
      }
      setRunning(false);
    },
    [task],
  );

  const download = useCallback(() => {
    const blob = new Blob([toCsv(rows, task)], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'pv-mepcg-batch-' + task + '.csv';
    anchor.click();
    URL.revokeObjectURL(url);
  }, [rows, task]);

  return (
    <section className={className} aria-label="Batch prediction">
      <div className="flex flex-wrap items-center gap-3">
        <input
          ref={input}
          type="file"
          multiple
          accept={prediction.upload.accepted_suffixes.join(',')}
          className="sr-only"
          id="batch-files"
          disabled={running}
          onChange={(event) => {
            const files = event.target.files;
            if (files !== null && files.length > 0) void start(files);
          }}
        />
        <label
          htmlFor="batch-files"
          className={cn(
            'cursor-pointer rounded border border-slate-300 px-3 py-1.5 text-sm font-medium',
            'hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800',
            running && 'pointer-events-none opacity-60',
          )}
        >
          Choose several WAV files
        </label>
        <button
          type="button"
          onClick={download}
          disabled={rows.length === 0 || running}
          className={cn(
            'rounded border border-slate-300 px-3 py-1.5 text-sm font-medium',
            'hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800',
          )}
        >
          Export results as CSV
        </button>
        {rows.length > 0 ? (
          <span className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
            {running ? position + ' of ' + rows.length + ' — scoring one at a time' : null}
            {!running ? done + ' scored, ' + failed + ' refused' : null}
          </span>
        ) : null}
      </div>

      {rows.length === 0 ? (
        <EmptyState
          className="mt-4"
          title="No batch run yet"
          description="Recordings are sent one at a time: the service does real preprocessing and feature extraction per file, so parallel uploads would only make each one slower."
        />
      ) : (
        <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase tracking-widest text-slate-500 dark:bg-slate-900/60">
              <tr>
                <th scope="col" className="px-3 py-2">File</th>
                <th scope="col" className="px-3 py-2">Indication</th>
                <th scope="col" className="px-3 py-2">Confidence</th>
                <th scope="col" className="px-3 py-2">Margin</th>
                <th scope="col" className="px-3 py-2">Note</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.name} className="border-t border-slate-100 dark:border-slate-800">
                  <td className="px-3 py-1.5 font-mono">{row.name}</td>
                  <td className="px-3 py-1.5">
                    {row.result?.predicted_class ?? (row.status === 'failed' ? '—' : '…')}
                  </td>
                  <td className="px-3 py-1.5 tabular-nums">
                    {row.result?.display.confidence ?? ''}
                  </td>
                  <td className="px-3 py-1.5 tabular-nums">{row.result?.display.margin ?? ''}</td>
                  <td className="px-3 py-1.5">
                    {row.error !== null ? (
                      <span className="text-rose-700 dark:text-rose-300">{row.error}</span>
                    ) : row.result?.low_confidence === true ? (
                      <Badge tone="warn">Low confidence</Badge>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-3')}>
        {prediction.disclaimer}
      </p>
    </section>
  );
}
