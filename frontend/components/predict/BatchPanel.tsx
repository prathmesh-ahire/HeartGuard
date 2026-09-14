'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';

import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button, ButtonLink } from '@/components/ui/Button';
import { EmptyState } from '@/components/ui/States';
import { validateRecording } from '@/components/ui/FileUpload';
import { CompareView, compareItemFromPrediction } from '@/components/predict/CompareView';
import { TYPE_SCALE } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import { ApiError, exportBatchCsv, predictFile, saveDocument } from '@/lib/api';
import {
  DEFAULT_SORT,
  MAX_BATCH_FILES,
  STATE_LABEL,
  batchSummary,
  exportRows,
  isFinished,
  newBatchId,
  nextSort,
  sortRows,
  type BatchRow,
  type RowState,
  type SortKey,
  type SortState,
} from '@/lib/batch';
import type { StatusTone } from '@/lib/tokens';

/**
 * Several recordings at once (T116.5, rebuilt in T131.1-T131.6).
 *
 * ## Sequential, not parallel
 *
 * Files are sent one at a time. The service is one CPU-bound process doing real
 * preprocessing and feature extraction per recording; parallel requests make
 * every one slower and the progress list meaningless. Sequential also makes a
 * failure attributable: row four failed, rows one to three are still results.
 *
 * ## A refused file is a row, not a gap (T131.5)
 *
 * The upload check refuses a wrong format or an empty file before the network;
 * the server refuses a corrupt, too-short or too-long one. Either way the file
 * keeps its row with the reason, and the batch carries on.
 *
 * ## Cancel stops what has not started
 *
 * Cancelling marks every waiting file cancelled; the file already being analysed
 * is allowed to finish. Aborting its request would not stop the server, which
 * would still save the result -- a History row the table no longer showed.
 *
 * ## One batch id, and a CSV the server writes (T131.3, T131.6)
 *
 * Each batch gets an id when it starts, sent with every file, so History holds
 * the batch as one. The CSV is built by the API from those History rows, in the
 * order the table is sorted, and formatted in Python; the page sends no number.
 */

const STATE_TONE: Record<RowState, StatusTone> = {
  queued: 'neutral',
  running: 'info',
  scored: 'good',
  not_scored: 'warn',
  failed: 'danger',
  cancelled: 'neutral',
};

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: 'file', label: 'File' },
  { key: 'status', label: 'Status' },
  { key: 'result', label: 'Result' },
  { key: 'confidence', label: 'Confidence' },
];

export function BatchPanel({
  task,
  taskTitle,
  classes,
  onRunningChange,
  className,
}: {
  /** One task for the whole batch: label spaces are never mixed in one table. */
  task: string;
  taskTitle: string;
  classes: readonly string[];
  onRunningChange?: (running: boolean) => void;
  className?: string;
}) {
  const inputId = useId();
  const [rows, setRows] = useState<BatchRow[]>([]);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);
  const [selected, setSelected] = useState<number[]>([]);
  const [comparing, setComparing] = useState<[number, number] | null>(null);
  const [chooseProblem, setChooseProblem] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportProblem, setExportProblem] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const cancelRequested = useRef(false);
  const skipped = useRef(new Set<number>());

  useEffect(() => onRunningChange?.(running), [running, onRunningChange]);

  const update = useCallback((key: number, patch: Partial<BatchRow>) => {
    setRows((current) => current.map((row) => (row.key === key ? { ...row, ...patch } : row)));
  }, []);

  const choose = useCallback((list: FileList | null) => {
    const files = list === null ? [] : Array.from(list);
    if (files.length === 0) return;
    if (files.length > MAX_BATCH_FILES) {
      setChooseProblem(
        String(files.length) + ' files were chosen; one batch takes at most ' + String(MAX_BATCH_FILES) + '.',
      );
      return;
    }
    setChooseProblem(null);
    setRows(
      files.map((file, index) => {
        const problem = validateRecording(file);
        return { key: index, file, name: file.name, state: problem === null ? 'queued' : 'failed', result: null, message: problem };
      }),
    );
    setBatchId(null);
    setSort(DEFAULT_SORT);
    setSelected([]);
    setComparing(null);
    setExportProblem(null);
  }, []);

  const start = useCallback(async () => {
    const id = newBatchId();
    const queue = rows.filter((row) => row.state === 'queued');
    cancelRequested.current = false;
    skipped.current = new Set();
    setBatchId(id);
    setRunning(true);
    for (const row of queue) {
      if (cancelRequested.current || skipped.current.has(row.key)) {
        update(row.key, { state: 'cancelled' });
        continue;
      }
      update(row.key, { state: 'running' });
      try {
        const result = await predictFile(row.file, task, { batchId: id });
        if (result.scorable === false) {
          update(row.key, {
            state: 'not_scored',
            result,
            message: result.not_scorable_reason ?? 'The recording could not be screened.',
          });
        } else {
          update(row.key, { state: 'scored', result, message: null });
        }
      } catch (error) {
        update(row.key, {
          state: 'failed',
          result: null,
          message: error instanceof ApiError ? error.message : String(error),
        });
      }
    }
    setRunning(false);
  }, [rows, task, update]);

  const cancel = useCallback(() => {
    cancelRequested.current = true;
    setRows((current) => current.map((row) => (row.state === 'queued' ? { ...row, state: 'cancelled' } : row)));
  }, []);

  const shown = useMemo(() => sortRows(rows, sort), [rows, sort]);
  const summary = useMemo(() => batchSummary(rows, classes), [rows, classes]);
  const done = rows.length > 0 && rows.every(isFinished);
  const started = batchId !== null;
  const waiting = rows.filter((row) => row.state === 'queued').length;
  const scoredCount = rows.filter((row) => row.state === 'scored').length;

  const download = useCallback(async () => {
    if (batchId === null) return;
    setExporting(true);
    setExportProblem(null);
    try {
      saveDocument(await exportBatchCsv(batchId, task, exportRows(shown)));
    } catch (error) {
      setExportProblem(error instanceof Error ? error.message : String(error));
    } finally {
      setExporting(false);
    }
  }, [batchId, shown, task]);

  const toggleSelected = (key: number) => {
    setSelected((current) =>
      current.includes(key) ? current.filter((item) => item !== key) : [...current, key].slice(-2),
    );
  };

  const pair = comparing === null ? null : comparing.map((key) => rows.find((row) => row.key === key));

  return (
    <section className={className} aria-label="Batch analysis" data-batch-id={batchId ?? undefined}>
      <div
        onDragOver={(event) => {
          event.preventDefault();
          if (!running) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!running) choose(event.dataTransfer.files);
        }}
        className={cn(
          'rounded-lg border-2 border-dashed p-6 text-center transition-colors',
          dragging ? 'border-accent bg-accent-soft' : 'border-line',
          running && 'opacity-60',
        )}
      >
        <p className={cn(TYPE_SCALE.body, 'font-medium')}>Drop several recordings here</p>
        <p className={cn(TYPE_SCALE.caption, 'mt-1 text-ink-3')}>
          Up to {MAX_BATCH_FILES} WAV files, analysed one at a time for {taskTitle}. Each result is saved to History as
          part of this batch.
        </p>
        <label
          htmlFor={inputId}
          className={cn(
            'mt-4 inline-block cursor-pointer rounded border border-line px-3 py-1.5 text-sm font-medium hover:bg-sunken',
            running && 'pointer-events-none',
          )}
        >
          Choose recordings
        </label>
        <input
          id={inputId}
          type="file"
          multiple
          className="sr-only"
          accept={prediction.upload.accepted_suffixes.join(',')}
          disabled={running}
          onChange={(event) => {
            choose(event.target.files);
            event.target.value = '';
          }}
        />
      </div>

      {chooseProblem !== null ? (
        <p role="alert" className={cn(TYPE_SCALE.caption, 'mt-3 rounded border-2 border-danger-line bg-danger-soft p-3 text-danger')}>
          {chooseProblem}
        </p>
      ) : null}

      {rows.length === 0 ? (
        <EmptyState
          className="mt-4"
          icon="pulse"
          title="No recordings chosen"
          description="Choose or drop several files. They are analysed one after another, and a file that cannot be read keeps its row with the reason."
        />
      ) : (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            {!started ? (
              <Button tone="primary" icon="pulse" onClick={() => void start()} disabled={waiting === 0}>
                {'Analyse ' + String(waiting) + (waiting === 1 ? ' recording' : ' recordings')}
              </Button>
            ) : null}
            {running ? (
              <Button tone="secondary" onClick={cancel}>
                Cancel batch
              </Button>
            ) : null}
            {done && started ? (
              <Button icon="reports" onClick={() => void download()} disabled={exporting}>
                {exporting ? 'Preparing CSV…' : 'Export CSV'}
              </Button>
            ) : null}
            {done && started ? (
              <Button
                onClick={() => setComparing(selected.length === 2 ? [selected[0] as number, selected[1] as number] : null)}
                disabled={selected.length !== 2}
              >
                Compare selected
              </Button>
            ) : null}
            {done && started && scoredCount > 0 ? (
              <ButtonLink href={'/history/?batch=' + encodeURIComponent(batchId as string)} icon="history">
                View batch in History
              </ButtonLink>
            ) : null}
          </div>

          <div className="mt-4">
            <div
              role="progressbar"
              aria-label="Batch progress"
              aria-valuemin={0}
              aria-valuemax={summary.total}
              aria-valuenow={summary.finished}
              className="h-1.5 w-full overflow-hidden rounded bg-sunken"
            >
              <div
                className="h-full bg-accent transition-[width]"
                style={{ width: (summary.total === 0 ? 0 : (summary.finished / summary.total) * 100) + '%' }}
              />
            </div>
            <p className={cn(TYPE_SCALE.caption, 'mt-1 text-ink-3')} data-testid="batch-progress">
              {String(summary.finished) + ' of ' + String(summary.total) + ' finished'}
              {running ? ' — analysing one at a time' : ''}
            </p>
          </div>

          <dl data-testid="batch-summary" className="mt-4 flex flex-wrap gap-2">
            {[
              { id: 'total', label: 'Recordings', count: summary.total, tone: 'neutral' as StatusTone },
              ...summary.byClass.map((entry) => ({
                id: 'class:' + entry.name,
                label: entry.name,
                count: entry.count,
                tone: 'info' as StatusTone,
              })),
              { id: 'low_confidence', label: 'Low confidence', count: summary.lowConfidence, tone: 'warn' as StatusTone },
              { id: 'not_scored', label: 'Not scored', count: summary.notScored, tone: 'warn' as StatusTone },
              { id: 'failed', label: 'Failed', count: summary.failed, tone: 'danger' as StatusTone },
              { id: 'cancelled', label: 'Cancelled', count: summary.cancelled, tone: 'neutral' as StatusTone },
            ].map((entry) => (
              <div
                key={entry.id}
                data-count={entry.id}
                className="flex items-baseline gap-2 rounded-lg border border-line bg-panel px-3 py-1.5"
              >
                <dt className="font-mono text-label-sm uppercase text-ink-3">{entry.label}</dt>
                <dd className="stat font-mono text-label-lg text-ink">{String(entry.count)}</dd>
              </div>
            ))}
          </dl>

          <div className="mt-4 overflow-x-auto rounded-lg border border-line">
            <table className="min-w-full text-left text-sm">
              <thead className="border-b-2 border-line bg-sunken font-mono text-label-sm uppercase text-ink-3">
                <tr>
                  <th scope="col" className="px-3 py-2">
                    <span className="sr-only">Select to compare</span>
                  </th>
                  {COLUMNS.map((column) => {
                    const active = sort.key === column.key;
                    return (
                      <th
                        key={column.key}
                        scope="col"
                        className="px-3 py-2"
                        aria-sort={active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}
                      >
                        <button
                          type="button"
                          className="inline-flex items-center gap-1 uppercase hover:text-accent-strong"
                          onClick={() => setSort((current) => nextSort(current, column.key))}
                        >
                          {column.label}
                          <span aria-hidden="true">{active ? (sort.direction === 'asc' ? '↑' : '↓') : '↕'}</span>
                        </button>
                      </th>
                    );
                  })}
                  <th scope="col" className="px-3 py-2">
                    Note
                  </th>
                  <th scope="col" className="px-3 py-2">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {shown.map((row) => (
                  <tr key={row.key} className="border-t border-line align-top" data-status={row.state} data-file={row.name}>
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        aria-label={'Select ' + row.name + ' to compare'}
                        checked={selected.includes(row.key)}
                        disabled={row.state !== 'scored' || running}
                        onChange={() => toggleSelected(row.key)}
                        className="accent-accent"
                      />
                    </td>
                    <td className="max-w-[16rem] truncate px-3 py-2 font-mono" title={row.name}>
                      {row.name}
                    </td>
                    <td className="px-3 py-2">
                      <Badge tone={STATE_TONE[row.state]} dot={row.state === 'running'} pulse={row.state === 'running'}>
                        {STATE_LABEL[row.state]}
                      </Badge>
                    </td>
                    <td className="px-3 py-2 capitalize" data-cell="result">
                      {row.state === 'scored' ? row.result?.predicted_class : ''}
                    </td>
                    <td className="stat px-3 py-2 font-mono" data-cell="confidence">
                      {row.state === 'scored' ? row.result?.display.confidence : ''}
                    </td>
                    <td className="px-3 py-2" data-cell="note">
                      {row.message !== null ? (
                        <span className={row.state === 'failed' ? 'text-danger' : 'text-warn'}>{row.message}</span>
                      ) : row.state === 'scored' && row.result?.low_confidence === true ? (
                        <Badge tone="warn">Low confidence</Badge>
                      ) : row.state === 'scored' && row.result?.history_saved !== true ? (
                        <span className="text-warn">{row.result?.history_note ?? 'Not saved to History.'}</span>
                      ) : null}
                    </td>
                    <td className="px-3 py-2 text-right">
                      {!started && row.state === 'queued' ? (
                        <Button
                          tone="ghost"
                          size="sm"
                          aria-label={'Remove ' + row.name}
                          onClick={() => setRows((current) => current.filter((item) => item.key !== row.key))}
                        >
                          Remove
                        </Button>
                      ) : null}
                      {running && row.state === 'queued' ? (
                        <Button
                          tone="ghost"
                          size="sm"
                          aria-label={'Skip ' + row.name}
                          onClick={() => {
                            skipped.current.add(row.key);
                            update(row.key, { state: 'cancelled' });
                          }}
                        >
                          Skip
                        </Button>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {done && started ? (
            <p className={cn(TYPE_SCALE.caption, 'mt-2 text-ink-3')}>
              Select two scored recordings to compare them side by side. The CSV follows the table’s current order.
            </p>
          ) : null}

          {exportProblem !== null ? (
            <p role="alert" className={cn(TYPE_SCALE.caption, 'mt-3 rounded border border-danger-line bg-danger-soft p-2 text-danger')}>
              The CSV was not produced. {exportProblem}
            </p>
          ) : null}

          {pair !== null && pair[0]?.result && pair[1]?.result ? (
            <CompareView
              className="mt-6"
              onClose={() => setComparing(null)}
              items={[
                compareItemFromPrediction({
                  key: String(pair[0].key),
                  label: pair[0].name,
                  source: pair[0].file,
                  result: pair[0].result,
                  taskTitle,
                  classes,
                }),
                compareItemFromPrediction({
                  key: String(pair[1].key),
                  label: pair[1].name,
                  source: pair[1].file,
                  result: pair[1].result,
                  taskTitle,
                  classes,
                }),
              ]}
            />
          ) : null}
        </>
      )}
    </section>
  );
}
