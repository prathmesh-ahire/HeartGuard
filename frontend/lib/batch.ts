import type { BatchExportRow, BatchStatus, PredictResult } from '@/lib/api';

/**
 * The batch table's state and its pure helpers (T131.1-T131.3, T131.6).
 *
 * Nothing here formats a number. Sorting reads the numeric `confidence` only to
 * order rows; every value a row shows is the server's display string. The
 * summary strip counts rows on screen -- the operator's own files, never a model
 * metric -- and a batch is capped well below 1,000 files, so a count's plain
 * digits are exactly what `format_value(n, "count")` would print.
 */

/** Files per batch. Sequential scoring on one CPU makes a larger batch a wait, not a feature. */
export const MAX_BATCH_FILES = 100;

export type RowState = 'queued' | 'running' | BatchStatus;

export interface BatchRow {
  /** The file's position when chosen: stable across sorting, unique for duplicate names. */
  key: number;
  file: File;
  name: string;
  state: RowState;
  result: PredictResult | null;
  /** The server's (or the upload check's) own words for a refused file. */
  message: string | null;
}

export type SortKey = 'order' | 'file' | 'status' | 'result' | 'confidence';

export interface SortState {
  key: SortKey;
  direction: 'asc' | 'desc';
}

export const DEFAULT_SORT: SortState = { key: 'order', direction: 'asc' };

const STATE_ORDER: readonly RowState[] = ['running', 'queued', 'scored', 'not_scored', 'failed', 'cancelled'];

export const STATE_LABEL: Record<RowState, string> = {
  queued: 'Waiting',
  running: 'Analysing',
  scored: 'Scored',
  not_scored: 'Not scored',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export function isFinished(row: BatchRow): boolean {
  return row.state !== 'queued' && row.state !== 'running';
}

function sortValue(row: BatchRow, key: SortKey): string | number | null {
  switch (key) {
    case 'order':
      return row.key;
    case 'file':
      return row.name.toLowerCase();
    case 'status':
      return STATE_ORDER.indexOf(row.state);
    case 'result':
      return row.state === 'scored' ? (row.result?.predicted_class ?? null) : null;
    case 'confidence':
      return row.state === 'scored' ? (row.result?.confidence ?? null) : null;
  }
}

/** A copy, sorted. A row with no value sorts last in either direction; ties keep file order. */
export function sortRows(rows: readonly BatchRow[], sort: SortState): BatchRow[] {
  const sign = sort.direction === 'asc' ? 1 : -1;
  return [...rows].sort((a, b) => {
    const left = sortValue(a, sort.key);
    const right = sortValue(b, sort.key);
    if (left === null && right !== null) return 1;
    if (right === null && left !== null) return -1;
    if (left !== null && right !== null && left !== right) {
      const order =
        typeof left === 'number' && typeof right === 'number'
          ? left - right
          : String(left).localeCompare(String(right));
      if (order !== 0) return sign * order;
    }
    return a.key - b.key;
  });
}

export function nextSort(current: SortState, key: SortKey): SortState {
  if (current.key !== key) return { key, direction: key === 'confidence' ? 'desc' : 'asc' };
  return { key, direction: current.direction === 'asc' ? 'desc' : 'asc' };
}

export interface BatchSummary {
  total: number;
  finished: number;
  byClass: { name: string; count: number }[];
  lowConfidence: number;
  notScored: number;
  failed: number;
  cancelled: number;
}

/** T131.2: counts per outcome. Classes in the task's declared order, never merged across tasks. */
export function batchSummary(rows: readonly BatchRow[], classes: readonly string[]): BatchSummary {
  const scored = rows.filter((row) => row.state === 'scored');
  return {
    total: rows.length,
    finished: rows.filter(isFinished).length,
    byClass: classes.map((name) => ({
      name,
      count: scored.filter((row) => row.result?.predicted_class === name).length,
    })),
    lowConfidence: scored.filter((row) => row.result?.low_confidence === true).length,
    notScored: rows.filter((row) => row.state === 'not_scored').length,
    failed: rows.filter((row) => row.state === 'failed').length,
    cancelled: rows.filter((row) => row.state === 'cancelled').length,
  };
}

/** T131.6: the table as the export request names it -- which rows, in shown order, no values. */
export function exportRows(rows: readonly BatchRow[]): BatchExportRow[] {
  return rows.map((row) => {
    if (!isFinished(row)) throw new Error(row.name + ' has not finished yet.');
    const state = row.state as BatchStatus;
    return state === 'scored'
      ? { file_name: row.name, status: state, history_id: row.result?.history_id ?? null }
      : { file_name: row.name, status: state, message: row.message };
  });
}

/** 32 lowercase hex characters: the form `/predict` accepts as a batch id. */
export function newBatchId(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}
