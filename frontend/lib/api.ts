'use client';

/**
 * The only runtime network boundary in the dashboard (T116.1).
 *
 * ## What may cross this wire, and what may not
 *
 * Everything precomputed reaches the browser through build-time codegen into
 * `lib/generated/`. This module exists for the one thing that cannot: scoring a
 * recording that did not exist when the export ran. So it calls `POST /predict`,
 * `POST /predict/sample`, and the three status endpoints that say what the
 * running process can serve — and it never fetches a metric.
 *
 * ## No number is formatted here
 *
 * A live probability has no precomputed display string, so the API produces one:
 * every response carries a `display` block rounded in Python by the same
 * `tables.format_value` that formatted every table in `generated/`. Components
 * render `display.*`. There is no `toFixed` in this file or in anything that
 * consumes it, and a page that showed a live value at one precision and a
 * precomputed one at another would be showing two rounding conventions.
 *
 * ## The base URL
 *
 * Same-origin by default, which is the deployed arrangement: FastAPI serves the
 * exported site from `frontend/out/` and answers `/predict` from the same
 * process. `next dev` on :3000 is the exception, and it is the only reason the
 * API declares a CORS origin at all, so the dev fallback points at :8000.
 *
 * `NEXT_PUBLIC_API_BASE` overrides both. It is read at build time by Next, so
 * it is a build input rather than a runtime one — which is the right shape: a
 * static export has no server to read an environment variable at request time.
 */

export const API_BASE: string = (() => {
  const declared = process.env.NEXT_PUBLIC_API_BASE;
  if (typeof declared === 'string' && declared.length > 0) return declared.replace(/\/$/, '');
  if (typeof window !== 'undefined' && window.location.port === '3000') {
    // `next dev`. The exported build is served by the API itself and is
    // same-origin, so this branch never runs in a deployed dashboard.
    return window.location.protocol + '//' + window.location.hostname + ':8000';
  }
  return '';
})();

export interface PredictDisplay {
  probabilities: Record<string, string>;
  probabilities_percent: Record<string, string>;
  confidence: string;
  margin: string;
  low_confidence_margin: string;
  operating_threshold: string | null;
  duration_seconds: string | null;
  timings_seconds: Record<string, string>;
  n_features: string;
  n_missing_features: string;
}

export interface PredictResult {
  task: string;
  predicted_class: string;
  predicted_index: number;
  probabilities: Record<string, number | null>;
  confidence: number | null;
  margin: number | null;
  low_confidence: boolean;
  low_confidence_margin: number;
  operating_threshold: number | null;
  operating_point_note: string;
  timings_seconds: Record<string, number | null>;
  n_features: number;
  n_missing_features: number;
  feature_flags: string[];
  quality: Record<string, unknown>;
  model: {
    task: string;
    model_id: string | null;
    estimator_class: string | null;
    n_features: number | null;
    saved_at: string | null;
    n_records_fitted: number | null;
    selection_rule: string[] | null;
    note: string | null;
    path: string | null;
  };
  source: string;
  disclaimer: string;
  warnings: string[];
  display: PredictDisplay;
  /**
   * What drove THIS recording's decision (T117.2), formatted by the API. Absent
   * from a response made before Phase 117, and `available: false` with a reason
   * for any estimator the decomposition is not exact for.
   */
  explanation?: PredictExplanation | null;
}

export interface PredictExplanationRow {
  feature: string;
  family: string;
  /** Bar geometry only. Render `contribution_display`. */
  contribution: number | null;
  contribution_display: string;
  value_display: string;
  scaled_display: string;
  coefficient_display: string;
  direction: string;
}

export interface PredictExplanation {
  available: boolean;
  reason: string | null;
  method?: string;
  units?: string;
  n_shown?: number;
  rows: PredictExplanationRow[];
}

export interface TaskStatus {
  task: string;
  title: string;
  classes: string[];
  description: string;
  available: boolean;
  model_dir: string;
  reason: string | null;
  loaded: boolean;
}

export interface SampleStatus {
  sample_id: string;
  record_uid: string;
  tasks: string[];
  selection: string;
  available: boolean;
  reason: string | null;
  bytes: number | null;
}

/**
 * A failure with the reason kept.
 *
 * The reason is the whole point. `ErrorState` renders it, and "the inference
 * service is not running" and "that recording is 0.3 s and the shortest the
 * models were fitted on is 0.76 s" are different problems with different fixes.
 * Collapsing both into "request failed" is how a user concludes the prototype
 * is broken when they uploaded the wrong file.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly offline: boolean;

  constructor(message: string, status: number, offline = false) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.offline = offline;
  }
}

const OFFLINE_HINT =
  'The inference service did not answer. Start it with ' +
  '`python -m src.api.main` and try again — the rest of this dashboard is ' +
  'precomputed and needs no server, but scoring a recording does.';

async function send(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(API_BASE + path, init);
  } catch {
    throw new ApiError(OFFLINE_HINT, 0, true);
  }
  if (!response.ok) {
    let detail = response.status + ' ' + response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      // A non-JSON error body is still an error; the status line stands.
    }
    throw new ApiError(detail, response.status);
  }
  return response;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await send(path, init);
  return (await response.json()) as T;
}

/** A document the API generated, with the filename it chose for it. */
export interface GeneratedDocument {
  blob: Blob;
  filename: string;
}

async function document(path: string, init?: RequestInit): Promise<GeneratedDocument> {
  const response = await send(path, init);
  const disposition = response.headers.get('content-disposition') ?? '';
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition);
  const filename = match?.[1] ? decodeURIComponent(match[1]) : 'report.docx';
  return { blob: await response.blob(), filename };
}

/**
 * T117.3: one recording's DOCX report, rendered by the API from the single
 * predict pass. Either a built-in sample id or an uploaded file.
 */
export function sampleReport(task: string, source: { sampleId: string } | { file: File }): Promise<GeneratedDocument> {
  const body = new FormData();
  body.append('task', task);
  if ('sampleId' in source) body.append('sample_id', source.sampleId);
  else body.append('file', source.file);
  return document('/report/sample', { method: 'POST', body });
}

/** T117.3: one experiment run, summarised from the files it wrote. */
export function experimentReport(expId: string): Promise<GeneratedDocument> {
  return document('/report/experiment/' + encodeURIComponent(expId));
}

/** T117.3: the objective-coverage report (T29's DOCX). */
export function objectivesReport(): Promise<GeneratedDocument> {
  return document('/report/objectives');
}

/** Hand a generated document to the browser's own download. */
export function saveDocument({ blob, filename }: GeneratedDocument): void {
  const url = URL.createObjectURL(blob);
  const anchor = window.document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function health(): Promise<{ status: string; tasks: TaskStatus[]; n_available: number }> {
  return call('/health');
}

export function tasks(): Promise<TaskStatus[]> {
  return call('/tasks');
}

export function samples(): Promise<SampleStatus[]> {
  return call('/samples');
}

export function sampleAudioUrl(sampleId: string): string {
  return API_BASE + '/samples/' + encodeURIComponent(sampleId) + '/audio';
}

/**
 * The bytes of a recording to preview: a built-in sample's URL or a local
 * `blob:` URL of an upload. Audio, never a metric -- it lives here because this
 * module is the dashboard's only network boundary, and T119.2's guard forbids a
 * `fetch` anywhere in `app/` or `components/`.
 */
export async function audioBytes(source: string): Promise<ArrayBuffer> {
  // Fetched as given: a sample URL already carries API_BASE, and a `blob:` URL
  // must not be prefixed with anything.
  let response: Response;
  try {
    response = await fetch(source);
  } catch {
    throw new ApiError(OFFLINE_HINT, 0, true);
  }
  if (!response.ok) throw new ApiError(response.status + ' ' + response.statusText, response.status);
  return response.arrayBuffer();
}

export function predictFile(file: File, task: string): Promise<PredictResult> {
  const body = new FormData();
  body.append('file', file);
  body.append('task', task);
  return call('/predict', { method: 'POST', body });
}

export function predictSample(sampleId: string, task: string): Promise<PredictResult> {
  const body = new FormData();
  body.append('sample_id', sampleId);
  body.append('task', task);
  return call('/predict/sample', { method: 'POST', body });
}

export interface PatientRule {
  rule: string;
  predicted_class: string;
  score: number | null;
  score_display: string;
  note: string;
}

export interface PatientResult {
  task: string;
  classes: string[];
  positive_class: string;
  n_recordings: number;
  recordings: PredictResult[];
  rules: PatientRule[];
  locations: Record<string, string>;
  disclaimer: string;
}

/**
 * Several recordings of one subject, collapsed by every declared rule.
 *
 * The collapse happens on the server. It produces a reported indication, and a
 * browser that computed one would be a second implementation of a rule the
 * CirCor experiments already define.
 */
export function predictPatient(sampleIds: string[], task: string): Promise<PatientResult> {
  const body = new FormData();
  body.append('sample_ids', sampleIds.join(','));
  body.append('task', task);
  return call('/predict/patient', { method: 'POST', body });
}
