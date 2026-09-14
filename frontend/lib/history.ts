import type { CompareItem } from '@/components/predict/CompareView';
import { sampleAudioUrl, type HistoryRecord } from '@/lib/api';

/**
 * The History page's state and the few conversions it needs (T132.1-T132.5).
 *
 * Filtering, searching and paging happen on the server; this module only says
 * which filters are set, keeps them in the address (so a reload or a shared
 * link shows the same list), and turns the viewer's chosen days into times.
 * It formats no number.
 */

export const PAGE_SIZE = 20;

export interface HistoryFilters {
  q: string;
  task: string;
  result: string;
  tag: string;
  from: string;
  to: string;
  batch: string;
  page: number;
}

export const NO_FILTERS: HistoryFilters = { q: '', task: '', result: '', tag: '', from: '', to: '', batch: '', page: 1 };

const TEXT_KEYS = ['q', 'task', 'result', 'tag', 'from', 'to', 'batch'] as const;
const DAY = /^\d{4}-\d{2}-\d{2}$/;

/** `?q=…&task=…&page=2` -> filters. Unknown or malformed values are dropped. */
export function filtersFromSearch(search: string): HistoryFilters {
  const params = new URLSearchParams(search);
  const filters: HistoryFilters = { ...NO_FILTERS };
  for (const key of TEXT_KEYS) filters[key] = (params.get(key) ?? '').trim();
  if (!DAY.test(filters.from)) filters.from = '';
  if (!DAY.test(filters.to)) filters.to = '';
  if (!/^[0-9a-f]{32}$/.test(filters.batch)) filters.batch = '';
  const page = Number.parseInt(params.get('page') ?? '', 10);
  filters.page = Number.isInteger(page) && page > 1 ? page : 1;
  return filters;
}

/** Filters -> the address's query string, without the empty ones. */
export function searchFromFilters(filters: HistoryFilters, record: string | null = null): string {
  const params = new URLSearchParams();
  for (const key of TEXT_KEYS) if (filters[key] !== '') params.set(key, filters[key]);
  if (filters.page > 1) params.set('page', String(filters.page));
  if (record !== null) params.set('record', record);
  const text = params.toString();
  return text === '' ? '' : '?' + text;
}

export function hasFilters(filters: HistoryFilters): boolean {
  return TEXT_KEYS.some((key) => filters[key] !== '');
}

/**
 * A chosen day as the start or end of that day in the viewer's own time zone.
 * The server's `YYYY-MM-DD` means a UTC day, which is not the day the viewer
 * picked: at UTC+05:30 an analysis at 02:00 on the 15th is the 14th in UTC.
 */
export function dayBound(day: string, end: boolean): string {
  const [year, month, date] = day.split('-').map((part) => Number.parseInt(part, 10)) as [number, number, number];
  const moment = end ? new Date(year, month - 1, date, 23, 59, 59, 999) : new Date(year, month - 1, date);
  return moment.toISOString();
}

/** Filters -> `/api/history` query parameters. */
export function listParams(filters: HistoryFilters): Record<string, string> {
  const params: Record<string, string> = {
    page: String(filters.page),
    page_size: String(PAGE_SIZE),
    sort: 'created_at',
    order: 'desc',
  };
  if (filters.q !== '') params.q = filters.q;
  if (filters.task !== '') params.task = filters.task;
  if (filters.result !== '') params.result = filters.result;
  if (filters.tag !== '') params.tag = filters.tag;
  if (filters.batch !== '') params.batch_id = filters.batch;
  if (filters.from !== '') params.date_from = dayBound(filters.from, false);
  if (filters.to !== '') params.date_to = dayBound(filters.to, true);
  return params;
}

/**
 * When an analysis was saved, in the viewer's own locale and time zone. A
 * moment in time, not a metric: the server stores UTC and cannot know the
 * viewer's zone.
 */
export function formatWhen(iso: string): string {
  const moment = new Date(iso);
  if (Number.isNaN(moment.getTime())) return iso;
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(moment);
}

/** "a, b, , A" -> ["a", "b"]: trimmed, blanks dropped, repeats (any case) once. */
export function parseTags(text: string): string[] {
  const seen = new Set<string>();
  const tags: string[] = [];
  for (const raw of text.split(',')) {
    const tag = raw.split(/\s+/).filter(Boolean).join(' ');
    if (tag === '' || seen.has(tag.toLowerCase())) continue;
    seen.add(tag.toLowerCase());
    tags.push(tag);
  }
  return tags;
}

/**
 * The recording behind an entry, when it can still be fetched. History never
 * keeps audio; a built-in sample's recording is served by the API, so an entry
 * saved from a sample can still be heard. Uploads cannot.
 */
export function recordAudio(record: HistoryRecord): string | null {
  return record.source_kind === 'sample' ? sampleAudioUrl(record.file_name) : null;
}

/** T132.5: a History entry as the Phase 131 compare view takes it. */
export function compareItemFromRecord(record: HistoryRecord, classes: readonly string[]): CompareItem {
  return {
    key: record.id,
    label: record.file_name,
    source: recordAudio(record),
    task: record.task,
    taskTitle: record.task_title,
    classes,
    result: record.display.result,
    confidenceDisplay: record.display.confidence,
    lowConfidence: record.low_confidence,
    probabilities: record.probabilities,
    probabilitiesDisplay: record.display.probabilities,
  };
}
