import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api')>();
  return {
    ...actual,
    listHistory: vi.fn(),
    getHistoryRecord: vi.fn(),
    updateHistoryRecord: vi.fn(),
    deleteHistoryRecord: vi.fn(),
    restoreHistoryRecord: vi.fn(),
    deleteAllHistory: vi.fn(),
  };
});

import { HistoryBrowser } from '@/components/history/HistoryBrowser';
import {
  deleteAllHistory,
  deleteHistoryRecord,
  listHistory,
  restoreHistoryRecord,
  updateHistoryRecord,
  type HistoryPage,
  type HistoryRecord,
} from '@/lib/api';
import {
  NO_FILTERS,
  compareItemFromRecord,
  dayBound,
  filtersFromSearch,
  listParams,
  parseTags,
  recordAudio,
  searchFromFilters,
} from '@/lib/history';

/**
 * T132.1-T132.6, component half. The browser half, against the real API and its
 * TinyDB file, is `e2e/history.spec.ts`. Display strings are deliberately not
 * numbers, so a component that formatted a value itself would show something
 * not found here.
 */

const BATCH = '0123456789abcdef0123456789abcdef';

function record(id: string, extra: Partial<HistoryRecord> = {}): HistoryRecord {
  return {
    id,
    created_at: '2026-09-14T10:00:00.000000Z',
    updated_at: '2026-09-14T10:00:00.000000Z',
    file_name: id + '.wav',
    task: 'binary',
    task_title: 'Title binary',
    source_kind: 'upload',
    scorable: true,
    result: 'abnormal',
    confidence: 0.5,
    low_confidence: false,
    probabilities: { normal: 0.5, abnormal: 0.5 },
    notes: '',
    tags: [],
    batch_id: null,
    display: {
      result: 'abnormal',
      confidence: 'conf-' + id,
      confidence_percent: 'pct-' + id,
      probabilities: { normal: 'pn-' + id, abnormal: 'pa-' + id },
      probabilities_percent: {},
    },
    ...extra,
  };
}

function listing(items: HistoryRecord[]): HistoryPage {
  return {
    items,
    total: items.length,
    page: 1,
    page_size: 20,
    pages: 1,
    display: { total: 'total-' + String(items.length), page: 'page-one', pages: 'pages-one' },
    notice: null,
  };
}

describe('lib/history', () => {
  it('keeps the filters in the address and drops what is malformed', () => {
    const filters = { ...NO_FILTERS, q: 'beta', task: 'binary', batch: BATCH, from: '2026-09-01', page: 3 };
    const search = searchFromFilters(filters, 'rec1');
    expect(new URLSearchParams(search).get('record')).toBe('rec1');
    expect(filtersFromSearch(search)).toEqual(filters);
    expect(filtersFromSearch('?batch=nope&from=yesterday&page=-4')).toEqual(NO_FILTERS);
    expect(searchFromFilters(NO_FILTERS)).toBe('');
  });

  it('sends the viewer’s chosen days as times, and the batch as batch_id', () => {
    const params = listParams({ ...NO_FILTERS, from: '2026-09-14', to: '2026-09-14', batch: BATCH, tag: 'x' });
    expect(params.batch_id).toBe(BATCH);
    expect(params.tag).toBe('x');
    expect(params.date_from).toBe(new Date(2026, 8, 14).toISOString());
    expect(params.date_to).toBe(dayBound('2026-09-14', true));
    expect(new Date(params.date_to as string).getTime() - new Date(params.date_from as string).getTime()).toBe(
      24 * 60 * 60 * 1000 - 1,
    );
    expect(params.q).toBeUndefined();
  });

  it('parses tags the way the store keeps them', () => {
    expect(parseTags(' clinic  a, repeat,, Clinic A ')).toEqual(['clinic a', 'repeat']);
  });

  it('has audio only for a sample entry, and compares from display strings', () => {
    expect(recordAudio(record('u'))).toBeNull();
    expect(recordAudio(record('s', { source_kind: 'sample', file_name: 'binary-normal' }))).toMatch(
      /\/samples\/binary-normal\/audio$/,
    );
    const item = compareItemFromRecord(record('c'), ['normal', 'abnormal']);
    expect([item.confidenceDisplay, item.probabilitiesDisplay.abnormal, item.source]).toEqual(['conf-c', 'pa-c', null]);
  });
});

describe('HistoryBrowser', () => {
  beforeEach(() => {
    for (const mock of [listHistory, updateHistoryRecord, deleteHistoryRecord, restoreHistoryRecord, deleteAllHistory]) {
      vi.mocked(mock).mockReset();
    }
    window.history.replaceState(null, '', '/history/');
  });

  it('shows the empty state for a fresh install, linking to Analyse', async () => {
    vi.mocked(listHistory).mockResolvedValue(listing([]));
    render(<HistoryBrowser />);
    expect(await screen.findByText('No analyses yet')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Analyse a recording' }).getAttribute('href')).toBe('/');
  });

  it('lists display strings, reads a batch link and sends its filter', async () => {
    window.history.replaceState(null, '', '/history/?batch=' + BATCH);
    vi.mocked(listHistory).mockResolvedValue(listing([record('a', { tags: ['clinic'] }), record('b')]));
    render(<HistoryBrowser />);
    expect(await screen.findByText('conf-a')).toBeTruthy();
    expect(screen.getByTestId('history-total').textContent).toContain('total-2');
    expect(screen.getByTestId('history-page').textContent).toBe('Page page-one of pages-one');
    expect(screen.getByText('clinic')).toBeTruthy();
    expect(vi.mocked(listHistory).mock.calls.at(-1)?.[0].batch_id).toBe(BATCH);
  });

  it('saves notes and tags, deletes with an undo, and compares two entries', async () => {
    const a = record('a');
    vi.mocked(listHistory).mockResolvedValue(listing([a, record('b')]));
    vi.mocked(updateHistoryRecord).mockImplementation(async (_id, changes) => ({
      ...a,
      notes: changes.notes ?? '',
      tags: changes.tags ?? [],
      updated_at: 'later',
    }));
    vi.mocked(deleteHistoryRecord).mockResolvedValue({ deleted: 1, id: 'a' });
    vi.mocked(restoreHistoryRecord).mockResolvedValue(a);
    render(<HistoryBrowser />);

    fireEvent.click(await screen.findByRole('button', { name: 'Open a.wav' }));
    const drawer = await screen.findByRole('dialog');
    expect(within(drawer).getByTestId('record-confidence').textContent).toBe('conf-a');
    expect(within(drawer).getByText('Audio not kept')).toBeTruthy();
    fireEvent.change(within(drawer).getByLabelText('Notes'), { target: { value: 'follow up' } });
    fireEvent.change(within(drawer).getByLabelText('Tags'), { target: { value: 'x, y, X' } });
    fireEvent.click(within(drawer).getByRole('button', { name: 'Save notes and tags' }));
    await waitFor(() => expect(updateHistoryRecord).toHaveBeenCalledWith('a', { notes: 'follow up', tags: ['x', 'y'] }));

    fireEvent.click(await screen.findByRole('button', { name: 'Delete this analysis' }));
    await waitFor(() => expect(deleteHistoryRecord).toHaveBeenCalledWith('a'));
    fireEvent.click(await screen.findByRole('button', { name: 'Undo' }));
    await waitFor(() => expect(restoreHistoryRecord).toHaveBeenCalledWith('a'));

    const compare = screen.getByRole('button', { name: 'Compare selected' });
    expect((compare as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Select a.wav to compare' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Select b.wav to compare' }));
    fireEvent.click(compare);
    const view = await screen.findByRole('region', { name: 'Comparison' });
    expect(within(view).getAllByText('Audio not kept')).toHaveLength(2);
    expect(within(view).getByText('pa-b')).toBeTruthy();
  });

  it('a search still waiting to apply does not come back after Clear filters', async () => {
    window.history.replaceState(null, '', '/history/?tag=clinic');
    vi.mocked(listHistory).mockResolvedValue(listing([record('a')]));
    render(<HistoryBrowser />);
    fireEvent.change(await screen.findByLabelText('Search file names, notes and tags'), { target: { value: 'late' } });
    fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }));
    await new Promise((resolve) => setTimeout(resolve, 500));
    const last = vi.mocked(listHistory).mock.calls.at(-1)?.[0];
    expect(last?.q).toBeUndefined();
    expect(last?.tag).toBeUndefined();
    expect(window.location.search).toBe('');
  });

  it('deletes everything only after the confirmation', async () => {
    vi.mocked(listHistory).mockResolvedValueOnce(listing([record('a')])).mockResolvedValue(listing([]));
    vi.mocked(deleteAllHistory).mockResolvedValue({ deleted: 1, deleted_display: 'one' });
    render(<HistoryBrowser />);
    fireEvent.click(await screen.findByRole('button', { name: 'Delete all' }));
    expect(deleteAllHistory).not.toHaveBeenCalled();
    fireEvent.click(await screen.findByRole('button', { name: 'Delete everything' }));
    await waitFor(() => expect(deleteAllHistory).toHaveBeenCalledTimes(1));
    expect(await screen.findByText('No analyses yet')).toBeTruthy();
    expect(screen.getByText('History was emptied (one deleted).')).toBeTruthy();
  });
});
