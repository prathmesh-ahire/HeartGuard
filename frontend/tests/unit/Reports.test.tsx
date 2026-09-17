import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api')>();
  return {
    ...actual,
    listHistory: vi.fn(),
    listRecentReports: vi.fn(),
    recordingReportPdf: vi.fn(),
    modelSummaryReportPdf: vi.fn(),
    exportHistoryCsv: vi.fn(),
    redownloadReport: vi.fn(),
    saveDocument: vi.fn(),
  };
});

import { ReportsBoard } from '@/components/reports/ReportsBoard';
import {
  exportHistoryCsv,
  listHistory,
  listRecentReports,
  modelSummaryReportPdf,
  recordingReportPdf,
  redownloadReport,
  saveDocument,
  type HistoryPage,
  type HistoryRecord,
  type RecentReport,
} from '@/lib/api';

/**
 * T134.1-T134.6, component half. The browser half, against the real API and a
 * real generated PDF/CSV, is `e2e/reports.spec.ts`. This file checks the
 * component reads and displays only strings the API already formatted, and
 * that every action's success and failure path renders visibly.
 */

const DOC = { blob: new Blob(['x']), filename: 'x.pdf' };

function record(id: string): HistoryRecord {
  return {
    id,
    created_at: '2026-09-16T10:00:00.000000Z',
    updated_at: '2026-09-16T10:00:00.000000Z',
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
  } as HistoryRecord;
}

function historyPage(items: HistoryRecord[]): HistoryPage {
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

function recentReport(id: string, kind: RecentReport['kind'] = 'recording'): RecentReport {
  return {
    id,
    kind,
    title: 'Report ' + id,
    filename: id + '.pdf',
    size_bytes: 1234,
    size_display: 'size-' + id,
    record_id: null,
    created_at: '2026-09-16T10:00:00.000000Z',
  };
}

describe('ReportsBoard', () => {
  beforeEach(() => {
    for (const mock of [
      listHistory,
      listRecentReports,
      recordingReportPdf,
      modelSummaryReportPdf,
      exportHistoryCsv,
      redownloadReport,
      saveDocument,
    ]) {
      vi.mocked(mock).mockReset();
    }
  });

  it('shows the empty states for a fresh install', async () => {
    vi.mocked(listHistory).mockResolvedValue(historyPage([]));
    vi.mocked(listRecentReports).mockResolvedValue({ items: [], notice: null });
    render(<ReportsBoard />);
    expect(await screen.findByText('No analyses yet')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Analyse a recording' }).getAttribute('href')).toBe('/');
    expect(
      screen.getByText('Nothing generated yet in this session. Reports made above appear here for re-download.'),
    ).toBeTruthy();
  });

  it('lists recent recordings and downloads a per-recording PDF', async () => {
    vi.mocked(listHistory).mockResolvedValue(historyPage([record('a')]));
    vi.mocked(listRecentReports)
      .mockResolvedValueOnce({ items: [], notice: null })
      .mockResolvedValue({ items: [recentReport('r1')], notice: null });
    vi.mocked(recordingReportPdf).mockResolvedValue(DOC);
    render(<ReportsBoard />);

    expect(await screen.findByText('a.wav')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Download PDF' }));
    await waitFor(() => expect(recordingReportPdf).toHaveBeenCalledWith('a'));
    await waitFor(() => expect(saveDocument).toHaveBeenCalledWith(DOC));

    // A successful download bumps the recent-reports list, which is fetched again.
    expect(await screen.findByText('Report r1')).toBeTruthy();
    expect(screen.getByText(/size-r1/)).toBeTruthy();
  });

  it('shows a visible failure beside the row when a per-recording PDF fails', async () => {
    vi.mocked(listHistory).mockResolvedValue(historyPage([record('a')]));
    vi.mocked(listRecentReports).mockResolvedValue({ items: [], notice: null });
    vi.mocked(recordingReportPdf).mockRejectedValue(new Error('no bundle'));
    render(<ReportsBoard />);

    fireEvent.click(await screen.findByRole('button', { name: 'Download PDF' }));
    const alert = await screen.findByRole('alert');
    expect(within(alert).getByText('The report was not produced')).toBeTruthy();
    expect(within(alert).getByText('no bundle')).toBeTruthy();
  });

  it('sends the bulk export filters and downloads the CSV', async () => {
    vi.mocked(listHistory).mockResolvedValue(historyPage([]));
    vi.mocked(listRecentReports).mockResolvedValue({ items: [], notice: null });
    vi.mocked(exportHistoryCsv).mockResolvedValue({ blob: new Blob(['x']), filename: 'history.csv' });
    render(<ReportsBoard />);

    fireEvent.change(await screen.findByLabelText('Check type'), { target: { value: 'binary' } });
    fireEvent.change(screen.getByLabelText('From'), { target: { value: '2026-09-01' } });
    fireEvent.click(screen.getByRole('button', { name: 'Export CSV' }));
    await waitFor(() =>
      expect(exportHistoryCsv).toHaveBeenCalledWith({ task: 'binary', dateFrom: '2026-09-01', dateTo: undefined }),
    );
    expect(saveDocument).toHaveBeenCalled();
  });

  it('downloads the model summary and shows a visible failure on rejection', async () => {
    vi.mocked(listHistory).mockResolvedValue(historyPage([]));
    vi.mocked(listRecentReports).mockResolvedValue({ items: [], notice: null });
    vi.mocked(modelSummaryReportPdf).mockRejectedValueOnce(new Error('service down')).mockResolvedValue(DOC);
    render(<ReportsBoard />);

    const button = await screen.findByRole('button', { name: 'Download model summary PDF' });
    fireEvent.click(button);
    expect(await screen.findByText('service down')).toBeTruthy();

    fireEvent.click(button);
    await waitFor(() => expect(saveDocument).toHaveBeenCalledWith(DOC));
  });

  it('re-downloads a recent report byte-identical to what was first produced', async () => {
    vi.mocked(listHistory).mockResolvedValue(historyPage([]));
    vi.mocked(listRecentReports).mockResolvedValue({ items: [recentReport('r9')], notice: null });
    vi.mocked(redownloadReport).mockResolvedValue(DOC);
    render(<ReportsBoard />);

    fireEvent.click(await screen.findByRole('button', { name: 'Re-download' }));
    await waitFor(() => expect(redownloadReport).toHaveBeenCalledWith('r9'));
    expect(saveDocument).toHaveBeenCalledWith(DOC);
  });
});
