import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api')>();
  return { ...actual, predictFile: vi.fn(), exportBatchCsv: vi.fn(), saveDocument: vi.fn() };
});

import { BatchPanel } from '@/components/predict/BatchPanel';
import { CompareView, type CompareItem } from '@/components/predict/CompareView';
import { ApiError, exportBatchCsv, predictFile, saveDocument, type PredictResult } from '@/lib/api';
import { batchSummary, exportRows, newBatchId, nextSort, sortRows, type BatchRow } from '@/lib/batch';

/**
 * T131.1-T131.6, component half. The browser half, against the real API, is
 * `e2e/batch.spec.ts`. Display strings are deliberately not numbers, so a
 * component that formatted a value itself would show something not found here.
 */

const HEX = /^[0-9a-f]{32}$/;

function result(predicted: string, confidence: number, id: string, low = false): PredictResult {
  return {
    task: 'binary',
    predicted_class: predicted,
    confidence,
    low_confidence: low,
    probabilities: { normal: 0, abnormal: 1 },
    display: { confidence: 'conf-' + id, probabilities: { normal: 'pn-' + id, abnormal: 'pa-' + id } },
    history_id: 'history-' + id,
    history_saved: true,
    scorable: true,
  } as unknown as PredictResult;
}

function row(key: number, state: BatchRow['state'], outcome: PredictResult | null = null): BatchRow {
  const name = 'file-' + String(key) + '.wav';
  return { key, file: new File(['x'], name), name, state, result: outcome, message: state === 'failed' ? 'refused' : null };
}

function wavFile(name: string): File {
  return new File([new Uint8Array(64)], name, { type: 'audio/wav' });
}

describe('lib/batch', () => {
  const rows = [
    row(0, 'scored', result('normal', 0.2, 'a')),
    row(1, 'failed'),
    row(2, 'scored', result('abnormal', 0.9, 'b', true)),
    row(3, 'cancelled'),
  ];

  it('sorts by confidence with rows that have none last, in either direction', () => {
    expect(sortRows(rows, { key: 'confidence', direction: 'desc' }).map((r) => r.key)).toEqual([2, 0, 1, 3]);
    expect(sortRows(rows, { key: 'confidence', direction: 'asc' }).map((r) => r.key)).toEqual([0, 2, 1, 3]);
    expect(sortRows(rows, { key: 'status', direction: 'asc' }).map((r) => r.key)).toEqual([0, 2, 1, 3]);
    expect(rows.map((r) => r.key)).toEqual([0, 1, 2, 3]); // a copy, never in place
  });

  it('toggles a column and starts confidence from the highest', () => {
    expect(nextSort({ key: 'order', direction: 'asc' }, 'confidence')).toEqual({ key: 'confidence', direction: 'desc' });
    expect(nextSort({ key: 'file', direction: 'asc' }, 'file')).toEqual({ key: 'file', direction: 'desc' });
  });

  it('counts outcomes per declared class', () => {
    const summary = batchSummary(rows, ['normal', 'abnormal']);
    expect(summary.byClass).toEqual([
      { name: 'normal', count: 1 },
      { name: 'abnormal', count: 1 },
    ]);
    expect([summary.total, summary.finished, summary.failed, summary.cancelled, summary.lowConfidence]).toEqual([4, 4, 1, 1, 1]);
  });

  it('names the rows for the export without a single value', () => {
    expect(exportRows(rows)).toEqual([
      { file_name: 'file-0.wav', status: 'scored', history_id: 'history-a' },
      { file_name: 'file-1.wav', status: 'failed', message: 'refused' },
      { file_name: 'file-2.wav', status: 'scored', history_id: 'history-b' },
      { file_name: 'file-3.wav', status: 'cancelled', message: null },
    ]);
    expect(() => exportRows([row(9, 'queued')])).toThrow(/not finished/);
  });

  it('mints a batch id the API accepts', () => {
    const id = newBatchId();
    expect(id).toMatch(HEX);
    expect(newBatchId()).not.toBe(id);
  });
});

describe('CompareView', () => {
  const item = (key: string, task: string, verdict: string): CompareItem => ({
    key,
    label: key + '.wav',
    source: null,
    task,
    taskTitle: 'Title ' + task,
    classes: ['normal', 'abnormal'],
    result: verdict,
    confidenceDisplay: 'conf-' + key,
    lowConfidence: key === 'right',
    probabilities: { normal: 0, abnormal: 1 },
    probabilitiesDisplay: { normal: 'pn-' + key, abnormal: 'pa-' + key },
  });

  it('shows both sides from their display strings, and says when results differ', () => {
    render(<CompareView items={[item('left', 'binary', 'normal'), item('right', 'binary', 'abnormal')]} />);
    expect(screen.getAllByText('Audio not kept')).toHaveLength(2);
    expect(screen.getByText('conf-left')).toBeTruthy();
    expect(screen.getByText('pa-right')).toBeTruthy();
    expect(screen.getByText('Different results')).toBeTruthy();
    expect(screen.getByText('Low confidence')).toBeTruthy();
  });

  it('calls results from different tasks not comparable', () => {
    render(<CompareView items={[item('left', 'pascal_a', 'normal'), item('right', 'pascal_b', 'normal')]} />);
    expect(screen.getByRole('note').textContent).toMatch(/different sets of categories/);
    expect(screen.queryByText('Same result')).toBeNull();
  });
});

describe('BatchPanel', () => {
  beforeEach(() => {
    vi.mocked(predictFile).mockReset();
    vi.mocked(exportBatchCsv).mockReset();
    vi.mocked(saveDocument).mockReset();
  });

  function choose(container: HTMLElement, files: File[]) {
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files } });
  }

  const status = (container: HTMLElement, name: string) =>
    container.querySelector('tr[data-file="' + name + '"]')?.getAttribute('data-status');

  it('keeps a refused file as a row, carries on, saves under one batch id and exports in table order', async () => {
    vi.mocked(predictFile).mockImplementation(async (file) => {
      if (file.name === 'c.wav') throw new ApiError('c.wav could not be read as audio', 400);
      return result('abnormal', 0.8, 'a');
    });
    vi.mocked(exportBatchCsv).mockResolvedValue({ blob: new Blob(['x']), filename: 'batch.csv' });

    const { container } = render(<BatchPanel task="binary" taskTitle="Binary" classes={['normal', 'abnormal']} />);
    choose(container, [wavFile('a.wav'), new File(['x'], 'b.mp3'), wavFile('c.wav')]);
    expect(status(container, 'b.mp3')).toBe('failed');

    fireEvent.click(screen.getByRole('button', { name: 'Analyse 2 recordings' }));
    await waitFor(() => expect(status(container, 'c.wav')).toBe('failed'));
    expect(status(container, 'a.wav')).toBe('scored');
    expect(screen.getByText('c.wav could not be read as audio')).toBeTruthy();
    expect(screen.getByText('conf-a')).toBeTruthy();
    expect(container.querySelector('[data-count="failed"] dd')?.textContent).toBe('2');

    const ids = vi.mocked(predictFile).mock.calls.map((call) => call[2]?.batchId);
    expect(ids).toHaveLength(2);
    expect(ids[0]).toMatch(HEX);
    expect(ids[1]).toBe(ids[0]);
    expect(container.querySelector('[data-batch-id]')?.getAttribute('data-batch-id')).toBe(ids[0]);

    fireEvent.click(screen.getByRole('button', { name: 'Export CSV' }));
    await waitFor(() => expect(saveDocument).toHaveBeenCalledTimes(1));
    expect(vi.mocked(exportBatchCsv).mock.calls[0]).toEqual([
      ids[0],
      'binary',
      [
        { file_name: 'a.wav', status: 'scored', history_id: 'history-a' },
        { file_name: 'b.mp3', status: 'failed', message: expect.stringMatching(/not a WAV file/) },
        { file_name: 'c.wav', status: 'failed', message: 'c.wav could not be read as audio' },
      ],
    ]);
  });

  it('cancel lets the file in flight finish and cancels the rest', async () => {
    let finish: (value: PredictResult) => void = () => undefined;
    vi.mocked(predictFile).mockImplementation(
      () => new Promise<PredictResult>((resolve) => {
        finish = resolve;
      }),
    );
    const { container } = render(<BatchPanel task="binary" taskTitle="Binary" classes={['normal', 'abnormal']} />);
    choose(container, [wavFile('1.wav'), wavFile('2.wav'), wavFile('3.wav')]);
    fireEvent.click(screen.getByRole('button', { name: 'Analyse 3 recordings' }));
    await waitFor(() => expect(status(container, '1.wav')).toBe('running'));

    fireEvent.click(screen.getByRole('button', { name: 'Cancel batch' }));
    finish(result('normal', 0.7, 'one'));
    await waitFor(() => expect(status(container, '1.wav')).toBe('scored'));
    expect([status(container, '2.wav'), status(container, '3.wav')]).toEqual(['cancelled', 'cancelled']);
    expect(predictFile).toHaveBeenCalledTimes(1);
  });

  it('refuses more files than one batch takes, before any row is made', () => {
    const { container } = render(<BatchPanel task="binary" taskTitle="Binary" classes={['normal', 'abnormal']} />);
    choose(
      container,
      Array.from({ length: 101 }, (_, index) => wavFile(String(index) + '.wav')),
    );
    expect(screen.getByRole('alert').textContent).toMatch(/at most/);
    expect(container.querySelectorAll('tbody tr')).toHaveLength(0);
  });
});
