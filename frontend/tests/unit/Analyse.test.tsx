import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { CheckSelector, type AnalyseCheck } from '@/components/predict/CheckSelector';
import { ResultCard } from '@/components/predict/ResultCard';
import type { PredictResult } from '@/lib/api';
import { clock, decodePcmWav, encodeWav, recordingFileName, resampleLinear } from '@/lib/wav';

/**
 * T130.2, T130.4-T130.6, component half. The browser half, against the real
 * API, is `e2e/analyse.spec.ts` and `e2e/record.spec.ts`.
 *
 * The result below is a shaped API response whose display strings are
 * deliberately not numbers, so a card that formatted a value itself instead of
 * rendering the server's string would show something these tests do not find.
 */

describe('lib/wav', () => {
  it('resamples a listening copy to the new length, keeping the samples it lands on', () => {
    const ramp = new Float32Array([0, 0.25, 0.5, 0.75]);
    const up = resampleLinear(ramp, 2000, 8000);
    expect(up.length).toBe(16);
    expect(up[0]).toBe(0);
    expect(up[4]).toBeCloseTo(0.25, 6);
    expect(up[2]).toBeCloseTo(0.125, 6); // halfway between the first two
    expect(resampleLinear(ramp, 2000, 2000)).not.toBe(ramp); // a copy, never the input
  });

  it('writes a WAV that the page decoder reads back, across capture blocks', () => {
    const blocks = [new Float32Array([0, 0.5]), new Float32Array([-0.5, 1, -1])];
    const decoded = decodePcmWav(encodeWav(blocks, 8000));
    expect(decoded).not.toBeNull();
    expect(decoded?.sampleRate).toBe(8000);
    expect(decoded?.channels).toBe(1);
    expect(decoded?.samples.length).toBe(5);
    expect(decoded?.samples[1]).toBeCloseTo(0.5, 3);
    expect(decoded?.samples[3]).toBeCloseTo(1, 3);
    expect(decoded?.samples[4]).toBe(-1);
  });

  it('refuses bytes that are not a RIFF WAVE', () => {
    expect(decodePcmWav(new TextEncoder().encode('ID3 this is not a wav file at all, honestly').buffer)).toBeNull();
  });

  it('names a recording from the local clock', () => {
    expect(recordingFileName(new Date(2026, 8, 14, 10, 5, 7))).toBe('recording-20260914-100507.wav');
  });

  it('reads a player clock in whole seconds', () => {
    expect(clock(0)).toBe('0:00');
    expect(clock(65.9)).toBe('1:05');
    expect(clock(Number.NaN)).toBe('0:00');
  });
});

const CHECKS: AnalyseCheck[] = [
  { id: 'normal', label: 'Normal or abnormal', description: 'one', tasks: [{ task: 'binary', label: 'Binary' }] },
  {
    id: 'sound',
    label: 'Sound type',
    description: 'two',
    tasks: [
      { task: 'pascal_a', label: 'PASCAL A' },
      { task: 'pascal_b', label: 'PASCAL B' },
    ],
  },
];

describe('CheckSelector', () => {
  it('selects exactly one task, and asks which only for a two-task check', () => {
    const onChange = vi.fn();
    const { rerender } = render(<CheckSelector checks={CHECKS} task="binary" onChange={onChange} />);
    expect(screen.queryByRole('group', { name: 'Which set of categories' })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /Sound type/ }));
    expect(onChange).toHaveBeenCalledWith('pascal_a');

    rerender(<CheckSelector checks={CHECKS} task="pascal_b" onChange={onChange} />);
    expect(screen.getByRole('button', { name: /Sound type/ }).getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByRole('button', { name: 'PASCAL B' }).getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByRole('button', { name: 'PASCAL A' }).getAttribute('aria-pressed')).toBe('false');
  });
});

function result(overrides: Partial<PredictResult> = {}): PredictResult {
  return {
    task: 'binary',
    predicted_class: 'abnormal',
    predicted_index: 1,
    probabilities: { normal: 0.25, abnormal: 0.75 },
    confidence: 0.75,
    margin: 0.5,
    low_confidence: false,
    low_confidence_margin: 0.2,
    operating_threshold: 0.5,
    operating_point_note: 'operating point note',
    timings_seconds: { total: 1 },
    n_features: 138,
    n_missing_features: 0,
    feature_flags: [],
    quality: {},
    model: {
      task: 'binary',
      model_id: 'M1',
      estimator_class: null,
      n_features: 138,
      saved_at: null,
      n_records_fitted: null,
      selection_rule: null,
      note: null,
      path: null,
    },
    source: 'heart.wav',
    disclaimer: 'disclaimer text',
    warnings: [],
    scorable: true,
    not_scorable_reason: null,
    display: {
      probabilities: { normal: 'server-normal', abnormal: 'server-abnormal' },
      probabilities_percent: {},
      confidence: 'server-confidence',
      margin: 'server-margin',
      low_confidence_margin: 'server-lcm',
      operating_threshold: 'server-threshold',
      duration_seconds: 'server-duration',
      timings_seconds: { total: 'server-time' },
      n_features: '138',
      n_missing_features: '0',
    },
    history_id: 'row 1',
    history_saved: true,
    history_note: null,
    ...overrides,
  };
}

describe('ResultCard', () => {
  it('renders the outcome, the gauge and the probabilities from display strings', () => {
    render(<ResultCard result={result()} classes={['normal', 'abnormal']} taskTitle="Binary screening" />);
    expect(screen.getByTestId('result-class').textContent).toBe('abnormal');
    expect(screen.getByRole('img', { name: 'Confidence server-confidence' })).toBeTruthy();
    expect(screen.getByRole('img', { name: 'abnormal probability server-abnormal' })).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('links to the History row the API saved', () => {
    render(<ResultCard result={result()} classes={['normal', 'abnormal']} />);
    const link = screen.getByRole('link', { name: 'View in History' });
    const target = new URL(link.getAttribute('href') ?? '', 'http://localhost');
    expect(target.pathname.replace(/\/$/, '')).toBe('/history');
    expect(target.searchParams.get('record')).toBe('row 1');
  });

  it('says why a result was not saved, and offers no link to nothing', () => {
    render(
      <ResultCard
        result={result({ history_id: null, history_saved: false, history_note: 'History is unavailable.' })}
        classes={['normal', 'abnormal']}
      />,
    );
    expect(screen.queryByRole('link', { name: 'View in History' })).toBeNull();
    expect(screen.getByText('History is unavailable.')).toBeTruthy();
  });

  it('puts the low-confidence hint directly under the answer', () => {
    render(<ResultCard result={result({ low_confidence: true })} classes={['normal', 'abnormal']} />);
    const alert = screen.getByRole('alert');
    expect(alert.textContent).toContain('server-margin');
    expect(alert.textContent).toContain('server-lcm');
  });

  it('downloads the report, and a failed report is an error, not silence', async () => {
    const onDownloadReport = vi.fn().mockRejectedValue(new Error('service down'));
    render(<ResultCard result={result()} classes={['normal', 'abnormal']} onDownloadReport={onDownloadReport} />);
    fireEvent.click(screen.getByRole('button', { name: 'Download report' }));
    expect(onDownloadReport).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('service down'));
  });
});
