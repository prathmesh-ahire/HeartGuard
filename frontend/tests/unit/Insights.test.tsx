import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api')>();
  return { ...actual, getInsights: vi.fn() };
});

import { InsightsBoard } from '@/components/insights/InsightsBoard';
import { ApiError, getInsights, type InsightsData } from '@/lib/api';

/**
 * T133.1-T133.6, component half. The browser half, against the real API and
 * seeded History, is `e2e/insights.spec.ts`. Every number this file checks is
 * a `*_display` string the mock hands in verbatim -- if the component ever
 * formatted one itself, changing the mock string here and not the assertion
 * would still catch it.
 */

function insights(overrides: Partial<InsightsData> = {}): InsightsData {
  return {
    task: null,
    total: 0,
    total_display: 'total-0',
    low_confidence: { count: 0, count_display: 'lc-count-0', percent: null, percent_display: 'lc-pct-0' },
    by_task: [],
    result_split: [],
    trend: {
      days: 30,
      tz_offset_minutes: 0,
      first_date: '2026-08-18',
      last_date: '2026-09-16',
      in_window: 0,
      in_window_display: 'window-0',
      points: [],
    },
    confidence_distribution: { n: 0, n_display: 'n-0', bins: [] },
    notice: null,
    ...overrides,
  };
}

describe('InsightsBoard', () => {
  beforeEach(() => {
    vi.mocked(getInsights).mockReset();
  });

  it('shows the no-history empty state and links to Analyse', async () => {
    vi.mocked(getInsights).mockResolvedValue(insights());
    render(<InsightsBoard />);
    expect(await screen.findByText('No trends to show yet')).toBeTruthy();
    expect(screen.getByRole('link', { name: 'Analyse a recording' }).getAttribute('href')).toBe('/');
  });

  it('shows a service-unavailable state and retries on click', async () => {
    vi.mocked(getInsights)
      .mockRejectedValueOnce(new ApiError('offline', 0, true))
      .mockResolvedValue(insights({ total: 1, total_display: 'total-1' }));
    render(<InsightsBoard />);
    expect(await screen.findByText('The analysis service is not responding')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(getInsights).toHaveBeenCalledTimes(2));
  });

  it('shows a visible error for a non-offline failure', async () => {
    vi.mocked(getInsights).mockRejectedValue(new Error('bad request'));
    render(<InsightsBoard />);
    expect(await screen.findByText('Insights could not be loaded')).toBeTruthy();
    expect(screen.getByText('bad request')).toBeTruthy();
  });

  it('shows the filtered-empty state when a check type has no data', async () => {
    vi.mocked(getInsights).mockResolvedValue(insights());
    render(<InsightsBoard />);
    fireEvent.change(await screen.findByLabelText('Check type'), { target: { value: 'binary' } });
    await waitFor(() => expect(getInsights).toHaveBeenLastCalledWith(expect.objectContaining({ task: 'binary' })));
    expect(await screen.findByText('No analyses match this filter')).toBeTruthy();
  });

  it('renders the notice and the display strings for tiles that are not animated', async () => {
    vi.mocked(getInsights).mockResolvedValue(
      insights({
        total: 12,
        notice: 'Only the last 90 days are kept.',
        low_confidence: { count: 2, count_display: '2', percent: 0.1667, percent_display: '16.7%' },
        result_split: [
          {
            task: 'binary',
            title: 'Normal or abnormal',
            total: 12,
            total_display: '12',
            classes: [
              { class: 'normal', count: 9, count_display: '9', percent: 75, percent_display: '75%' },
              { class: 'abnormal', count: 3, count_display: '3', percent: 25, percent_display: '25%' },
            ],
          },
        ],
        confidence_distribution: {
          n: 12,
          n_display: '12',
          bins: [{ lower: 0.5, upper: 1, label: '50-100%', count: 12, count_display: '12', percent: 100, percent_display: '100%' }],
        },
        trend: {
          days: 30,
          tz_offset_minutes: 0,
          first_date: '2026-08-18',
          last_date: '2026-09-16',
          in_window: 12,
          in_window_display: 'window-12',
          points: [{ date: '2026-09-16', count: 12, count_display: '12' }],
        },
      }),
    );
    render(<InsightsBoard />);
    expect(await screen.findByText('Only the last 90 days are kept.')).toBeTruthy();
    expect(screen.getByText('16.7%')).toBeTruthy();
    expect(screen.getByText('normal')).toBeTruthy();
    expect(screen.getByText('75% of Normal or abnormal')).toBeTruthy();
    // The reliability summary is always present, read from generated/prediction.
    expect(screen.getByRole('region', { name: 'Model reliability summary' })).toBeTruthy();
  });

  it('sends the browser timezone offset and the chosen trend window', async () => {
    vi.mocked(getInsights).mockResolvedValue(insights());
    render(<InsightsBoard />);
    await waitFor(() => expect(getInsights).toHaveBeenCalled());
    fireEvent.change(await screen.findByLabelText('Trend window'), { target: { value: '7' } });
    await waitFor(() =>
      expect(getInsights).toHaveBeenLastCalledWith(
        expect.objectContaining({ days: 7, tzOffsetMinutes: expect.any(Number) }),
      ),
    );
  });
});
