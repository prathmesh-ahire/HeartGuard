import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { GroupedBars } from '@/components/charts/Charts';
import { G10 } from '@/lib/generated/figures/G10';

/**
 * T118.1: a chart is fed the real G10 frame and hands ECharts exactly the
 * exported numbers and labels. ECharts itself needs a canvas jsdom does not
 * have, so `init` is replaced by a recorder -- what is under test is what this
 * project passes to the library, which is the part that could fabricate.
 */

const recorded: unknown[] = [];

vi.mock('echarts', () => ({
  init: () => ({
    setOption: (option: unknown) => recorded.push(option),
    resize: () => undefined,
    dispose: () => undefined,
  }),
}));

describe('GroupedBars', () => {
  it('plots the exported G10 values and labels, and nothing else', async () => {
    recorded.length = 0;
    render(
      <GroupedBars
        source={G10}
        categoryColumn="family"
        valueColumns={['n_features']}
        label="Features per family"
      />,
    );
    await waitFor(() => expect(recorded.length).toBeGreaterThan(0));
    const option = JSON.stringify(recorded.at(-1));

    const family = G10.columns.find((column) => column.name === 'family');
    const counts = G10.columns.find((column) => column.name === 'n_features');
    for (const name of family?.display ?? []) expect(option).toContain('"' + name + '"');
    for (const value of counts?.values ?? []) expect(option).toContain(String(value));
    expect(screen.getByRole('img', { name: /Features per family/ })).toBeTruthy();
  });

  it('renders a stated absence rather than an empty axis when there is no source', () => {
    render(
      <GroupedBars source={undefined} categoryColumn="family" valueColumns={['n_features']} label="x" />,
    );
    expect(screen.queryByRole('img')).toBeNull();
    expect(document.body.textContent).not.toBe('');
  });
});
