import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { EChart } from '@/components/charts/EChart';

/**
 * T118.1 / T118.5: when the charting library cannot draw, the chart says so as
 * an alert. A blank box is indistinguishable from a real result of zero.
 */

vi.mock('echarts', () => ({
  init: () => {
    throw new Error('canvas unavailable');
  },
}));

describe('EChart', () => {
  it('renders a visible failure, never a silent empty chart', async () => {
    render(<EChart option={{}} ariaLabel="probe" />);
    await waitFor(() => expect(screen.getByRole('alert')).toBeTruthy());
    expect(screen.getByRole('alert').textContent).toMatch(/could not be drawn/);
    expect(screen.getByRole('alert').textContent).toMatch(/canvas unavailable/);
  });
});
