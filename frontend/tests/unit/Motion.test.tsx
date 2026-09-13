import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { Stagger, StaggerItem } from '@/components/motion/Reveal';
import { PageHeader } from '@/components/ui/PageHeader';
import {
  Skeleton,
  SkeletonChart,
  SkeletonList,
  SkeletonText,
  SkeletonTiles,
} from '@/components/ui/Skeleton';
import { HistoryLoading, InsightsLoading, ServiceUnavailable } from '@/components/ui/PageStates';
import { LoadingState } from '@/components/ui/States';
import { EASE_OUT, transitionFor } from '@/lib/motion';

/**
 * T128.5 / T128.6 / T128.7: motion switches off under reduced motion, the
 * skeletons animate only behind `motion-safe:`, loading is announced once, and
 * the page error state is an alert rather than an empty panel.
 */

function preferReducedMotion(): void {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    (query: string) =>
      ({
        matches: query.includes('reduce'),
        media: query,
        onchange: null,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        addListener: () => undefined,
        removeListener: () => undefined,
        dispatchEvent: () => false,
      }) as MediaQueryList,
  );
}

describe('transitionFor', () => {
  it('is a zero-length state change under reduced motion', () => {
    expect(transitionFor(true, { duration: 0.35, delay: 0.2, ease: EASE_OUT })).toEqual({
      duration: 0,
      delay: 0,
    });
  });

  it('passes the transition through otherwise', () => {
    const transition = { duration: 0.35, ease: EASE_OUT };
    expect(transitionFor(false, transition)).toBe(transition);
  });
});

describe('Stagger', () => {
  it('shows every item in its final state under reduced motion', async () => {
    preferReducedMotion();
    render(
      <Stagger>
        <StaggerItem>first</StaggerItem>
        <StaggerItem>second</StaggerItem>
      </Stagger>,
    );
    for (const label of ['first', 'second']) {
      const item = screen.getByText(label);
      await waitFor(() => expect(item.style.opacity).toBe('1'));
    }
  });
});

describe('skeletons', () => {
  it('pulse only behind motion-safe', () => {
    render(
      <>
        <Skeleton />
        <SkeletonText />
        <SkeletonTiles />
        <SkeletonList />
        <SkeletonChart />
      </>,
    );
    const animated = Array.from(document.querySelectorAll('[data-skeleton]'))
      .flatMap((node) => node.className.split(/\s+/))
      .filter((name) => name.includes('animate-'));
    expect(animated.length).toBeGreaterThan(0);
    for (const name of animated) expect(name).toBe('motion-safe:animate-skeleton');
  });

  it('are hidden from assistive technology', () => {
    render(<SkeletonList />);
    for (const node of Array.from(document.querySelectorAll('[data-skeleton]'))) {
      expect(node.closest('[aria-hidden="true"]')).not.toBeNull();
    }
  });
});

describe('page states', () => {
  it('announce loading once, as busy', () => {
    render(
      <>
        <LoadingState label="Loading a table" />
        <HistoryLoading />
        <InsightsLoading />
      </>,
    );
    const regions = screen.getAllByRole('status');
    expect(regions.length).toBe(3);
    for (const region of regions) expect(region.getAttribute('aria-busy')).toBe('true');
    expect(screen.getByText('Loading your history')).toBeTruthy();
  });

  it('render a service failure as an alert that says it is not a zero', () => {
    render(<ServiceUnavailable onRetry={() => undefined} />);
    const alert = screen.getByRole('alert');
    expect(alert.textContent).toMatch(/not responding/);
    expect(alert.textContent).toMatch(/this is a failure, not a/);
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy();
  });
});

describe('PageHeader', () => {
  it('renders secondary actions before the one primary action', () => {
    render(
      <PageHeader
        title="Analyse a recording"
        actions={<button type="button">Samples</button>}
        primaryAction={<button type="button">Upload</button>}
      />,
    );
    const buttons = screen.getAllByRole('button').map((button) => button.textContent);
    expect(buttons).toEqual(['Samples', 'Upload']);
    expect(screen.getByRole('heading', { level: 1, name: 'Analyse a recording' })).toBeTruthy();
  });
});
