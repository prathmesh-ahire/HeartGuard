import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';

import { Drawer, Modal, Sheet } from '@/components/ui/Overlay';

/**
 * T128.3 / T128.7: the three overlays open as labelled dialogs, close on every
 * route out (Escape, the close button, the scrim), hand focus back to whatever
 * opened them, render nothing while closed, and animate only behind
 * `motion-safe:`. The browser half -- the top layer, and reduced motion really
 * switching the entrance off -- is `e2e-design/design.spec.ts`.
 */

const FRAMES = [
  ['Modal', Modal],
  ['Drawer', Drawer],
  ['Sheet', Sheet],
] as const;

describe.each(FRAMES)('%s', (_, Frame) => {
  it('renders nothing inside while closed', () => {
    render(
      <Frame open={false} onClose={() => undefined} title="Record detail">
        body text
      </Frame>,
    );
    expect(screen.queryByText('body text')).toBeNull();
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('opens as a dialog named by its title', () => {
    render(
      <Frame open onClose={() => undefined} title="Record detail" description="More about it">
        body text
      </Frame>,
    );
    const dialog = screen.getByRole('dialog', { name: 'Record detail' });
    expect(dialog.hasAttribute('open')).toBe(true);
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(screen.getByText('body text')).toBeTruthy();
    expect(screen.getByText('More about it')).toBeTruthy();
  });

  it('closes on Escape, on the close button and on the scrim', () => {
    const onClose = vi.fn();
    render(
      <Frame open onClose={onClose} title="Record detail">
        body text
      </Frame>,
    );
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    const scrim = document.querySelector('[data-scrim]');
    if (scrim === null) throw new Error('no scrim rendered');
    fireEvent.click(scrim);
    expect(onClose).toHaveBeenCalledTimes(3);
  });

  it('moves focus in on open and back to the trigger on close', () => {
    function Harness() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Open it
          </button>
          <Frame open={open} onClose={() => setOpen(false)} title="Record detail">
            body text
          </Frame>
        </>
      );
    }
    render(<Harness />);
    const trigger = screen.getByRole('button', { name: 'Open it' });
    trigger.focus();
    fireEvent.click(trigger);
    const close = screen.getByRole('button', { name: 'Close' });
    expect(document.activeElement).toBe(close);
    fireEvent.click(close);
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it('animates its entrance only behind motion-safe', () => {
    render(
      <Frame open onClose={() => undefined} title="Record detail">
        body text
      </Frame>,
    );
    const animated = Array.from(document.querySelectorAll('[data-panel], [data-scrim]'))
      .flatMap((node) => node.className.split(/\s+/))
      .filter((name) => name.includes('animate-'));
    expect(animated.length).toBe(2);
    for (const name of animated) expect(name.startsWith('motion-safe:')).toBe(true);
  });
});
