'use client';

import { useEffect, useId, useRef, type KeyboardEvent, type ReactNode } from 'react';

import { Icon } from '@/components/ui/Icon';
import { cn } from '@/lib/cn';

/**
 * Modal, drawer and sheet (T128.3): detail shown on demand, instead of added
 * to the page.
 *
 * All three are one frame -- a native `<dialog>` opened with `showModal()` --
 * with a different panel shape. The native element is chosen over a portal
 * library for what it gives for free and cannot get wrong: the top layer (no
 * z-index fight with the sticky bar), an inert page behind it (focus cannot
 * tab out), and zero bytes on the bundle budget.
 *
 * * **Modal** -- a decision or a short form. Centred.
 * * **Drawer** -- the detail of one row: a History record, a table entry. From
 *   the right, or from the left for the mobile navigation.
 * * **Sheet** -- supporting content on a phone-sized screen. From the bottom.
 *
 * Escape, the close button and a click on the scrim all call `onClose`; the
 * parent owns `open`, so the dialog can never be open while React thinks it is
 * closed. Focus returns to whatever opened it. Content is rendered only while
 * open, so a closed drawer adds no hidden text to the static HTML.
 *
 * Every entrance is behind `motion-safe:`, so under reduced motion the panel
 * simply appears.
 */

type Frame = 'modal' | 'drawer-right' | 'drawer-left' | 'sheet';

const PANEL: Record<Frame, string> = {
  modal:
    'relative m-auto flex max-h-[85vh] w-[min(36rem,calc(100vw-2rem))] flex-col rounded-2xl border motion-safe:animate-modal-in',
  'drawer-right':
    'absolute inset-y-0 right-0 flex w-[min(28rem,100vw)] flex-col border-l motion-safe:animate-drawer-in-right',
  'drawer-left':
    'absolute inset-y-0 left-0 flex w-[min(20rem,85vw)] flex-col border-r motion-safe:animate-drawer-in-left',
  sheet:
    'absolute inset-x-0 bottom-0 mx-auto flex max-h-[85vh] w-full max-w-content flex-col rounded-t-2xl border-t motion-safe:animate-sheet-in',
};

export interface OverlayProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: ReactNode;
  children: ReactNode;
  /** Actions, right-aligned under the body. At most one primary. */
  footer?: ReactNode;
  className?: string;
}

function DialogFrame({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  className,
  frame,
}: OverlayProps & { frame: Frame }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const descriptionId = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog === null || !open) return undefined;

    const returnTo = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (!dialog.hasAttribute('open')) {
      // jsdom has no showModal; the attribute is the same state without the top layer.
      if (typeof dialog.showModal === 'function') dialog.showModal();
      else dialog.setAttribute('open', '');
    }
    if (!dialog.contains(document.activeElement)) closeRef.current?.focus();

    const root = document.documentElement;
    const previousOverflow = root.style.overflow;
    root.style.overflow = 'hidden';

    return () => {
      root.style.overflow = previousOverflow;
      if (typeof dialog.close === 'function') dialog.close();
      dialog.removeAttribute('open');
      returnTo?.focus();
    };
  }, [open]);

  const onKeyDown = (event: KeyboardEvent<HTMLDialogElement>) => {
    if (event.key !== 'Escape') return;
    event.preventDefault();
    event.stopPropagation();
    onClose();
  };

  return (
    <dialog
      ref={dialogRef}
      aria-modal="true"
      aria-labelledby={titleId}
      aria-describedby={description ? descriptionId : undefined}
      data-frame={frame}
      // The browser's own close on Escape would shut the dialog behind React's
      // back. The key handler above closes it through `onClose` instead.
      onCancel={(event) => event.preventDefault()}
      onKeyDown={onKeyDown}
      className="fixed inset-0 z-50 m-0 h-full max-h-none w-full max-w-none border-0 bg-transparent p-0 text-ink backdrop:bg-transparent open:flex"
    >
      {open ? (
        <>
          <div
            aria-hidden="true"
            data-scrim=""
            onClick={onClose}
            className="absolute inset-0 bg-scrim/40 backdrop-blur-[2px] motion-safe:animate-overlay-in"
          />
          <div
            data-panel=""
            className={cn('border-line bg-panel shadow-raised', PANEL[frame], className)}
          >
            <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
              <div className="min-w-0">
                <h2 id={titleId} className="text-headline-md text-ink">
                  {title}
                </h2>
                {description ? (
                  <div id={descriptionId} className="mt-1 text-body-md text-ink-2">
                    {description}
                  </div>
                ) : null}
              </div>
              <button
                ref={closeRef}
                type="button"
                onClick={onClose}
                aria-label="Close"
                className="-mr-1 shrink-0 rounded-lg p-1.5 text-ink-2 transition-colors hover:bg-accent-soft hover:text-accent-strong"
              >
                <Icon name="close" className="h-4 w-4" />
              </button>
            </header>
            <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">{children}</div>
            {footer ? (
              <footer className="flex flex-wrap items-center justify-end gap-2 border-t border-line px-5 py-3">
                {footer}
              </footer>
            ) : null}
          </div>
        </>
      ) : null}
    </dialog>
  );
}

export function Modal(props: OverlayProps) {
  return <DialogFrame {...props} frame="modal" />;
}

export function Drawer({ side = 'right', ...props }: OverlayProps & { side?: 'left' | 'right' }) {
  return <DialogFrame {...props} frame={side === 'left' ? 'drawer-left' : 'drawer-right'} />;
}

export function Sheet(props: OverlayProps) {
  return <DialogFrame {...props} frame="sheet" />;
}
