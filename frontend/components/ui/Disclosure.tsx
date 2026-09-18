'use client';

import { useEffect, useRef, type ReactNode } from 'react';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { Icon } from '@/components/ui/Icon';

/**
 * Secondary detail behind a `<details>` (T135.5, animated open/close in T142.5).
 *
 * Still a real `<details>`/`<summary>` element -- not a div-based accordion --
 * because that is what gives two things away for free that a from-scratch
 * accordion has to rebuild by hand: keyboard behaviour (Enter/Space on a
 * focused `<summary>` already toggles it -- no key handler needed, and none is
 * added here) and a browser auto-opening the closed `<details>` ancestor of a
 * URL fragment's target (T135.5's own deep-link behaviour, still exercised by
 * `e2e/about.spec.ts`). Swapping the element for a `<div>` would have to
 * reimplement both by hand and could silently regress either.
 *
 * What T142.5 changes is the TRANSITION: a native `<details>` snaps its
 * content open and closed with no animation at all. The click handler below
 * intercepts only that snap and replaces it with a height tween via
 * `element.animate()` -- the Web Animations API, a browser primitive, not a
 * library, so this adds no bundle weight. It still finishes by setting the
 * same boolean `open` a native toggle would have set, just after the tween
 * ends rather than instantly, so anything reading `.open` (the fragment
 * behaviour above, or `e2e`'s own assertions) sees the same state either way.
 *
 * A fragment jump sets `.open` directly and never dispatches a `click` on
 * `<summary>`, so it never runs through this handler and stays instant --
 * exactly the native behaviour T135.5 relied on. Reduced motion, and a
 * browser with no `element.animate`, both fall through to the plain native
 * toggle untouched.
 */
export function Disclosure({
  id,
  summary,
  defaultOpen = false,
  className,
  children,
}: {
  id?: string;
  summary: ReactNode;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const detailsRef = useRef<HTMLDetailsElement>(null);
  const animationRef = useRef<Animation | null>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = detailsRef.current;
    if (el === null || reduced || typeof el.animate !== 'function') return;

    const summaryEl = el.querySelector('summary');
    if (summaryEl === null) return;

    const onClick = (event: MouseEvent) => {
      event.preventDefault();
      animationRef.current?.cancel();

      const opening = !el.open;
      const startHeight = el.getBoundingClientRect().height;
      if (opening) el.open = true; // lays out the full content so scrollHeight below is the real target
      const endHeight = opening ? el.scrollHeight : summaryEl.getBoundingClientRect().height;

      const animation = el.animate(
        { height: [startHeight + 'px', endHeight + 'px'] },
        { duration: 220, easing: 'ease-out' },
      );
      animationRef.current = animation;
      animation.onfinish = () => {
        el.open = opening;
        animationRef.current = null;
      };
    };

    summaryEl.addEventListener('click', onClick);
    return () => summaryEl.removeEventListener('click', onClick);
  }, [reduced]);

  return (
    <details
      ref={detailsRef}
      open={defaultOpen}
      className={cn(SURFACE.card, 'group overflow-hidden', className)}
    >
      <summary
        className={cn(
          'flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3',
          'select-none [&::-webkit-details-marker]:hidden',
          'hover:bg-sunken',
        )}
      >
        <span className={cn(TYPE_SCALE.body, 'font-medium text-ink')}>{summary}</span>
        <Icon name="chevronDown" className="h-4 w-4 shrink-0 text-ink-3 transition-transform group-open:rotate-180" />
      </summary>
      <div className="border-t border-line p-4">
        {id ? <span id={id} className="block scroll-mt-24" aria-hidden="true" /> : null}
        {children}
      </div>
    </details>
  );
}
