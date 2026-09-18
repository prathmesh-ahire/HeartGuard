'use client';

import { useEffect, useRef, useState } from 'react';

import { cn } from '@/lib/cn';

/**
 * A bar that grows from 0 to its target width the first time it scrolls into
 * view, then stays put (T139.1).
 *
 * One element, styled by the caller exactly as a plain `<span style={{
 * width }}>` bar was before it -- `percent` resolves against whatever the
 * caller's layout already gives it (a flex row, a grid track), same as a
 * static width would. `percent` is geometry the caller already computed from
 * a real value (an importance score against its group's largest, a share of
 * a total) -- the same rule `ImportanceView` documents for its own bar: length
 * may be derived from a number, but the only text ever rendered next to it is
 * a Python-formatted `_display` string. This component draws no text at all.
 *
 * Reduced motion (or a missing IntersectionObserver) renders at the final
 * width immediately -- same start and end state as `AnimatedCounter`, no
 * animation, no `duration: 0` needed because there is nothing to race.
 */
export function GrowBar({
  percent,
  color,
  className,
  durationMs = 700,
}: {
  /** Target width, as a percentage of the element's own containing box. */
  percent: number;
  color?: string;
  className?: string;
  durationMs?: number;
}) {
  const node = useRef<HTMLSpanElement>(null);
  const [grown, setGrown] = useState(false);

  useEffect(() => {
    const element = node.current;
    if (element === null) return;

    if (
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      setGrown(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setGrown(true);
          observer.disconnect();
        }
      },
      { threshold: 0.25 },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const clamped = Math.max(0, Math.min(100, percent));

  return (
    <span
      ref={node}
      aria-hidden="true"
      className={cn('block h-3 rounded-sm ease-out', className)}
      style={{
        width: (grown ? clamped : 0) + '%',
        backgroundColor: color,
        transitionProperty: 'width',
        transitionDuration: durationMs + 'ms',
      }}
    />
  );
}
