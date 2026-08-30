'use client';

import { useEffect, useRef, useState } from 'react';

import { cn } from '@/lib/cn';
import { TYPE_SCALE } from '@/lib/tokens';

/**
 * A count that animates up to its final value on first view.
 *
 * **It takes both the number and the finished string, and it always settles on
 * the string.** `value` drives the animation; `display` is what Python
 * formatted, and it is what the reader ends up looking at. The intermediate
 * frames are a transition, not a reported figure, and the component makes that
 * literal: the moment the animation completes it renders `display` verbatim,
 * and `prefers-reduced-motion` or a missing `value` skips straight to it.
 *
 * **Only for counts.** A metric is not animated at all. Counting 0.000 -> 0.857
 * would mean formatting a metric in the browser on every frame, which is the
 * one thing the client must never do; `StatTile` reveals metrics without
 * touching their digits.
 */
export function AnimatedCounter({
  value,
  display,
  durationMs = 900,
  className,
}: {
  value: number | null;
  display: string;
  durationMs?: number;
  className?: string;
}) {
  const [frame, setFrame] = useState<string | null>(null);
  const node = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const element = node.current;
    if (element === null || value === null || !Number.isFinite(value)) return;

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) return;

    let raf = 0;
    let start: number | null = null;
    const integers = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

    const step = (now: number) => {
      start ??= now;
      const progress = Math.min((now - start) / durationMs, 1);
      // Ease-out cubic: fast first, so the number is readable most of the time.
      const eased = 1 - Math.pow(1 - progress, 3);
      if (progress >= 1) {
        setFrame(null); // settle on the Python-formatted string
        return;
      }
      setFrame(integers.format(value * eased));
      raf = window.requestAnimationFrame(step);
    };

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          raf = window.requestAnimationFrame(step);
        }
      },
      { threshold: 0.25 },
    );
    observer.observe(element);

    return () => {
      observer.disconnect();
      window.cancelAnimationFrame(raf);
    };
  }, [value, durationMs]);

  return (
    <span ref={node} className={cn(TYPE_SCALE.stat, className)}>
      {frame ?? display}
    </span>
  );
}
