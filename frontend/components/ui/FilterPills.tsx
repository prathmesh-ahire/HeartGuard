'use client';

import { useEffect, useRef, useState } from 'react';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';

export interface FilterPillOption {
  id: string;
  label: string;
}

/**
 * A pill-group filter with a sliding highlight (T142.6).
 *
 * The highlight is measured `left`/`width` on a plain CSS transition -- the
 * same technique `TopBar`'s `NavLinks` uses for its own active-item pill
 * (T139.5) -- rather than framer's `layout`/`layoutId`, which this project
 * has twice deliberately avoided because it pulls in the heavier `domMax`
 * feature bundle for a bundle-budget-sensitive page. Selecting a pill is a
 * plain controlled `value`/`onChange`; it decides nothing about what the
 * options mean.
 */
export function FilterPills({
  options,
  value,
  onChange,
  ariaLabel,
  className,
}: {
  options: readonly FilterPillOption[];
  value: string;
  onChange: (id: string) => void;
  ariaLabel: string;
  className?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLButtonElement>(null);
  const reduced = useReducedMotion();
  const [pill, setPill] = useState<{ left: number; width: number } | null>(null);

  useEffect(() => {
    const measure = (): void => {
      const container = containerRef.current;
      const active = activeRef.current;
      if (container === null || active === null) {
        setPill(null);
        return;
      }
      const containerBox = container.getBoundingClientRect();
      const activeBox = active.getBoundingClientRect();
      setPill({ left: activeBox.left - containerBox.left, width: activeBox.width });
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [value, options.length]);

  return (
    <div
      ref={containerRef}
      role="group"
      aria-label={ariaLabel}
      className={cn(
        'relative flex flex-wrap items-center gap-1 rounded-full border border-line bg-sunken p-1',
        className,
      )}
    >
      {pill ? (
        <span
          aria-hidden="true"
          className="absolute inset-y-1 rounded-full bg-accent"
          style={{
            left: pill.left,
            width: pill.width,
            transitionProperty: 'left, width',
            transitionDuration: reduced ? '0ms' : '280ms',
            transitionTimingFunction: 'cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        />
      ) : null}
      {options.map((option) => {
        const active = option.id === value;
        return (
          <button
            key={option.id}
            type="button"
            ref={active ? activeRef : undefined}
            aria-pressed={active}
            onClick={() => onChange(option.id)}
            className={cn(
              'relative z-10 whitespace-nowrap rounded-full px-3 py-1.5 text-label-md uppercase transition-colors',
              active ? 'text-on-accent' : 'text-ink-2 hover:text-accent-strong',
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
