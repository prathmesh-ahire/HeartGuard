'use client';

import { useScrolled } from '@/lib/capability';
import { cn } from '@/lib/cn';

/**
 * A small "there's more below" cue (T139.6): a bouncing chevron over a short
 * line, faded out once the visitor actually scrolls.
 *
 * Purely decorative -- `aria-hidden`, nothing focusable -- so a keyboard or
 * screen-reader visit never has to pass through it to reach the next
 * section. `animate-bounce` is Tailwind's own utility, so it only needs the
 * usual `motion-safe:` guard, not a JS reduced-motion read.
 */
export function ScrollCue({ className }: { className?: string }) {
  const scrolled = useScrolled(40);

  return (
    <div
      aria-hidden="true"
      className={cn(
        'flex flex-col items-center gap-1.5 text-ink-3 transition-opacity duration-500',
        scrolled ? 'pointer-events-none opacity-0' : 'opacity-100',
        className,
      )}
    >
      <span className="text-label-sm uppercase tracking-widest">Scroll</span>
      <svg
        viewBox="0 0 16 24"
        className="h-6 w-4 motion-safe:animate-bounce"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
      >
        <path d="M8 1v18" strokeLinecap="round" />
        <path d="M2 13l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}
