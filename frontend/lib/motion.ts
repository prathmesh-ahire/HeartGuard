import type { Transition } from 'framer-motion';

/**
 * Motion constants (T128.5), shared so a reveal, a stagger and a page
 * transition move with one character rather than three.
 *
 * Type-only import from framer-motion: nothing here adds to a bundle.
 */

/** A fast start and a long settle: movement that arrives rather than slides. */
export const EASE_OUT: [number, number, number, number] = [0.16, 1, 0.3, 1];

/** Seconds. */
export const DURATION = {
  quick: 0.18,
  base: 0.35,
  page: 0.22,
} as const;

/** Seconds between siblings in a staggered reveal. */
export const STAGGER_STEP = 0.06;

/**
 * The one rule every animated primitive follows: under reduced motion the
 * transition is a state change, not an animation. The start and end states
 * never change -- only how long it takes to get between them -- so the server
 * markup and the first client render stay identical (see `Reveal.tsx`).
 */
export function transitionFor(reduced: boolean, transition: Transition): Transition {
  return reduced ? { duration: 0, delay: 0 } : transition;
}
