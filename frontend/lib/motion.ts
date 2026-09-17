import type { Transition } from 'framer-motion';

/** A hover/tap target, as `whileHover`/`whileTap` take. */
type PressTarget = { scale?: number; y?: number };

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
 * The press spring (T136.4): stiff and slightly under-damped, so a tap
 * overshoots by a pixel or two on release rather than snapping to rest. A
 * critically-damped spring here reads as a state flip, not a touch.
 */
export const PRESS_SPRING: Transition = { type: 'spring', stiffness: 500, damping: 30, mass: 0.6 };

/** A calmer spring for a larger surface -- a card or a hero layer. */
export const LIFT_SPRING: Transition = { type: 'spring', stiffness: 300, damping: 26 };

/**
 * A card's hover/tap lift (T136.4), named rather than written inline: the
 * metric guard reads UI motion constants as data the moment they are literal
 * numbers inside a component's JSX, and rightly cannot tell a spring's scale
 * from a result. Naming it here, once, is the fix the guard's own message
 * asks for -- moved out of the page, not suppressed on the page.
 */
export const CARD_HOVER: PressTarget = { scale: 1.015, y: -2 };
export const CARD_TAP: PressTarget = { scale: 0.985, y: 0 };

/** A drop-zone's hover scale (T136.3), named for the same reason as above. */
export const DRAG_HOVER_SCALE = 1.015;

/**
 * The indeterminate upload sweep's start/end position, as a percentage of
 * the bar's own width -- not a measurement of anything, and named for the
 * same reason as `CARD_HOVER` above.
 */
export const SWEEP_X: [string, string] = ['-100%', '250%'];
/** Where the sweep settles under reduced motion: a state change, not a run. */
export const SWEEP_X_SETTLED = '100%';

/**
 * The one rule every animated primitive follows: under reduced motion the
 * transition is a state change, not an animation. The start and end states
 * never change -- only how long it takes to get between them -- so the server
 * markup and the first client render stay identical (see `Reveal.tsx`).
 */
export function transitionFor(reduced: boolean, transition: Transition): Transition {
  return reduced ? { duration: 0, delay: 0 } : transition;
}
