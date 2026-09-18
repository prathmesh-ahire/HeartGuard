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
 * The Analyse hero's cursor-follow tilt (redesign, 2026-09-17). Degrees of
 * rotation at the pointer's furthest reported position from centre -- a plain
 * CSS transform on the stage wrapping the 3D canvas, not the canvas' own
 * camera, so it costs one composited layer and never touches `OrbitControls`.
 */
export const HERO_TILT_MAX_DEG = 7;
/** The stage's return-to-rest transition once the pointer leaves it. */
export const HERO_TILT_TRANSITION = 'transform 400ms cubic-bezier(0.22, 1, 0.36, 1)';

/**
 * The top bar's morph into the floating nav pill on scroll (redesign,
 * 2026-09-17). Both bars stay mounted at all times and swap
 * opacity/scale/position on the same symmetric ease-in-out curve -- unlike
 * `EASE_OUT` above, this one spends its motion evenly across the whole
 * duration rather than front-loading it, so the eye can track the shrink/grow
 * the whole way through instead of seeing it snap into its final state in the
 * first fifth of the transition (the bug in the first version of this). 600ms
 * is long enough to read as a deliberate glide without dragging.
 */
export const TOPBAR_NAV_TRANSITION =
  'opacity 600ms cubic-bezier(0.65, 0, 0.35, 1), transform 600ms cubic-bezier(0.65, 0, 0.35, 1)';

/**
 * The home page's white-ground intro (redesign, 2026-09-17): the page opens
 * on a plain white ground, and the first stretch of scroll -- this many
 * pixels of it -- shrinks and fades the hero heart while the white ground
 * fades onto the brand's usual cream/wine surface underneath. One shared
 * distance so `AnalyseHero`'s heart tween and `HomeBackgroundReveal`'s
 * overlay tween, two separate components, read as a single motion: both are
 * `scrub` tweens driven straight off real scroll position, so sharing the
 * distance is what keeps them in step rather than any explicit link between
 * the two components.
 */
export const HOME_INTRO_SCROLL_PX = 640;
/**
 * How small the heart shrinks to by the end of that scroll, from its resting
 * 1 (2026-09-18 fix): growing it used to clip visibly against the hero
 * panel's own `overflow-hidden` edge -- a hard rectangular line cutting across
 * the heart mid-scroll. Shrinking never presses against that edge, because
 * the heart's resting size already fits inside it uncropped.
 */
export const HOME_INTRO_HEART_SCALE = 0.55;

/**
 * The check-selector cards' scroll reveal (redesign, 2026-09-18). Each card
 * in `CheckSelector` grows from this scale and fades in, once, the first
 * time it scrolls into view -- a plain `whileInView`, the same shape
 * `Reveal.tsx` uses everywhere else, not a GSAP `scrub` tied to scroll
 * position.
 *
 * Two GSAP `scrub` versions were tried first, both chasing a dramatic "card 1
 * pushes out card 2" handoff, and both rejected for the same underlying
 * reason: that handoff needs real scroll distance between cards to read, and
 * the only place to find that distance in a normal document-flow stack is to
 * put it there on purpose -- which just moves the problem, since inserted
 * spacing is also what shows up as dead space around a small card. A
 * `min-h-[320px]` slot per card (the second attempt, after a `70vh` first
 * attempt made it far worse) still read as "too much empty space" and was
 * cut entirely. The cards are close together again, and the reveal is a
 * one-shot entrance rather than a scroll-tied push.
 */
/** The card's shrunk-and-faded starting size, before its one-shot reveal. */
export const CHECK_CARD_INTRO_SCALE = 0.92;

/**
 * The shrink/fade/blur target shared by every "small, faint and blurred at
 * rest; sharp at full size when scrolled to" card on the Analyse page --
 * the Step 1 card, and (2026-09-18) Step 2 and the Result card via
 * `useCardScrollReveal`. Kept as three named values, not a single object,
 * for the same reason as every other motion constant here: each is read
 * individually into a `gsap.fromTo`.
 */
export const CARD_SCROLL_REVEAL_SCALE = 0.9;
export const CARD_SCROLL_REVEAL_OPACITY = 0.35;
/** Pixels: `filter: blur()` takes a length, not a bare number. */
export const CARD_SCROLL_REVEAL_BLUR_PX = 4;
// Stronger than the shared CARD_SCROLL_REVEAL_BLUR_PX, and ramped over the
// exit's full shrink duration now (see the `.to()` call's own comment in
// PredictionPanel), not a short fixed stretch -- 14, not 9, because even
// climbing the whole way, 9px read as barely-there next to the intro's 4px
// starting value. Scoped to Step 1's exit only -- the intro and Step 2/
// Result keep the shared 4px.
/** Step 1 card's own exit blur target, stronger than the shared intro/Step-2/Result blur. */
export const STEP1_CARD_EXIT_BLUR_PX = 14;

/**
 * The Step 1 card's own scroll intro (2026-09-18): unlike Step 2 and Result
 * (see `useCardScrollReveal`), this card sits right at the top of the page
 * from the first paint, with no "scrolling up into view" phase to trigger
 * from -- so its intro is a `scrub` tween tied to real page-top scroll
 * position, the same technique as `HOME_INTRO_SCROLL_PX` above, rather than
 * to the card's own position. At the top of the page it is small, faint and
 * blurred; by `STEP1_CARD_SCROLL_PX` of scroll it is at its normal size and
 * fully opaque.
 *
 * The blur runs on its own, shorter `STEP1_CARD_BLUR_SCROLL_PX` (2026-09-18,
 * split out): asked to make it sharpen sooner without changing how the size
 * and opacity settle, which the one shared tween this started as could not
 * do -- scale, opacity and blur were one `fromTo` on one scroll range, so
 * shortening blur's distance would have shortened all three. Two separate
 * `scrub` tweens on the same element, same trigger, different distances.
 *
 * The exit (2026-09-18, same day) reuses `CARD_SCROLL_REVEAL_*` above as its
 * own end state, reversed -- same shrink, same fade, same short blur window
 * -- as the tail of the SAME timeline the intro runs on, not a second
 * `ScrollTrigger`. A separate exit `ScrollTrigger` was tried first and
 * fought the intro's wherever the two came close in scroll terms (the card
 * sits high enough on the page that "intro finished" and "exit starting"
 * are not far apart); see `PredictionPanel`'s own comment for the fix.
 */
// 200, not the original 520: the user felt it stayed blurred until almost
// halfway down the page, so it now resolves within the first small stretch
// of scroll rather than over a distance that reads as "most of a screen".
export const STEP1_CARD_SCROLL_PX = 200;
/** Shorter than `STEP1_CARD_SCROLL_PX`: the blur alone clears sooner. */
export const STEP1_CARD_BLUR_SCROLL_PX = 90;
/**
 * The exit's blur starts this many pixels into the card's own exit transit,
 * after the shrink has already begun (2026-09-18): both used to start at the
 * same instant ('top top'), which read as the blur arriving too eagerly,
 * racing the shrink instead of following it. The shrink still starts
 * immediately; only the blur's own trigger is offset by this much.
 */
// 28, not 12: still read as too early once the stale-build confusion was
// resolved and the real ramp was visible -- pushed later again so the
// shrink has a clearer head start before the blur follows.
export const STEP1_CARD_EXIT_BLUR_DELAY_PX = 28;

/**
 * `useCardScrollReveal`'s own phase timing (2026-09-18), for Step 2 and the
 * Result card: unlike Step 1, both start below the fold, so their whole
 * intro-hold-exit cycle can run off ONE natural trigger -- the card's own
 * transit through the viewport, `'top bottom'` to `'bottom top'` -- with no
 * page-absolute distance to pick. Expressed as fractions of that transit
 * (0 to 1) rather than pixels, because GSAP maps a scrub timeline's own
 * duration onto whatever real scroll distance the trigger resolves to,
 * whatever that turns out to be for a given card and viewport.
 *
 * Symmetric by construction: the exit starts at
 * `1 - CARD_SCROLL_REVEAL_INTRO_FRACTION` and runs for the same length the
 * intro did, so entering and leaving read as the same motion in reverse.
 */
export const CARD_SCROLL_REVEAL_INTRO_FRACTION = 0.12;
// 0.09 for the intro specifically (2026-09-18, split from the exit's own
// value below): the user felt Step 2/Result's intro blur was weaker than
// Step 1's, even though both peak at the same CARD_SCROLL_REVEAL_BLUR_PX --
// the exit fraction (0.05) cleared it too fast relative to these cards' own
// much longer transit (top-bottom-of-viewport to bottom-top, hundreds of
// pixels more than Step 1's fixed 200px intro) to read as deliberate while
// scrolling. The exit itself was fine at 0.05, so only the intro moved.
export const CARD_SCROLL_REVEAL_INTRO_BLUR_FRACTION = 0.09;
/** The exit's own blur-clear fraction -- shorter, since the exit already read right. */
export const CARD_SCROLL_REVEAL_BLUR_FRACTION = 0.05;
// 0.003, not 0.01, for the same reason as STEP1_CARD_EXIT_BLUR_DELAY_PX's own
// third reduction: kept proportional to it so Step 1 and Step 2/Result still
// read as the same motion.
/** Same idea as `STEP1_CARD_EXIT_BLUR_DELAY_PX`, as a fraction of the transit. */
export const CARD_SCROLL_REVEAL_EXIT_BLUR_DELAY_FRACTION = 0.003;

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
