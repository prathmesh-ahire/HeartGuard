'use client';

import { useEffect, type RefObject } from 'react';

import { useReducedMotion } from '@/lib/capability';
import {
  CARD_SCROLL_REVEAL_BLUR_FRACTION,
  CARD_SCROLL_REVEAL_BLUR_PX,
  CARD_SCROLL_REVEAL_EXIT_BLUR_DELAY_FRACTION,
  CARD_SCROLL_REVEAL_INTRO_BLUR_FRACTION,
  CARD_SCROLL_REVEAL_INTRO_FRACTION,
  CARD_SCROLL_REVEAL_OPACITY,
  CARD_SCROLL_REVEAL_SCALE,
} from '@/lib/motion';

/**
 * The shrink/fade/blur reveal Step 2 and the Result card share (2026-09-18,
 * extracted from the Step 1 card's own hand-built version once the same
 * effect was asked for on both): small, faint and blurred below the fold;
 * full size, opaque and sharp while passing through the viewport; the same
 * reversed as it scrolls past the top.
 *
 * Unlike the Step 1 card (see `STEP1_CARD_SCROLL_PX`'s own comment in
 * `lib/motion.ts`), this one card's own full transit through the viewport
 * -- `'top bottom'` to `'bottom top'` -- carries all three phases, because
 * both callers start below the fold and so have a natural "scrolling up
 * into view" distance to trigger the intro from, which the Step 1 card
 * (visible from the first paint) does not.
 *
 * One timeline per property (scale+opacity, and separately filter), not one
 * timeline for everything or a second `ScrollTrigger` for the exit: the
 * Step 1 card hit exactly that bug -- two tweens both writing `filter`
 * fighting each other -- and the fix was one owner per property across the
 * whole intro-hold-exit journey. See `PredictionPanel`'s own comment on its
 * Step 1 effect for the full story.
 */
export function useCardScrollReveal(ref: RefObject<HTMLElement | null>): void {
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = ref.current;
    if (el === null) return;

    // `reduced` (this project's own hook) starts `false` and only flips
    // after ITS OWN effect reads `matchMedia` and re-renders. On a page
    // where reduced motion is already on from the very first paint --
    // Playwright's `contextOptions.reducedMotion: 'reduce'`, or a real
    // visitor's OS setting -- this effect could still run once with
    // `reduced` stale at `false`. GSAP's `fromTo` below writes its "from"
    // values (opacity < 1) to the element's inline style SYNCHRONOUSLY on
    // creation, before the later re-render tears it down; `.kill()` stops
    // the tween but never reverts an inline style it already wrote, so that
    // first frame's faded opacity stuck on the element forever -- caught by
    // T143.3's screenshot suite as a `<section>` frozen at opacity 0.35.
    // Checking `matchMedia` directly here, in addition to `reduced`, closes
    // the race, and clearing the three properties this hook ever sets
    // recovers an element a previous mount already left mid-tween.
    const prefersReduced =
      reduced ||
      (typeof window.matchMedia === 'function' &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    if (prefersReduced) {
      el.style.removeProperty('opacity');
      el.style.removeProperty('transform');
      el.style.removeProperty('filter');
      return;
    }
    let cleanup: (() => void) | undefined;

    void (async () => {
      const [{ default: gsap }, { ScrollTrigger }] = await Promise.all([
        import('gsap'),
        import('gsap/ScrollTrigger'),
      ]);
      gsap.registerPlugin(ScrollTrigger);

      const exitStart = 1 - CARD_SCROLL_REVEAL_INTRO_FRACTION;

      const sizeTl = gsap.timeline({
        defaults: { ease: 'none' },
        scrollTrigger: { trigger: el, start: 'top bottom', end: 'bottom top', scrub: true },
      });
      sizeTl
        .fromTo(
          el,
          { scale: CARD_SCROLL_REVEAL_SCALE, opacity: CARD_SCROLL_REVEAL_OPACITY },
          { scale: 1, opacity: 1, duration: CARD_SCROLL_REVEAL_INTRO_FRACTION },
          0,
        )
        .to(
          el,
          { scale: CARD_SCROLL_REVEAL_SCALE, opacity: CARD_SCROLL_REVEAL_OPACITY, duration: CARD_SCROLL_REVEAL_INTRO_FRACTION },
          exitStart,
        );

      const blurTl = gsap.timeline({
        defaults: { ease: 'none' },
        scrollTrigger: { trigger: el, start: 'top bottom', end: 'bottom top', scrub: true },
      });
      blurTl
        .fromTo(
          el,
          { filter: `blur(${CARD_SCROLL_REVEAL_BLUR_PX}px)` },
          { filter: 'blur(0px)', duration: CARD_SCROLL_REVEAL_INTRO_BLUR_FRACTION },
          0,
        )
        .to(
          el,
          { filter: `blur(${CARD_SCROLL_REVEAL_BLUR_PX}px)`, duration: CARD_SCROLL_REVEAL_BLUR_FRACTION },
          exitStart + CARD_SCROLL_REVEAL_EXIT_BLUR_DELAY_FRACTION,
        );

      // Re-measure whenever the card's own content changes height
      // (2026-09-18): both `ScrollTrigger`s above compute their start/end
      // from `el`'s height once, when they are built here. The Result card
      // starts as the short "Nothing scored yet" placeholder and grows a lot
      // once an actual result renders -- behind an `AnimatePresence
      // mode="wait"` swap, so the new height lands on its own schedule, not
      // in sync with the React state change that caused it. The stale range
      // from the short version made the exit blur finish while most of the
      // now-taller card was still on screen, well before it had actually
      // scrolled far enough. A `ResizeObserver` reacts to the real DOM
      // change directly instead of guessing at a delay, and covers Step 2
      // the same way if its own content ever grows (a batch's rows, an
      // error banner) without needing its own fix later.
      const resize = new ResizeObserver(() => ScrollTrigger.refresh());
      resize.observe(el);

      cleanup = () => {
        resize.disconnect();
        sizeTl.scrollTrigger?.kill();
        sizeTl.kill();
        blurTl.scrollTrigger?.kill();
        blurTl.kill();
      };
    })();

    return () => cleanup?.();
  }, [reduced, ref]);
}
