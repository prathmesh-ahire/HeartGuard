'use client';

import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import {
  HERO_TILT_MAX_DEG,
  HERO_TILT_TRANSITION,
  HOME_INTRO_HEART_SCALE,
  HOME_INTRO_SCROLL_PX,
} from '@/lib/motion';
import { Hero3D } from '@/components/three/Hero3D';

/**
 * The Analyse hero (T136.5, redesigned 2026-09-17): the 3D heart, sharing one
 * bento panel with the page title in `app/page.tsx` rather than sitting in its
 * own bordered box above it. Trying this placement was tried once and
 * reverted -- see the 2026-09-12 Phase 127 note -- so this component, and
 * removing `three.js` from `NOT_YET_BUNDLED` in
 * `scripts/20_check_bundle_budget.py`, is what closes that out.
 *
 * Three motion layers, deliberately kept apart:
 *
 * 1. **Scroll depth** (T136.1) -- a GSAP ScrollTrigger `scrub` parallax on
 *    `layer`, the same async-import-then-bail-under-reduced-motion shape as
 *    `PipelineWalkthrough.tsx`. Drifts a few percent slower than the page as
 *    it scrolls past; not mounted at all under reduced motion.
 * 2. **White-ground intro** (redesign, 2026-09-17, shrink direction fixed
 *    2026-09-18) -- a second `scrub` tween on the same `layer`, tied to real
 *    scroll position rather than to when the element enters the viewport:
 *    over `HOME_INTRO_SCROLL_PX` of scroll from the very top of the page, the
 *    heart shrinks to `HOME_INTRO_HEART_SCALE` and fades to nothing, in step
 *    with `HomeBackgroundReveal`'s white sheet fading out behind it (the two
 *    are not linked directly -- both are scrub tweens reading the same real
 *    scroll position, over the same named distance, which is what keeps them
 *    in step). Shrinking, not growing: growing used to clip against the hero
 *    panel's own `overflow-hidden` edge, a hard box-shaped line cutting
 *    across the heart mid-scroll.
 * 3. **Cursor tilt** (redesign) -- a plain CSS 3D transform on `stage`, the
 *    div wrapping the canvas, driven by `pointermove`. This is NOT
 *    `OrbitControls` (still off, per `Hero3D`'s `interactive` default): it
 *    rotates the whole stage as one composited layer rather than the scene's
 *    own camera, and `pointermove` never calls `preventDefault`, so it cannot
 *    trap wheel or touch scroll the way a drag-to-orbit control would. Mouse
 *    only -- `pointerType !== 'mouse'` bails, so a touch tap does not leave
 *    the heart tilted toward wherever it was last touched.
 *
 * The ambient glow behind the heart is the same motif the hero panel's own
 * background carries in `app/page.tsx`, just local and slightly stronger, so
 * the heart reads as the panel's glow made solid rather than a second light
 * source competing with it.
 */
export function AnalyseHero({ className }: { className?: string }) {
  const reduced = useReducedMotion();
  const layer = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [tilt, setTilt] = useState({ rotateX: 0, rotateY: 0 });

  useEffect(() => {
    if (reduced) return;
    const el = layer.current;
    if (el === null) return;
    let cleanup: (() => void) | undefined;

    void (async () => {
      const [{ default: gsap }, { ScrollTrigger }] = await Promise.all([
        import('gsap'),
        import('gsap/ScrollTrigger'),
      ]);
      gsap.registerPlugin(ScrollTrigger);

      const tween = gsap.to(el, {
        yPercent: 12,
        ease: 'none',
        scrollTrigger: {
          trigger: el,
          start: 'top bottom',
          end: 'bottom top',
          scrub: true,
        },
      });

      const intro = gsap.to(el, {
        scale: HOME_INTRO_HEART_SCALE,
        opacity: 0,
        ease: 'none',
        scrollTrigger: {
          trigger: document.body,
          start: 'top top',
          end: `+=${HOME_INTRO_SCROLL_PX}`,
          scrub: true,
        },
      });

      cleanup = () => {
        tween.scrollTrigger?.kill();
        tween.kill();
        intro.scrollTrigger?.kill();
        intro.kill();
      };
    })();

    return () => cleanup?.();
  }, [reduced]);

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (reduced || event.pointerType !== 'mouse') return;
    const bounds = stage.current?.getBoundingClientRect();
    if (bounds === undefined) return;
    const px = (event.clientX - bounds.left) / bounds.width - 0.5;
    const py = (event.clientY - bounds.top) / bounds.height - 0.5;
    setTilt({ rotateX: py * -HERO_TILT_MAX_DEG, rotateY: px * HERO_TILT_MAX_DEG });
  };

  const resetTilt = () => setTilt({ rotateX: 0, rotateY: 0 });

  return (
    <div className={cn('relative', className)}>
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_closest-side_at_50%_45%,rgb(var(--accent)/0.28),transparent_75%)] blur-2xl motion-safe:animate-pulse-subtle"
      />
      <div ref={layer}>
        <div
          className="mx-auto h-64 w-full max-w-sm sm:h-72 lg:h-80"
          style={{ perspective: '800px' }}
        >
          <div
            ref={stage}
            onPointerMove={handlePointerMove}
            onPointerLeave={resetTilt}
            style={{
              transform: `rotateX(${tilt.rotateX}deg) rotateY(${tilt.rotateY}deg)`,
              transition: HERO_TILT_TRANSITION,
            }}
            className="h-full w-full"
          >
            <Hero3D height="100%" className="h-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
