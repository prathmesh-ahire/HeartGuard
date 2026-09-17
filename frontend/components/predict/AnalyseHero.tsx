'use client';

import { useEffect, useRef } from 'react';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { Hero3D } from '@/components/three/Hero3D';

/**
 * The Analyse hero (T136.5): the 3D heart, reinstated on Analyse after the
 * Part XII restructure dropped it. Trying this placement earlier was reverted
 * on purpose -- see the 2026-09-12 Phase 127 note ("Putting the heart on
 * Analyse now was tried and reverted: that placement is T136.5's job") -- so
 * this component, and removing `three.js` from `NOT_YET_BUNDLED` in
 * `scripts/20_check_bundle_budget.py`, is what closes that out.
 *
 * Non-interactive (`Hero3D`'s default): dragging the model would fight the
 * page's own scroll and the note that shipped it says a hero must not trap
 * the scroll.
 *
 * "Depth" (T136.1) is a GSAP ScrollTrigger `scrub` parallax, the same
 * async-import-then-bail-under-reduced-motion shape as
 * `PipelineWalkthrough.tsx` -- the layer drifts a few percent slower than the
 * page as it scrolls past rather than pinning or animating in, and is not
 * mounted at all under reduced motion.
 */
export function AnalyseHero({ className }: { className?: string }) {
  const reduced = useReducedMotion();
  const layer = useRef<HTMLDivElement>(null);

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

      cleanup = () => {
        tween.scrollTrigger?.kill();
        tween.kill();
      };
    })();

    return () => cleanup?.();
  }, [reduced]);

  return (
    <div
      className={cn('relative overflow-hidden rounded-2xl border border-line bg-panel', className)}
    >
      <div ref={layer} aria-hidden="true" className="pointer-events-none">
        <Hero3D height="13rem" className="mx-auto max-w-sm" />
      </div>
    </div>
  );
}
