'use client';

import { LazyMotion, m } from 'framer-motion';
import type { ReactNode } from 'react';

import { useReducedMotion } from '@/lib/capability';

const loadFeatures = () => import('./features').then((mod) => mod.default);

/**
 * Card reveals and page transitions (T112.5).
 *
 * The reduced-motion read is the project's own hook in `lib/capability.ts`, NOT
 * Framer Motion's. Framer's reads `matchMedia` synchronously during the first
 * render, so under `prefers-reduced-motion: reduce` the client's first render
 * emits a plain `<div>` where the static export emitted this motion wrapper --
 * which is a hydration mismatch, and React threw #418/#423 on EVERY page for
 * every visitor with that setting on. The project's hook reads in an effect, so
 * the first client render matches the server and the switch happens after
 * mount. Found by the Phase 120 screenshot run, which is the first thing in the
 * project to open the site with reduced motion forced on; the Phase 118 smoke
 * test runs with the default setting and never saw it.
 *
 * For the same reason the tree is the SAME shape either way: reduced motion
 * turns the animation off through `initial`/`transition`, it does not swap the
 * wrapper for a plain `<div>`. Swapping it meant `LazyMotion` mounted, started
 * its async feature import, and unmounted one tick later on every reveal on the
 * page -- which showed up once as a hard client-side exception on the home page
 * during a capture run and passed on the retry. A gate that fails one run in
 * five is worse than no gate.
 *
 * Under reduced motion the reveal is driven by `animate` rather than
 * `whileInView`: content must not wait on an IntersectionObserver for someone
 * who asked for less motion, and `duration: 0` makes it a state change rather
 * than an animation.
 *
 * Everything animated goes through `LazyMotion` and the `m` component rather
 * than `motion`. `motion.div` drags the whole DOM feature set into the chunk
 * that imports it, and the page transition lives in `app/template.tsx`, so that
 * chunk is one every single route downloads. `m` plus an async feature loader
 * moves 34 kB gzipped off the critical path of pages that never animate
 * anything. `strict` makes the cheap path the only path: using `motion` under
 * it throws rather than silently re-bundling what was just split out.
 *
 * **A reveal must never gate content on the animation completing.** Everything
 * below animates opacity and a small translate, and starts from an opacity that
 * is already readable-adjacent; nothing is `display: none` until a viewport
 * callback fires. If the IntersectionObserver never runs -- a print stylesheet,
 * a headless screenshot, a browser extension -- the page still reads.
 */

export function Reveal({
  children,
  delay = 0,
  className,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const reduced = useReducedMotion();

  return (
    <LazyMotion features={loadFeatures} strict>
      <m.div
        className={className}
        // `initial` is the SAME on the server, on the first client render and
        // under reduced motion. It is what the static HTML carries, so making
        // it conditional is a hydration mismatch -- and making it `false` while
        // `whileInView` was also dropped left every section stuck at the opacity
        // the server wrote, which is a blank page. Reduced motion changes the
        // DURATION and the trigger, never the start or the end state.
        initial={{ opacity: 0, y: 12 }}
        animate={reduced ? { opacity: 1, y: 0 } : undefined}
        whileInView={reduced ? undefined : { opacity: 1, y: 0 }}
        // metric-guard: allow -- viewport geometry, not a measurement
      viewport={{ once: true, margin: '0px 0px -10% 0px' }}
        transition={reduced ? { duration: 0 } : { duration: 0.35, delay, ease: [0.16, 1, 0.3, 1] }}
      >
        {children}
      </m.div>
    </LazyMotion>
  );
}

/**
 * Staggered children. `index` rather than Framer's `staggerChildren` because
 * the reveals are independent viewport triggers, not one orchestrated sequence:
 * a card ten screens down should animate when it is reached, not on a delay
 * counted from when the first card appeared.
 */
export function RevealList({
  children,
  className,
  step = 0.05,
}: {
  children: ReactNode[];
  className?: string;
  step?: number;
}) {
  return (
    <div className={className}>
      {children.map((child, index) => (
        <Reveal key={index} delay={Math.min(index, 6) * step}>
          {child}
        </Reveal>
      ))}
    </div>
  );
}

/**
 * The route transition, mounted from `app/template.tsx`.
 *
 * `template.tsx` rather than `layout.tsx` on purpose: Next remounts a template
 * on every navigation and preserves a layout, and a transition that never
 * remounts never plays. The transition is a short fade with no movement --
 * a sliding page under a static export means the browser has already replaced
 * the document, and animating position after that reads as a stutter.
 */
export function PageTransition({ children }: { children: ReactNode }) {
  const reduced = useReducedMotion();

  return (
    <LazyMotion features={loadFeatures} strict>
      <m.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={reduced ? { duration: 0 } : { duration: 0.22, ease: 'easeOut' }}
      >
        {children}
      </m.div>
    </LazyMotion>
  );
}
