'use client';

import { LazyMotion, m, useReducedMotion as useFramerReducedMotion } from 'framer-motion';
import type { ReactNode } from 'react';

const loadFeatures = () => import('./features').then((mod) => mod.default);

/**
 * Card reveals and page transitions (T112.5).
 *
 * Framer Motion's own `useReducedMotion` is used here rather than the project's
 * hook in `lib/capability.ts`. They answer the same question, but Framer's
 * value is what its `MotionConfig` and layout animations consult internally, so
 * using it keeps one source of truth inside the animation library instead of
 * two that can disagree for a frame after the OS setting changes.
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
  const reduced = useFramerReducedMotion();

  if (reduced === true) {
    return <div className={className}>{children}</div>;
  }

  return (
    <LazyMotion features={loadFeatures} strict>
      <m.div
        className={className}
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        // metric-guard: allow -- viewport geometry, not a measurement
      viewport={{ once: true, margin: '0px 0px -10% 0px' }}
        transition={{ duration: 0.35, delay, ease: [0.16, 1, 0.3, 1] }}
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
  const reduced = useFramerReducedMotion();

  if (reduced === true) {
    return <>{children}</>;
  }

  return (
    <LazyMotion features={loadFeatures} strict>
      <m.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.22, ease: 'easeOut' }}
      >
        {children}
      </m.div>
    </LazyMotion>
  );
}
