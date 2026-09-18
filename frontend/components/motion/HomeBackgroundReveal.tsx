'use client';

import { useEffect, useRef } from 'react';

import { useReducedMotion } from '@/lib/capability';
import { HOME_INTRO_SCROLL_PX } from '@/lib/motion';

/**
 * The home page's white-to-brand-ground reveal (redesign, 2026-09-17): a
 * fixed white sheet sits behind the page's own content, from the very first
 * paint, and fades out over `HOME_INTRO_SCROLL_PX` of scroll -- so the page
 * opens on plain white and settles onto the usual cream/wine ground as the
 * hero heart (see `AnalyseHero`) grows and fades over the same distance.
 *
 * `-z-10`, same as the hero's own ambient glow: normal in-flow content
 * (everything on the page) paints above a negative-z sibling regardless of
 * DOM order, so this sits behind text and cards without needing them to
 * declare a z-index of their own.
 *
 * Light mode only (`dark:hidden`): dark mode's ground is already the wine
 * taken near-black, so there is no white-opening moment to stage there, and
 * this never mounts under reduced motion -- a scroll-scrubbed reveal is
 * exactly the kind of motion that setting exists to refuse, and skipping it
 * leaves the page on its ordinary cream ground rather than stuck white.
 */
export function HomeBackgroundReveal() {
  const reduced = useReducedMotion();
  const overlay = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (reduced) return;
    const el = overlay.current;
    if (el === null) return;
    let cleanup: (() => void) | undefined;

    void (async () => {
      const [{ default: gsap }, { ScrollTrigger }] = await Promise.all([
        import('gsap'),
        import('gsap/ScrollTrigger'),
      ]);
      gsap.registerPlugin(ScrollTrigger);

      const tween = gsap.to(el, {
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
      };
    })();

    return () => cleanup?.();
  }, [reduced]);

  if (reduced) return null;

  return (
    <div
      ref={overlay}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 bg-panel dark:hidden"
    />
  );
}
