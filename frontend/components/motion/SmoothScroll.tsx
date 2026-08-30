'use client';

import Lenis from 'lenis';
import { useEffect } from 'react';

import { useReducedMotion } from '@/lib/capability';

/**
 * Lenis smooth scrolling, and the one setting that makes it acceptable (T112.3).
 *
 * Smooth scroll hijacks a browser behaviour people rely on. Under
 * `prefers-reduced-motion: reduce` it is not started at all -- not started
 * gently, not started with a shorter duration. Someone who sets that flag is
 * frequently telling the page that eased scrolling makes them ill.
 *
 * It also drives GSAP's ScrollTrigger: Lenis moves the page on its own
 * schedule, so ScrollTrigger has to be told to read scroll position from Lenis
 * rather than from the native scroll event, or the pinned sections lag the
 * content by a frame or two and look broken.
 */

export function SmoothScroll({ children }: { children: React.ReactNode }) {
  const reduced = useReducedMotion();

  useEffect(() => {
    if (reduced) return;

    const lenis = new Lenis({
      duration: 0.9,
      // A short, nearly-linear ease. A long one feels like the page is
      // resisting the wheel, which reads as lag rather than polish.
      easing: (t: number) => 1 - Math.pow(1 - t, 3),
      smoothWheel: true,
      touchMultiplier: 1.4,
    });

    let frame = 0;
    const raf = (time: number): void => {
      lenis.raf(time);
      frame = requestAnimationFrame(raf);
    };
    frame = requestAnimationFrame(raf);

    // ScrollTrigger, if it is on the page, must read Lenis's position.
    let detach: (() => void) | undefined;
    void (async () => {
      const { ScrollTrigger } = await import('gsap/ScrollTrigger');
      const update = (): void => ScrollTrigger.update();
      lenis.on('scroll', update);
      detach = () => lenis.off('scroll', update);
      ScrollTrigger.refresh();
    })();

    return () => {
      cancelAnimationFrame(frame);
      detach?.();
      lenis.destroy();
    };
  }, [reduced]);

  return <>{children}</>;
}
