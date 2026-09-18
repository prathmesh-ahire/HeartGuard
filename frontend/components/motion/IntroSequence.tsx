'use client';

import { AnimatePresence, LazyMotion, m } from 'framer-motion';
import { useEffect, useState } from 'react';

import { useReducedMotion } from '@/lib/capability';

const loadFeatures = () => import('@/components/motion/features').then((mod) => mod.default);

const SESSION_KEY = 'pv-intro-seen';

/** Seconds. Each of the four scheme colours holds the screen for one step. */
const STEP = 0.28;
/** Seconds the last (accent) layer and its mark hold before auto-dismissing. */
const HOLD = 0.7;
const TOTAL_MS = (STEP * 4 + HOLD) * 1000;

/** In DOM order, so each later layer's fade-in visually covers the one before it. */
const LAYERS = ['bg-surface', 'bg-line', 'bg-accent-soft', 'bg-accent'] as const;

/**
 * The intro brand sequence (T136.2): the four scheme colours in turn, a
 * heart mark on the last one, then a fade to the site. Once per browser
 * session (`sessionStorage`), skippable, and never shown at all under three
 * conditions checked before the first paint of it:
 *
 * 1. **`prefers-reduced-motion: reduce`.** Not shortened -- skipped. A
 *    session's first render is always "not shown yet" (see `lib/capability.ts`'s
 *    reasoning on why that read happens in an effect, not during render), so
 *    there is one tick where this component could in principle flash before
 *    the reduced-motion check lands; `ready` gates that -- nothing renders
 *    until the check has actually run.
 * 2. **`navigator.webdriver`.** Playwright (and every other WebDriver-based
 *    tool) sets this flag. None of this project's specs expect a two-second
 *    full-screen overlay to appear and intercept the first click on a fresh
 *    context, and adding a per-spec workaround to every one of them is worse
 *    than reading a flag the automation already sets for exactly this
 *    purpose. The screenshot suite is covered twice over: it also forces
 *    `reducedMotion: 'reduce'` (see `playwright.screenshots.config.ts`).
 * 3. **A second render in the same session.** `sessionStorage`, set the
 *    moment the sequence starts (not when it finishes) so an interrupted
 *    first run never replays on the next navigation.
 *
 * Mounted once from `app/layout.tsx`, not `app/template.tsx` -- a template
 * remounts on every route change, which would replay this on every click.
 */
export function IntroSequence() {
  const reduced = useReducedMotion();
  const [ready, setReady] = useState(false);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setReady(true);
    if (reduced) return;
    const automated =
      typeof navigator !== 'undefined' &&
      (navigator as Navigator & { webdriver?: boolean }).webdriver === true;
    if (automated) return;
    try {
      if (sessionStorage.getItem(SESSION_KEY) !== null) return;
      sessionStorage.setItem(SESSION_KEY, '1');
    } catch {
      // Private-mode storage can throw on write; show it once for this
      // render rather than blocking the page over a decoration.
    }
    setVisible(true);
  }, [reduced]);

  useEffect(() => {
    if (!visible) return;
    const timer = setTimeout(() => setVisible(false), TOTAL_MS);
    const skip = (event: KeyboardEvent) => {
      if (event.key === 'Tab') return; // let keyboard users tab to the skip control
      setVisible(false);
    };
    window.addEventListener('keydown', skip);
    return () => {
      clearTimeout(timer);
      window.removeEventListener('keydown', skip);
    };
  }, [visible]);

  if (!ready) return null;

  return (
    <LazyMotion features={loadFeatures} strict>
      <AnimatePresence>
        {visible ? (
          <m.div
            role="button"
            aria-label="Skip intro"
            tabIndex={0}
            onClick={() => setVisible(false)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') setVisible(false);
            }}
            className="fixed inset-0 z-[100] cursor-pointer overflow-hidden"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.4, ease: 'easeOut' } }}
          >
            {LAYERS.map((layer, index) => (
              <m.div
                key={layer}
                aria-hidden="true"
                className={'absolute inset-0 ' + layer}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: index * STEP, duration: STEP * 0.85, ease: 'easeOut' }}
              />
            ))}

            <m.div
              aria-hidden="true"
              className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-on-accent"
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                delay: LAYERS.length * STEP,
                type: 'spring',
                stiffness: 260,
                damping: 20,
              }}
            >
              <span className="text-4xl">♥</span>
              <span className="text-label-sm uppercase tracking-[0.2em]">
                PV-MEPCG / PulseVision
              </span>
            </m.div>

            {/* Timed with the heart mark rather than shown from frame one: a
                fixed off-scheme colour (white, or `mix-blend-mode` against
                one) would be readable against all four layers as they change
                underneath it, but the brand rule is the four scheme colours
                and nothing else, and `text-on-accent` only reads against the
                accent layer -- which is exactly the one on screen once this
                fades in. The whole overlay is click- and keydown-skippable
                from the first frame regardless; this is the visible label. */}
            <m.span
              className="absolute bottom-6 right-6 text-label-sm uppercase tracking-wide text-on-accent"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: LAYERS.length * STEP, duration: STEP }}
            >
              Skip
            </m.span>
          </m.div>
        ) : null}
      </AnimatePresence>
    </LazyMotion>
  );
}
