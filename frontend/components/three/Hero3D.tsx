'use client';

import dynamic from 'next/dynamic';

import { useCapability } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * The SSR-safe, lazy-loaded 3D boundary (T112.1).
 *
 * Everything three.js touches sits behind this one dynamic import. Three things
 * follow from that, and all three are the reason the file exists:
 *
 * 1. **`ssr: false`.** `next.config.mjs` sets `output: 'export'`, so every page
 *    is pre-rendered at build time in Node, where there is no `window`, no
 *    `document` and no WebGL context. A three.js component rendered there does
 *    not degrade -- it throws, and the build fails.
 * 2. **The bundle.** three.js plus fiber plus drei is several hundred kilobytes.
 *    Behind `dynamic()` it becomes a separate chunk that no route's first load
 *    pays for; imported normally it would be in the shared chunk of every page
 *    including the ones with no 3D on them at all.
 * 3. **The fallback is not an error.** No WebGL, or reduced motion, is an
 *    ordinary outcome, so the fallback is a designed static panel rather than
 *    an `ErrorState`. Nothing failed.
 */

const HeartScene = dynamic(() => import('./HeartScene'), {
  ssr: false,
  loading: () => <HeroFallback reason="loading" />,
});

export interface Hero3DProps {
  className?: string;
  interactive?: boolean;
  /** Height of the stage. Fixed so the fallback and the scene do not reflow. */
  height?: string;
}

export function Hero3D({ className, interactive = false, height = '20rem' }: Hero3DProps) {
  const capability = useCapability();

  if (!capability.ready) {
    return (
      <div className={className} style={{ height }}>
        <HeroFallback reason="loading" />
      </div>
    );
  }

  if (!capability.webgl) {
    return (
      <div className={className} style={{ height }}>
        <HeroFallback reason="no-webgl" />
      </div>
    );
  }

  // Reduced motion still gets the model -- just posed rather than beating.
  // Replacing it with a flat panel would remove information the sighted
  // visitor was not asking to lose; they asked for the movement to stop.
  return (
    <div className={cn('overflow-hidden', className)} style={{ height }}>
      <HeartScene animate={capability.allow3d} interactive={interactive} />
    </div>
  );
}

const REASONS = {
  loading: {
    title: 'Preparing the model',
    body: 'The 3D layer loads separately from the page, so the rest of the dashboard is usable before it arrives.',
  },
  'no-webgl': {
    title: 'WebGL is unavailable',
    body: 'This browser or graphics driver does not provide a WebGL context, so the heart model cannot render. Nothing else on the dashboard depends on it — every result and figure below is unaffected.',
  },
} as const;

export function HeroFallback({ reason }: { reason: keyof typeof REASONS }) {
  const copy = REASONS[reason];
  return (
    <div
      className={cn(
        SURFACE.sunken,
        'flex h-full w-full flex-col items-center justify-center gap-2 px-6 text-center',
      )}
    >
      <span aria-hidden="true" className="text-3xl">
        ♥
      </span>
      <p className={cn(TYPE_SCALE.h3)}>{copy.title}</p>
      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'max-w-sm')}>{copy.body}</p>
    </div>
  );
}
