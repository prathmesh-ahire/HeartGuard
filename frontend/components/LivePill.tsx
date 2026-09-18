'use client';

import { useEffect, useState } from 'react';

import { cn } from '@/lib/cn';

type Status = 'checking' | 'live' | 'offline';

/**
 * The nav "Live" pill (T140.6): one `GET /health` on mount, nothing else.
 *
 * Goes through `lib/api.ts`'s own `health()` rather than a raw `fetch` --
 * T119.2's rule, that every runtime network call crosses through that one
 * module, has no carve-out for "it's just a status ping". A ping is not a
 * precomputed metric and not data that already exists on disk either; it is
 * live operational status, the same category `POST /predict` is allowed to
 * cross the wire for.
 *
 * The import is dynamic, not a top-level `import { health } from
 * '@/lib/api'`: `TopBar` renders on all five pages via `AppShell`, and a
 * static import would pull api.ts's whole export surface (history, reports,
 * batch predict, ...) into the shared chunk every route pays for -- which is
 * exactly what happened on the first attempt: the shared bundle went from
 * 127.1 kB to 129.7 kB against a 130 kB budget and pushed one route over its
 * own. A dynamic `import()` code-splits api.ts into its own lazy chunk,
 * fetched only after this component mounts, the same way `gsap`/`echarts`/
 * `wavesurfer.js` stay out of "First Load JS" elsewhere in this codebase.
 */
export function LivePill() {
  const [status, setStatus] = useState<Status>('checking');

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const { health } = await import('@/lib/api');
        await health();
        if (!cancelled) setStatus('live');
      } catch {
        if (!cancelled) setStatus('offline');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (status === 'checking') return null;

  return (
    <span
      className={cn(
        'hidden items-center gap-1.5 rounded-full border px-2 py-0.5 text-label-sm uppercase xl:flex',
        status === 'live'
          ? 'border-good-line bg-good-soft text-good'
          : 'border-line bg-sunken text-ink-3',
      )}
      title={
        status === 'live'
          ? 'The inference API answered /health.'
          : 'The inference API did not answer. Live scoring is unavailable; the rest of the dashboard is precomputed.'
      }
    >
      <span
        aria-hidden="true"
        className={cn(
          'h-1.5 w-1.5 shrink-0 rounded-full bg-current',
          status === 'live' && 'motion-safe:animate-pulse-subtle',
        )}
      />
      {status === 'live' ? 'Live' : 'Offline'}
    </span>
  );
}
