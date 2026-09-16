import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { Icon } from '@/components/ui/Icon';

/**
 * Secondary detail behind a native disclosure (T135.5).
 *
 * `<details>`/`<summary>` rather than a client-side accordion: it needs no
 * JavaScript, it is keyboard- and screen-reader-accessible for free, and a
 * browser auto-opens the closed `<details>` ancestor of a URL fragment's
 * target -- so a deep link into a collapsed section (the page's own "jump to"
 * nav, a link from another page) still lands open. A client accordion would
 * need its own fragment-handling code to get that; this gets it from the
 * platform.
 *
 * **`id` goes on a marker just inside the content, not on `<details>`
 * itself.** The auto-open behaviour fires for a fragment target that is
 * *hidden* by the closed ancestor; `<details>` itself is never hidden (only
 * its content region is), so a browser has no reason to open it for its own
 * id -- confirmed by running this in a real browser rather than assumed, T135.7's
 * Playwright spec pins it. The marker also carries `scroll-mt-24` so the open
 * panel lands below the sticky tab bar, not under it.
 *
 * `defaultOpen` marks the one or two panels a tab should show without a
 * click -- the "key charts" T135.3 asks for. Everything else here is real
 * content, not omitted, just not first on screen.
 */
export function Disclosure({
  id,
  summary,
  defaultOpen = false,
  className,
  children,
}: {
  id?: string;
  summary: ReactNode;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details open={defaultOpen} className={cn(SURFACE.card, 'group overflow-hidden', className)}>
      <summary
        className={cn(
          'flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3',
          'select-none [&::-webkit-details-marker]:hidden',
          'hover:bg-sunken',
        )}
      >
        <span className={cn(TYPE_SCALE.body, 'font-medium text-ink')}>{summary}</span>
        <Icon name="chevronDown" className="h-4 w-4 shrink-0 text-ink-3 transition-transform group-open:rotate-180" />
      </summary>
      <div className="border-t border-line p-4">
        {id ? <span id={id} className="block scroll-mt-24" aria-hidden="true" /> : null}
        {children}
      </div>
    </details>
  );
}
