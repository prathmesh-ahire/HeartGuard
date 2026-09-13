'use client';

import { usePathname } from 'next/navigation';

import { ThemeToggle } from '@/components/ThemeToggle';
import { Icon } from '@/components/ui/Icon';
import { sectionFor } from '@/lib/routes';

/**
 * The top bar: which page you are on, and the theme control.
 *
 * The title is read from the route table rather than passed in per page, for
 * the same reason the rail is: two lists of pages drift, one does not.
 *
 * Note what is deliberately NOT here. The Stitch reference carries a live
 * sample-rate readout, a DSP latency figure and an operator identity. This
 * build has no live device and no operator account, and a number on a chrome
 * bar is still a number -- inventing one to fill the space would be exactly
 * the failure research rule 1 exists to prevent.
 */
export function TopBar({ onMenu }: { onMenu: () => void }) {
  const pathname = usePathname();
  const route = sectionFor(pathname);

  return (
    <header className="sticky top-0 z-20 flex h-topbar items-center justify-between gap-4 border-b border-line bg-panel/95 px-4 backdrop-blur lg:px-6">
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={onMenu}
          aria-controls="primary-rail"
          aria-label="Open navigation"
          className="-ml-1 rounded-lg border border-line p-1.5 text-ink-2 hover:border-accent-line hover:text-accent-strong lg:hidden"
        >
          <Icon name="menu" className="h-4 w-4" />
        </button>

        <p className="truncate text-headline-sm text-ink">{route.label}</p>
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <span className="hidden items-center gap-1.5 border-r border-line pr-3 font-mono text-label-sm uppercase text-ink-3 xl:flex">
          <Icon name="check" className="h-3.5 w-3.5 text-accent" />
          Research prototype
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}
