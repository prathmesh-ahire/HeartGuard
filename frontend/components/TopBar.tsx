'use client';

import { usePathname } from 'next/navigation';

import { ThemeToggle } from '@/components/ThemeToggle';
import { Icon } from '@/components/ui/Icon';
import { cn } from '@/lib/cn';
import { sectionFor } from '@/lib/routes';
import { LAYOUT } from '@/lib/tokens';

/**
 * The top bar: which page you are on, the menu on a narrow screen, and the
 * theme control (T128.4). Deliberately quiet -- the page's own title area is
 * where a page says what it is for.
 *
 * The page name is read from the route table rather than passed in per page,
 * for the same reason the rail is: two lists of pages drift, one does not.
 *
 * Note what is deliberately NOT here. The Stitch reference carries a live
 * sample-rate readout, a DSP latency figure and an operator identity. This
 * build has no live device and no operator account, and a number on a chrome
 * bar is still a number -- inventing one to fill the space would be exactly
 * the failure research rule 1 exists to prevent.
 */
export function TopBar({ onMenu, menuOpen = false }: { onMenu: () => void; menuOpen?: boolean }) {
  const pathname = usePathname();
  const route = sectionFor(pathname);

  return (
    <header className="sticky top-0 z-20 border-b border-line/70 bg-surface/85 backdrop-blur">
      <div className={cn(LAYOUT.page, 'flex h-topbar items-center justify-between gap-4')}>
        <div className="flex min-w-0 items-center gap-3">
          <button
            type="button"
            onClick={onMenu}
            aria-haspopup="dialog"
            aria-expanded={menuOpen}
            aria-label="Open navigation"
            className="-ml-1 rounded-lg border border-line bg-panel p-2 text-ink-2 hover:border-accent-line hover:text-accent-strong lg:hidden"
          >
            <Icon name="menu" className="h-4 w-4" />
          </button>

          <p className="truncate text-body-lg font-medium text-ink-2">{route.label}</p>
        </div>

        <div className="flex shrink-0 items-center gap-3">
          <span className="hidden items-center gap-1.5 font-mono text-label-sm uppercase text-ink-3 xl:flex">
            <Icon name="check" className="h-3.5 w-3.5 text-accent" />
            Research prototype
          </span>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
