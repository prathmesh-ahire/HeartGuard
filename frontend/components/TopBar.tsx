'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { ThemeToggle } from '@/components/ThemeToggle';
import { Icon, type IconName } from '@/components/ui/Icon';
import { useReducedMotion, useScrolled } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { TOPBAR_NAV_TRANSITION } from '@/lib/motion';
import { ROUTES, sectionFor, type RouteDefinition } from '@/lib/routes';
import { LAYOUT, SURFACE } from '@/lib/tokens';

/**
 * The top bar: the brand mark, the five-page navigation on a wide screen (the
 * redesign moved it here from the side rail so the page runs full width), the
 * menu on a narrow screen, and the theme control (T128.4).
 *
 * **The scroll crossfade** (redesign, 2026-09-17): at the top of the page this
 * is the full bar below. Past `SCROLL_THRESHOLD` it fades and lifts away while
 * a second, smaller nav -- page links only, no logo, no badge, no toggle --
 * fades and settles into a centred floating pill above it. Both stay mounted
 * at all times (only opacity/transform/pointer-events change) and share one
 * named transition (`TOPBAR_NAV_TRANSITION`), so the swap reads as one
 * continuous morph rather than two independent fades landing at different
 * times. Desktop only (`lg:`) -- on a narrow screen there is no floating
 * replacement, so the ordinary bar (hamburger, page name, toggle) never fades.
 *
 * The nav is read from `lib/routes.ts` rather than a second hand-written list,
 * for the same reason the old rail was: two lists of pages drift, one does not.
 *
 * Note what is deliberately NOT here. The Stitch reference carries a live
 * sample-rate readout, a DSP latency figure and an operator identity. This
 * build has no live device and no operator account, and a number on a chrome
 * bar is still a number -- inventing one to fill the space would be exactly
 * the failure research rule 1 exists to prevent.
 */
const ICONS: Record<string, IconName> = {
  '/': 'pulse',
  '/history/': 'history',
  '/insights/': 'insights',
  '/reports/': 'reports',
  '/about/': 'about',
};

const SCROLL_THRESHOLD = 24;

function NavLinks({ current, iconSize }: { current: RouteDefinition; iconSize: string }) {
  return (
    <>
      {ROUTES.map((entry) => {
        const active = entry.href === current.href;
        return (
          <Link
            key={entry.href}
            href={entry.href}
            title={entry.summary}
            aria-current={active ? 'page' : undefined}
            className={cn(
              'flex items-center gap-1.5 rounded-full px-3 py-1.5 text-label-lg font-medium transition-colors',
              active
                ? 'bg-accent-soft text-accent-deep'
                : 'text-ink-2 hover:bg-accent-soft/60 hover:text-accent-strong',
            )}
          >
            <Icon
              name={ICONS[entry.href] ?? 'pulse'}
              className={cn(iconSize, 'shrink-0', active ? 'text-accent' : 'text-ink-3')}
            />
            {entry.label}
          </Link>
        );
      })}
    </>
  );
}

export function TopBar({ onMenu, menuOpen = false }: { onMenu: () => void; menuOpen?: boolean }) {
  const pathname = usePathname();
  const route = sectionFor(pathname);
  const scrolled = useScrolled(SCROLL_THRESHOLD);
  const reduced = useReducedMotion();

  return (
    <>
      <header
        className={cn(
          'sticky top-0 z-20 origin-top border-b border-line/70 bg-surface/85 backdrop-blur',
          scrolled && 'lg:pointer-events-none lg:-translate-y-3 lg:scale-[0.97] lg:opacity-0',
        )}
        style={{ transition: reduced ? 'none' : TOPBAR_NAV_TRANSITION }}
      >
        <div className={cn(LAYOUT.page, 'flex h-topbar items-center justify-between gap-4')}>
          <div className="flex min-w-0 items-center gap-4">
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

            <Link href="/" className="flex shrink-0 items-center gap-2">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent text-on-accent">
                <Icon name="pulse" className="h-4 w-4" strokeWidth={1.8} />
              </span>
              <span className="hidden truncate text-headline-sm font-bold tracking-tight text-accent-strong lg:inline">
                PulseVision
              </span>
            </Link>

            <p className="truncate text-body-lg font-medium text-ink-2 lg:hidden">{route.label}</p>

            <nav aria-label="Primary" aria-hidden={scrolled} className="hidden items-center gap-1 lg:flex">
              <NavLinks current={route} iconSize="h-4 w-4" />
            </nav>
          </div>

          <div className="flex shrink-0 items-center gap-3">
            <span className="hidden items-center gap-1.5 text-label-sm uppercase text-ink-3 xl:flex">
              <Icon name="check" className="h-3.5 w-3.5 text-accent" />
              Research prototype
            </span>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <nav
        aria-label="Primary"
        aria-hidden={!scrolled}
        className={cn(
          SURFACE.glass,
          'fixed left-1/2 top-4 z-30 hidden origin-top -translate-x-1/2 items-center gap-1 rounded-full px-2 py-1.5 shadow-raised lg:flex',
          scrolled
            ? 'translate-y-0 scale-100 opacity-100'
            : 'pointer-events-none -translate-y-4 scale-90 opacity-0',
        )}
        style={{ transition: reduced ? 'none' : TOPBAR_NAV_TRANSITION }}
      >
        <NavLinks current={route} iconSize="h-4 w-4" />
      </nav>
    </>
  );
}
