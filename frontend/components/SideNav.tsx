'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';

import { Icon, type IconName } from '@/components/ui/Icon';
import { ROUTES, sectionFor, type RouteDefinition } from '@/lib/routes';

/**
 * The left navigation rail: the five product pages and nothing else (T127.3).
 *
 * Built from `lib/routes.ts` rather than from a second hand-written list, so a
 * page and its nav entry cannot drift apart. The icon is the only thing chosen
 * here -- it is presentation, and putting it in the route table would make the
 * route table a design file.
 */
const ICONS: Record<string, IconName> = {
  '/': 'pulse',
  '/history/': 'history',
  '/insights/': 'insights',
  '/reports/': 'reports',
  '/about/': 'about',
};

function NavLink({
  route,
  active,
  onNavigate,
}: {
  route: RouteDefinition;
  active: boolean;
  onNavigate: () => void;
}) {
  return (
    <Link
      href={route.href}
      title={route.summary}
      onClick={onNavigate}
      aria-current={active ? 'page' : undefined}
      className={[
        'flex items-center gap-3 rounded-lg border-l-2 px-3 py-2 text-body-lg font-medium transition-colors',
        active
          ? 'border-accent bg-accent-soft text-accent-deep'
          : 'border-transparent text-ink-2 hover:bg-accent-soft/50 hover:text-accent-strong',
      ].join(' ')}
    >
      <Icon
        name={ICONS[route.href] ?? 'pulse'}
        className={active ? 'h-5 w-5 shrink-0 text-accent' : 'h-5 w-5 shrink-0 text-ink-3'}
      />
      <span className="truncate">{route.label}</span>
    </Link>
  );
}

export function SideNav({ open, onClose }: { open: boolean; onClose: () => void }) {
  const pathname = usePathname();
  const current = sectionFor(pathname);

  /* A route change must close the mobile drawer, or the next page renders
   * underneath an overlay the reader has to dismiss by hand. */
  useEffect(() => {
    onClose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  return (
    <>
      {open ? (
        <div
          className="fixed inset-0 z-30 bg-ink/30 backdrop-blur-[2px] lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      ) : null}

      <aside
        id="primary-rail"
        className={[
          'fixed inset-y-0 left-0 z-40 flex w-rail flex-col overflow-y-auto',
          'border-r border-line bg-panel p-3 transition-transform duration-200 lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        ].join(' ')}
      >
        <Link href="/" onClick={onClose} className="flex items-center gap-2.5 px-1 pt-1">
          <span className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-accent-line bg-accent-soft text-accent">
            <Icon name="pulse" className="h-5 w-5" strokeWidth={1.8} />
          </span>
          <span className="flex min-w-0 flex-col leading-none">
            <span className="truncate text-headline-sm font-bold tracking-tight text-accent-strong">
              PulseVision
            </span>
            <span className="label-micro mt-1 truncate">PV-MEPCG</span>
          </span>
        </Link>

        <nav aria-label="Primary" className="mt-6 flex flex-col gap-1">
          {ROUTES.map((route) => (
            <NavLink
              key={route.href}
              route={route}
              active={current.href === route.href}
              onNavigate={onClose}
            />
          ))}
        </nav>
      </aside>
    </>
  );
}
