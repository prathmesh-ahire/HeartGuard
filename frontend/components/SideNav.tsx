'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';

import { Icon, type IconName } from '@/components/ui/Icon';
import { Drawer } from '@/components/ui/Overlay';
import { cn } from '@/lib/cn';
import { ROUTES, sectionFor, type RouteDefinition } from '@/lib/routes';

/**
 * The navigation (T127.3, T128.4; moved to the top bar in the redesign).
 *
 * On a wide screen the five pages are horizontal links in `TopBar`, so the
 * page runs full width instead of losing a rail's worth of it. This component
 * now only supplies the narrow-screen equivalent: the same list opens in a
 * left `Drawer` from the top bar's menu button, which brings the drawer's
 * focus handling and Escape with it.
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
  onNavigate?: () => void;
}) {
  return (
    <Link
      href={route.href}
      title={route.summary}
      onClick={onNavigate}
      aria-current={active ? 'page' : undefined}
      className={cn(
        'flex items-center gap-3 rounded-xl px-3 py-2.5 text-body-lg font-medium transition-colors',
        active
          ? 'bg-accent-soft text-accent-deep'
          : 'text-ink-2 hover:bg-accent-soft/60 hover:text-accent-strong',
      )}
    >
      <Icon
        name={ICONS[route.href] ?? 'pulse'}
        className={cn('h-5 w-5 shrink-0', active ? 'text-accent' : 'text-ink-3')}
      />
      <span className="truncate">{route.label}</span>
    </Link>
  );
}

function PrimaryNav({ current, onNavigate }: { current: string; onNavigate?: () => void }) {
  return (
    <nav aria-label="Primary" className="flex flex-col gap-1">
      {ROUTES.map((route) => (
        <NavLink
          key={route.href}
          route={route}
          active={current === route.href}
          onNavigate={onNavigate}
        />
      ))}
    </nav>
  );
}

export function SideNav({ open, onClose }: { open: boolean; onClose: () => void }) {
  const pathname = usePathname();
  const current = sectionFor(pathname).href;

  /* A route change must close the mobile menu, or the next page renders
   * underneath a drawer the reader has to dismiss by hand. */
  useEffect(() => {
    onClose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  return (
    <Drawer side="left" open={open} onClose={onClose} title="PulseVision">
      <PrimaryNav current={current} onNavigate={onClose} />
    </Drawer>
  );
}
