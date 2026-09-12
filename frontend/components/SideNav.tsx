'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';

import { Icon, type IconName } from '@/components/ui/Icon';
import { manifest } from '@/lib/generated';
import { GROUP_LABELS, ROUTES, UTILITY_ROUTES, type RouteDefinition } from '@/lib/routes';

/**
 * The left telemetry rail.
 *
 * Built from `lib/routes.ts` rather than from a second hand-written list, so a
 * page and its nav entry cannot drift apart. The icon is the only thing chosen
 * here -- it is presentation, and putting it in the route table would make the
 * route table a design file.
 */
const ICONS: Record<string, IconName> = {
  '/': 'home',
  '/dataset/': 'dataset',
  '/preprocessing/': 'waveform',
  '/features/': 'features',
  '/models/': 'models',
  '/optimization/': 'optimization',
  '/robustness/': 'robustness',
  '/explainability/': 'explainability',
  '/reports/': 'reports',
  '/predict/binary/': 'binary',
  '/predict/multiclass/': 'multiclass',
  '/predict/murmur/': 'murmur',
  '/design/': 'design',
  '/limitations/': 'limitations',
};

const GROUP_ORDER: RouteDefinition['group'][] = ['overview', 'method', 'results', 'predict'];

function isActive(pathname: string, href: string): boolean {
  return pathname === href || pathname === href.slice(0, -1) || `${pathname}/` === href;
}

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
        'flex items-center gap-2.5 rounded-lg border-l-2 px-2.5 py-1.5 font-mono text-label-md transition-colors',
        active
          ? 'border-accent bg-accent-soft text-accent-deep'
          : 'border-transparent text-ink-2 hover:bg-accent-soft/50 hover:text-accent-strong',
      ].join(' ')}
    >
      <Icon
        name={ICONS[route.href] ?? 'features'}
        className={active ? 'h-4 w-4 shrink-0 text-accent' : 'h-4 w-4 shrink-0 text-ink-3'}
      />
      <span className="truncate">{route.label}</span>
    </Link>
  );
}

export function SideNav({ open, onClose }: { open: boolean; onClose: () => void }) {
  const pathname = usePathname();

  /* A route change must close the mobile drawer, or the next page renders
   * underneath an overlay the reader has to dismiss by hand. */
  useEffect(() => {
    onClose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  const close = onClose;

  return (
    <>
      {open ? (
        <div
          className="fixed inset-0 z-30 bg-ink/30 backdrop-blur-[2px] lg:hidden"
          onClick={close}
          aria-hidden="true"
        />
      ) : null}

      <aside
        id="primary-rail"
        className={[
          'fixed inset-y-0 left-0 z-40 flex w-rail flex-col justify-between overflow-y-auto',
          'border-r border-line bg-panel p-3 transition-transform duration-200 lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        ].join(' ')}
      >
        <div className="flex flex-col gap-4">
          <Link href="/" onClick={close} className="flex items-center gap-2.5 px-1 pt-1">
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

          <nav aria-label="Primary" className="flex flex-col gap-4">
            {GROUP_ORDER.map((group) => {
              const routes = ROUTES.filter((route) => route.group === group);
              if (routes.length === 0) return null;
              return (
                <div key={group} className="flex flex-col gap-1">
                  <p className="label-micro px-2.5 pb-0.5">{GROUP_LABELS[group]}</p>
                  {routes.map((route) => (
                    <NavLink
                      key={route.href}
                      route={route}
                      active={isActive(pathname, route.href)}
                      onNavigate={close}
                    />
                  ))}
                </div>
              );
            })}
          </nav>
        </div>

        <div className="mt-6 flex flex-col gap-1 border-t border-line pt-3">
          {UTILITY_ROUTES.map((route) => (
            <NavLink
              key={route.href}
              route={route}
              active={isActive(pathname, route.href)}
              onNavigate={close}
            />
          ))}

          {/*
           * The provenance chip. The run id is the export that produced every
           * number above it -- shown here rather than only in the footer,
           * because the rail is visible on every scroll position and the
           * footer is not.
           */}
          <div className="mt-2 rounded-lg border border-line bg-sunken px-2.5 py-2">
            <p className="label-micro flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                Build
              </span>
              <span className="truncate font-mono text-ink-2">
                {manifest.git_commit ? manifest.git_commit.slice(0, 7) : 'unknown'}
              </span>
            </p>
            <p className="label-micro mt-1 truncate" title={manifest.run_id}>
              {manifest.run_id || 'unrecorded run'}
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}
