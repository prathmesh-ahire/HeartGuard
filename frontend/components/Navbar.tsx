'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { GROUP_LABELS, ROUTES, type RouteDefinition } from '@/lib/routes';

/**
 * Navigation built from `lib/routes.ts` rather than from a second hand-written
 * list, so a page and its nav entry cannot drift apart.
 */
function groupedRoutes(): [RouteDefinition['group'], RouteDefinition[]][] {
  const order: RouteDefinition['group'][] = ['overview', 'method', 'results', 'predict'];
  return order.map((group) => [group, ROUTES.filter((route) => route.group === group)]);
}

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
      <nav
        aria-label="Primary"
        className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3"
      >
        <Link href="/" className="mr-2 shrink-0 font-semibold tracking-tight">
          PV-MEPCG <span className="text-slate-400">/</span> PulseVision
        </Link>

        {groupedRoutes().map(([group, routes]) => (
          <div key={group} className="flex items-center gap-3">
            <span className="text-[10px] uppercase tracking-widest text-slate-400 dark:text-slate-500">
              {GROUP_LABELS[group]}
            </span>
            {routes
              .filter((route) => route.href !== '/')
              .map((route) => {
                const active = pathname === route.href || pathname === route.href.slice(0, -1);
                return (
                  <Link
                    key={route.href}
                    href={route.href}
                    title={route.summary}
                    aria-current={active ? 'page' : undefined}
                    className={
                      active
                        ? 'text-sm font-medium text-sky-700 underline underline-offset-4 dark:text-sky-400'
                        : 'text-sm text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100'
                    }
                  >
                    {route.label}
                  </Link>
                );
              })}
          </div>
        ))}
      </nav>
    </header>
  );
}
