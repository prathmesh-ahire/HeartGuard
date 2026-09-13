'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { ABOUT_TABS, matchesRoute } from '@/lib/routes';

/**
 * The About the Model tab bar.
 *
 * Tabs are sub-routes rather than client-side panels. A Radix tab renders only
 * the active panel, so a single-page tab set would leave five tabs' tables out
 * of the static HTML -- invisible to the displayed-value audit -- and put every
 * tab's charts into one route's bundle.
 */
export function AboutTabs() {
  const pathname = usePathname();

  return (
    <nav aria-label="About the Model" className="-mb-px flex gap-1 overflow-x-auto border-b border-line">
      {ABOUT_TABS.map((tab) => {
        const active = matchesRoute(pathname, tab.href);
        return (
          <Link
            key={tab.href}
            href={tab.href}
            title={tab.summary}
            aria-current={active ? 'page' : undefined}
            className={[
              'whitespace-nowrap border-b-2 px-3 py-2 text-body-md font-medium transition-colors',
              active
                ? 'border-accent text-accent-strong'
                : 'border-transparent text-ink-2 hover:border-accent-line hover:text-accent-strong',
            ].join(' ')}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
