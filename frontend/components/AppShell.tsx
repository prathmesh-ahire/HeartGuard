'use client';

import { useState, type ReactNode } from 'react';

import { SideNav } from '@/components/SideNav';
import { TopBar } from '@/components/TopBar';
import { cn } from '@/lib/cn';
import { LAYOUT } from '@/lib/tokens';

/**
 * The page shell (T128.4): the navigation rail on a wide screen, a menu that
 * opens the same five pages in a drawer on a narrow one, a quiet top bar, and
 * the page in one centred content width.
 *
 * It exists as one client component only because the rail, the drawer and the
 * bar share a single piece of state -- whether the mobile menu is open.
 *
 * The disclaimer and the footer are SLOTS, filled by the root layout, not
 * imported here. T110.3's guarantee is that the root layout itself renders
 * them, so no route -- including one added later -- can render without its
 * scope notice; `tests/test_frontend_scaffold.py` reads `app/layout.tsx` for
 * exactly that.
 */
export function AppShell({
  children,
  disclaimer,
  footer,
}: {
  children: ReactNode;
  disclaimer: ReactNode;
  footer: ReactNode;
}) {
  const [navOpen, setNavOpen] = useState(false);

  return (
    <>
      <a
        href="#main"
        className="sr-only z-50 rounded-lg bg-accent px-3 py-2 text-on-accent focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
      >
        Skip to content
      </a>

      <SideNav open={navOpen} onClose={() => setNavOpen(false)} />

      <div className="flex min-h-screen flex-col lg:pl-rail">
        {disclaimer}
        <TopBar onMenu={() => setNavOpen(true)} menuOpen={navOpen} />
        <main id="main" className={cn(LAYOUT.page, LAYOUT.pageBlock, 'flex-1')}>
          {children}
        </main>
        {footer}
      </div>
    </>
  );
}
