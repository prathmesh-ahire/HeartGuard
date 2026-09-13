'use client';

import { useState, type ReactNode } from 'react';

import { SideNav } from '@/components/SideNav';
import { TopBar } from '@/components/TopBar';

/**
 * The workstation frame: a fixed rail on the left, a sticky bar on top, the
 * page in the remaining canvas.
 *
 * It exists as one client component only because the rail and the bar share a
 * single piece of state -- whether the mobile drawer is open. Everything else
 * here is layout.
 *
 * The disclaimer and the footer are SLOTS, filled by the root layout, not
 * imported here. T110.3's guarantee is that the root layout itself renders
 * them, so no route -- including one added later -- can render without its
 * scope notice or the provenance of its numbers; `tests/test_frontend_scaffold.py`
 * reads `app/layout.tsx` for exactly that.
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
      <SideNav open={navOpen} onClose={() => setNavOpen(false)} />

      <div className="flex min-h-screen flex-col lg:pl-rail">
        {disclaimer}
        <TopBar onMenu={() => setNavOpen(true)} />
        <main className="flex-1 px-4 py-6 lg:px-6">
          <div className="mx-auto w-full max-w-[1600px]">{children}</div>
        </main>
        {footer}
      </div>
    </>
  );
}
