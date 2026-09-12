'use client';

import { useState, type ReactNode } from 'react';

import { DisclaimerBanner } from '@/components/Disclaimer';
import { Footer } from '@/components/Footer';
import { SideNav } from '@/components/SideNav';
import { TopBar } from '@/components/TopBar';

/**
 * The workstation frame: a fixed rail on the left, a sticky bar on top, the
 * page in the remaining canvas.
 *
 * It exists as one client component only because the rail and the bar share a
 * single piece of state -- whether the mobile drawer is open. Everything else
 * here is layout, and the root layout stays a server component.
 *
 * The disclaimer sits ABOVE the bar, in the frame, not on the page. Per-page
 * placement is how a disclaimer goes missing: a route added later simply does
 * not get one and nothing fails.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const [navOpen, setNavOpen] = useState(false);

  return (
    <>
      <SideNav open={navOpen} onClose={() => setNavOpen(false)} />

      <div className="flex min-h-screen flex-col lg:pl-rail">
        <DisclaimerBanner />
        <TopBar onMenu={() => setNavOpen(true)} />
        <main className="flex-1 px-4 py-6 lg:px-6">
          <div className="mx-auto w-full max-w-[1600px]">{children}</div>
        </main>
        <Footer />
      </div>
    </>
  );
}
