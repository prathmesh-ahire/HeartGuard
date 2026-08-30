import type { Metadata } from 'next';

import './globals.css';

import { DisclaimerBanner } from '@/components/Disclaimer';
import { Footer } from '@/components/Footer';
import { SmoothScroll } from '@/components/motion/SmoothScroll';
import { Navbar } from '@/components/Navbar';
import { ThemeProvider } from '@/components/ThemeProvider';
import { ThemeToggle } from '@/components/ThemeToggle';

/**
 * The root layout (T110.3).
 *
 * The disclaimer, the navbar and the run-manifest footer live HERE rather than
 * on each page. Per-page placement is how a disclaimer goes missing: a route
 * added later simply does not get one and nothing fails. From here it is
 * structurally impossible for a page to render without its scope notice or
 * without the provenance of the numbers it is showing.
 */
export const metadata: Metadata = {
  title: {
    default: 'PV-MEPCG / PulseVision',
    template: '%s · PV-MEPCG / PulseVision',
  },
  description:
    'Search-optimized heterogeneous ensemble for phonocardiogram heart-sound ' +
    'classification. Academic screening and decision-support prototype; not a ' +
    'diagnostic tool.',
  applicationName: 'PV-MEPCG / PulseVision',
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="flex min-h-screen flex-col">
        <ThemeProvider>
          <SmoothScroll>
            <DisclaimerBanner />
            <Navbar />
            <div className="mx-auto flex w-full max-w-7xl justify-end px-4 pt-3">
              <ThemeToggle />
            </div>
            <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8">{children}</main>
            <Footer />
          </SmoothScroll>
        </ThemeProvider>
      </body>
    </html>
  );
}
