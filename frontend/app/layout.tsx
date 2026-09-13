import type { Metadata } from 'next';
import { Hanken_Grotesk, JetBrains_Mono } from 'next/font/google';

import './globals.css';

import { AppShell } from '@/components/AppShell';
import { DisclaimerBanner } from '@/components/Disclaimer';
import { Footer } from '@/components/Footer';
import { SmoothScroll } from '@/components/motion/SmoothScroll';
import { ThemeProvider } from '@/components/ThemeProvider';

/**
 * The root layout (T110.3).
 *
 * The disclaimer and the run-manifest footer are rendered HERE, handed to
 * `AppShell` as slots, and the navigation lives inside that shell -- none of
 * the three is placed on a page. Per-page placement is how
 * a disclaimer goes missing: a route added later simply does not get one and
 * nothing fails. From here it is structurally impossible for a page to render
 * without its scope notice or without the provenance of the numbers it shows.
 *
 * Both faces are self-hosted through next/font. A <link> to fonts.googleapis
 * would leave the static export dependent on a network it will not always
 * have, and the fallback metrics next/font emits also stop the layout shifting
 * when the face lands.
 */
const sans = Hanken_Grotesk({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-sans',
});

const mono = JetBrains_Mono({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-mono',
});

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
    <html lang="en" suppressHydrationWarning className={`${sans.variable} ${mono.variable}`}>
      <body>
        <ThemeProvider>
          <SmoothScroll>
            <AppShell disclaimer={<DisclaimerBanner />} footer={<Footer />}>
              {children}
            </AppShell>
          </SmoothScroll>
        </ThemeProvider>
      </body>
    </html>
  );
}
