import type { Metadata } from 'next';
import { Hanken_Grotesk, JetBrains_Mono } from 'next/font/google';

import './globals.css';

import { AppShell } from '@/components/AppShell';
import { DisclaimerBanner } from '@/components/Disclaimer';
import { Footer } from '@/components/Footer';
import { SmoothScroll } from '@/components/motion/SmoothScroll';
import { ThemeProvider } from '@/components/ThemeProvider';
import { SHOW_SCREENING_NOTICE } from '@/lib/flags';

/**
 * The root layout (T110.3, T127.2).
 *
 * The screening notice and the footer are rendered HERE, handed to `AppShell`
 * as slots, and the navigation lives inside that shell -- none of the three is
 * placed on a page. Per-page placement is how a notice goes missing: a route
 * added later simply does not get one and nothing fails.
 *
 * The notice is switched off for the presentation by `SHOW_SCREENING_NOTICE`
 * (Open Item 17). It stays wired here, so T138.6 restores it on every route by
 * flipping that one flag.
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
  description: 'Heart-sound analysis: upload a phonocardiogram recording and see the result.',
  applicationName: 'PV-MEPCG / PulseVision',
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${sans.variable} ${mono.variable}`}>
      <body>
        <ThemeProvider>
          <SmoothScroll>
            <AppShell
              disclaimer={SHOW_SCREENING_NOTICE ? <DisclaimerBanner /> : null}
              footer={<Footer />}
            >
              {children}
            </AppShell>
          </SmoothScroll>
        </ThemeProvider>
      </body>
    </html>
  );
}
