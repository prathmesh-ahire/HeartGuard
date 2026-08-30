'use client';

import { ThemeProvider as NextThemeProvider } from 'next-themes';
import type { ReactNode } from 'react';

/**
 * next-themes needs a client boundary, and the root layout is a server
 * component. This is that boundary and nothing else.
 *
 * `disableTransitionOnChange` avoids every colour on the page animating when
 * the theme flips, which on a page full of charts reads as a rendering bug.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  return (
    <NextThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      {children}
    </NextThemeProvider>
  );
}
