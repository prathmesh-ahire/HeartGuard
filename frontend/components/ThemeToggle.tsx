'use client';

import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

import { Icon } from '@/components/ui/Icon';

/**
 * The theme is only known in the browser, so the button renders a stable
 * placeholder until it mounts. Without that guard the server-rendered markup
 * and the first client render disagree and React logs a hydration error --
 * which on a static export is the whole page re-rendering.
 */
export function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const { resolvedTheme, setTheme } = useTheme();

  useEffect(() => setMounted(true), []);

  const dark = mounted && resolvedTheme === 'dark';
  const label = !mounted
    ? 'Theme'
    : dark
      ? 'Switch to light theme'
      : 'Switch to dark theme';

  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={() => setTheme(dark ? 'light' : 'dark')}
      className="flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2 py-1.5 font-mono text-label-md text-ink-2 transition-colors hover:border-accent-line hover:text-accent-strong"
    >
      <Icon name={dark ? 'sun' : 'moon'} className="h-3.5 w-3.5 text-accent" />
      <span className="hidden sm:inline">{mounted ? (dark ? 'LIGHT' : 'DARK') : 'THEME'}</span>
    </button>
  );
}
