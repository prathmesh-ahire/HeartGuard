'use client';

import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

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

  const label = !mounted
    ? 'Theme'
    : resolvedTheme === 'dark'
      ? 'Switch to light theme'
      : 'Switch to dark theme';

  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
      className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
    >
      {mounted ? (resolvedTheme === 'dark' ? 'Light' : 'Dark') : 'Theme'}
    </button>
  );
}
