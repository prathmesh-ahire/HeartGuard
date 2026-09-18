'use client';

import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

import styles from '@/components/ThemeToggle.module.css';

/**
 * A sun/moon switch (T-adhoc, 2026-09-17), adapted from a Uiverse.io toggle by
 * JkHuger onto this project's own CSS custom properties -- see
 * `ThemeToggle.module.css` for what changed and why.
 *
 * The theme is only known in the browser, so the checkbox renders unchecked
 * (its default) until mount. Without that guard the server-rendered markup
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
    <label className={styles.theme} title={label}>
      <span className={styles.toggleWrap}>
        <input
          className={styles.toggle}
          type="checkbox"
          role="switch"
          checked={dark}
          onChange={() => setTheme(dark ? 'light' : 'dark')}
          aria-label={label}
        />
        <span className={styles.icon} aria-hidden="true">
          <span className={styles.iconPart} />
          <span className={styles.iconPart} />
          <span className={styles.iconPart} />
          <span className={styles.iconPart} />
          <span className={styles.iconPart} />
          <span className={styles.iconPart} />
          <span className={styles.iconPart} />
          <span className={styles.iconPart} />
          <span className={styles.iconPart} />
        </span>
      </span>
    </label>
  );
}
