import { defineConfig, devices } from '@playwright/test';

/**
 * The screenshot run (Phase 120), behind the displayed-value audit (T119.4).
 *
 * `globalSetup` runs `scripts/45_audit_displayed_values.py --gate` before any
 * test starts. It refuses unless the site in `frontend/out/` is byte-for-byte
 * the build that passed the audit -- so no screenshot can be taken of an
 * unaudited page, or of a page rebuilt after its audit. Capture specs live in
 * `screenshots/` and are added by Phase 120.
 */
export default defineConfig({
  testDir: './screenshots',
  globalSetup: './e2e/screenshot-gate.ts',
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: { ...devices['Desktop Chrome'], channel: process.env.PW_CHANNEL },
});
