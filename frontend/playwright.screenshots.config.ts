import { resolve } from 'node:path';

import { defineConfig, devices } from '@playwright/test';

/**
 * The screenshot run (Phase 120), behind the displayed-value audit (T119.4).
 *
 * `globalSetup` runs `scripts/45_audit_displayed_values.py --gate` before any
 * test starts. It refuses unless the site in `frontend/out/` is byte-for-byte
 * the build that passed the audit -- so no screenshot can be taken of an
 * unaudited page, or of a page rebuilt after its audit.
 *
 * One server, not two: the real FastAPI process, which serves the static export
 * AND answers `/predict`. That is the deployed arrangement, and a screenshot of
 * anything else would be a screenshot of a test rig. Playwright starts and stops
 * it, so nothing is left running when the capture finishes.
 *
 * `contextOptions.reducedMotion` is set for the same reason the capture waits on
 * `data-chart-status` rather than a timer: GSAP, Lenis, framer-motion and
 * ECharts all animate, and a capture that races them is a capture that
 * eventually saves a half-drawn page.
 */

const PYTHON = JSON.stringify(
  process.env.PV_PYTHON ??
    (process.platform === 'win32'
      ? resolve(__dirname, '..', '.venv', 'Scripts', 'python.exe')
      : resolve(__dirname, '..', '.venv', 'bin', 'python')),
);

export const API_URL = 'http://127.0.0.1:8000';

export default defineConfig({
  testDir: './screenshots',
  globalSetup: './e2e/screenshot-gate.ts',
  workers: 1,
  retries: 0,
  timeout: 420_000,
  expect: { timeout: 60_000 },
  reporter: [['list']],
  use: {
    baseURL: API_URL,
    trace: 'off',
    screenshot: 'off',
    contextOptions: { reducedMotion: 'reduce' },
    ...devices['Desktop Chrome'],
    channel: process.env.PW_CHANNEL,
  },
  webServer: [
    {
      command: PYTHON + ' -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000',
      cwd: '..',
      url: API_URL + '/health',
      timeout: 180_000,
      reuseExistingServer: false,
      stdout: 'ignore',
      stderr: 'pipe',
    },
  ],
});
