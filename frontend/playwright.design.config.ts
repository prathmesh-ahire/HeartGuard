import { defineConfig, devices } from '@playwright/test';

/**
 * The design reference in a real browser (T128.7).
 *
 * Against `next dev`, not the built site, because T127.5 keeps `/design` out of
 * the production build -- `next dev` is the one server that serves it. Nothing
 * here touches the API. Playwright starts the server and stops it afterwards;
 * nothing is left running.
 *
 * Do not run this at the same time as a build: both write `.next/`.
 */
export const DESIGN_URL = 'http://127.0.0.1:3100';

export default defineConfig({
  testDir: './e2e-design',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 240_000,
  expect: { timeout: 60_000 },
  reporter: [['list']],
  use: {
    baseURL: DESIGN_URL,
    trace: 'off',
    screenshot: 'off',
    ...devices['Desktop Chrome'],
    channel: process.env.PW_CHANNEL,
  },
  webServer: {
    command: 'npx next dev --hostname 127.0.0.1 --port 3100',
    url: DESIGN_URL + '/design/',
    timeout: 300_000,
    reuseExistingServer: false,
    stdout: 'ignore',
    stderr: 'pipe',
  },
});
