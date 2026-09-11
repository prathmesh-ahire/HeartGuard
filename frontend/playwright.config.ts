import { resolve } from 'node:path';

import { defineConfig, devices } from '@playwright/test';

/**
 * Browser tests (T118.2-T118.5) against the BUILT site, never `next dev`.
 *
 * Two servers, started and stopped by Playwright itself so nothing is left
 * running afterwards:
 *
 * - :8000 -- the real FastAPI process (`src.api.main`), which serves the static
 *   export from `frontend/out/` AND answers `/predict`. This is the deployed
 *   arrangement, one process on one port.
 * - :8001 -- the same `frontend/out/` from a plain static file server with NO
 *   inference API behind it. T118.5's "API stopped" is this server for real:
 *   the pages are up and nothing answers `/predict`. It is not a mocked route.
 *
 * `PV_PYTHON` names the interpreter; it defaults to the project's `.venv`.
 */

// Absolute, because the two servers start in different directories: the API
// from the repository root, the static server from `frontend/`.
const PYTHON = JSON.stringify(
  process.env.PV_PYTHON ??
    (process.platform === 'win32'
      ? resolve(__dirname, '..', '.venv', 'Scripts', 'python.exe')
      : resolve(__dirname, '..', '.venv', 'bin', 'python')),
);

export const API_URL = 'http://127.0.0.1:8000';
export const STATIC_ONLY_URL = 'http://127.0.0.1:8001';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 90_000,
  expect: { timeout: 30_000 },
  reporter: [['list']],
  use: {
    baseURL: API_URL,
    trace: 'off',
    screenshot: 'off',
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
    {
      command: PYTHON + ' -m http.server 8001 --bind 127.0.0.1 --directory out',
      url: STATIC_ONLY_URL + '/index.html',
      timeout: 60_000,
      reuseExistingServer: false,
      stdout: 'ignore',
      stderr: 'ignore',
    },
  ],
});
