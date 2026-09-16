import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

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

/**
 * The API's History file for this run. Set once in the runner and inherited by
 * the workers, which load this config again under their own pid: T132.7 reads
 * the TinyDB file itself, so every process must name the same one.
 */
process.env.PV_E2E_HISTORY_DB ??= join(tmpdir(), 'pv-mepcg-e2e-' + String(process.pid), 'history.json');
export const E2E_HISTORY_DB: string = process.env.PV_E2E_HISTORY_DB;

/** T134.7: same isolation, for the Reports store (`cache/reports/`). */
process.env.PV_E2E_REPORTS_DB ??= join(tmpdir(), 'pv-mepcg-e2e-' + String(process.pid), 'reports.json');
process.env.PV_E2E_REPORTS_DIR ??= join(tmpdir(), 'pv-mepcg-e2e-' + String(process.pid), 'reports-files');
export const E2E_REPORTS_DB: string = process.env.PV_E2E_REPORTS_DB;

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
      // T130.7 scores recordings, and every success is saved to History. A test
      // run must never write into the operator's own History, so this process
      // gets a fresh file in the temp folder -- the same redirect the root
      // conftest applies to pytest.
      env: {
        HEARTGUARD__PATHS__CACHE__HISTORY_DB: E2E_HISTORY_DB,
        HEARTGUARD__PATHS__CACHE__REPORTS_DB: process.env.PV_E2E_REPORTS_DB as string,
        HEARTGUARD__PATHS__CACHE__REPORTS_DIR: process.env.PV_E2E_REPORTS_DIR as string,
      },
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
