import { expect, test } from '@playwright/test';

/**
 * T130.2: a microphone recording becomes a WAV and goes down the upload path.
 *
 * Chromium's fake capture device stands in for a microphone (a real one cannot
 * be driven by a test). It plays a tone, so what this proves is the plumbing --
 * permission, capture, WAV encoding, preview, scoring -- not anything about a
 * heart sound. Its own file because the launch flags apply to the whole worker.
 */

test.use({
  permissions: ['microphone'],
  launchOptions: {
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  },
});

test('a microphone recording is previewed and scored like an upload', async ({ page }) => {
  await page.goto('/', { waitUntil: 'networkidle' });

  await page.getByRole('button', { name: 'Record from microphone' }).click();
  await expect(page.getByText(/^Recording 0:0[2-9] \//)).toBeVisible({ timeout: 20_000 });
  await page.getByRole('button', { name: 'Stop and use' }).click();

  await expect(page.getByText(/^recording-\d{8}-\d{6}\.wav$/).first()).toBeVisible();
  await expect(page.locator('[data-player-status="ready"]')).toBeVisible();

  await page.getByRole('button', { name: 'Analyse recording' }).click();
  await expect(page.locator('[aria-label="Prediction result"]')).toBeVisible({ timeout: 120_000 });
  await expect(page.getByRole('alert').filter({ hasText: 'No prediction was produced' })).toHaveCount(0);
});
