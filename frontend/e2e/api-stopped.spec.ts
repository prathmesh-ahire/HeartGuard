import { expect, test } from '@playwright/test';

/**
 * T118.5: with no inference API, the analysis page shows a visible error --
 * never an empty chart and never a result panel that simply stays blank.
 *
 * These run against :8001, a plain static server over the same `frontend/out/`
 * with nothing behind `/predict`. The API is not mocked away; it is absent.
 */

const STATIC_ONLY = 'http://127.0.0.1:8001';

test('the Analyse page shows a visible error when the API is not there', async ({ page }) => {
  await page.goto(STATIC_ONLY + '/', { waitUntil: 'networkidle' });

  // The status probe failed, and the page says the service did not answer.
  await expect(page.getByText(/404|did not answer|File not found/i).first()).toBeVisible();

  // Scoring a built-in sample must fail loudly, with the reason.
  const sample = page.locator('button[data-sample-id]').first();
  await sample.click();
  await page.getByRole('button', { name: 'Analyse recording' }).click();
  const failure = page.getByRole('alert').filter({ hasText: 'No prediction was produced' });
  await expect(failure).toBeVisible();
  await expect(page.getByText('Nothing scored yet')).toHaveCount(0);
});

test('a report request fails visibly when the API is not there', async ({ page }) => {
  await page.goto(STATIC_ONLY + '/reports/', { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: 'Download objective-coverage report' }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'The report was not produced' })).toBeVisible();
});
