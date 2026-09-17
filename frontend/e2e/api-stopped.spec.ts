import { expect, test } from '@playwright/test';

/**
 * T118.5 / T137.4: with no inference API, every page that calls it shows a
 * visible error -- never an empty chart, an empty list, and never a panel
 * that simply stays blank.
 *
 * These run against :8001, a plain static server over the same `frontend/out/`
 * with nothing behind `/predict` or `/api/history`. The API is not mocked
 * away; it is absent. About is excluded on purpose: every one of its tabs is
 * static, codegen'd content with no runtime call.
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
  await page.getByRole('button', { name: 'Download model summary PDF' }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'The report was not produced' })).toBeVisible();
});

// Against :8001 the static server answers every path -- with a 404, not a
// connection failure -- so the API client sees an ordinary HTTP error, not the
// offline case. That renders as "... could not be loaded", not
// "The analysis service is not responding" (which is `ApiError.offline` only).

test('the History page shows a visible error when the API is not there', async ({ page }) => {
  await page.goto(STATIC_ONLY + '/history/', { waitUntil: 'networkidle' });
  await expect(
    page.getByRole('alert').filter({ hasText: 'History could not be loaded' }),
  ).toBeVisible();
});

test('the Insights page shows a visible error when the API is not there', async ({ page }) => {
  await page.goto(STATIC_ONLY + '/insights/', { waitUntil: 'networkidle' });
  await expect(
    page.getByRole('alert').filter({ hasText: 'Insights could not be loaded' }),
  ).toBeVisible();
});
