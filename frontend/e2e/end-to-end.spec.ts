import { expect, test } from '@playwright/test';

/**
 * T138.1 -- MEGA TEST 6. Every page already has its own gate (analyse.spec.ts,
 * history.spec.ts, insights.spec.ts, reports.spec.ts); this is the one test
 * that proves the WIRING between them -- one recording, followed through
 * Analyse, History, Insights and a downloaded report, with no manual step
 * in between and no page read in isolation.
 */

function wav(seconds: number, hz: number, fs = 2000): Buffer {
  const frames = Math.round(seconds * fs);
  const data = Buffer.alloc(frames * 2);
  for (let n = 0; n < frames; n += 1) {
    data.writeInt16LE(Math.round(3000 * Math.sin((2 * Math.PI * hz * n) / fs)), n * 2);
  }
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(fs, 24);
  header.writeUInt32LE(fs * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

test('T138.1: analyse a recording, find it in History, see it in Insights, download its report', async ({
  page,
}) => {
  test.setTimeout(120_000);
  expect(process.env.PV_E2E_HISTORY_DB, 'the API History file is named for this run').toBeTruthy();
  // Isolated from whatever order the other spec files ran in, same as reports.spec.ts.
  expect((await page.request.delete('/api/history?confirm=true')).status()).toBe(200);

  // 1. Analyse.
  await page.goto('/', { waitUntil: 'networkidle' });
  await page.locator('input[type="file"]').first().setInputFiles({
    name: 'end-to-end.wav',
    mimeType: 'audio/wav',
    buffer: wav(6, 90),
  });
  await page.getByRole('button', { name: 'Analyse recording' }).click();
  const card = page.locator('[aria-label="Prediction result"]');
  await expect(card).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole('alert').filter({ hasText: 'No prediction was produced' })).toHaveCount(0);

  // 2. Find it in History, following the card's own link -- never a re-typed id.
  const link = card.getByRole('link', { name: 'View in History' });
  await expect(link).toBeVisible();
  await link.click();
  await page.waitForURL(/\/history\//);
  await expect(page.getByRole('button', { name: 'Open end-to-end.wav' })).toBeVisible();

  // 3. See it counted in Insights -- the empty state must be gone.
  await page.goto('/insights/', { waitUntil: 'networkidle' });
  await expect(page.getByText('No trends to show yet')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Summary', exact: true })).toBeVisible();

  // 4. Download its report from Reports, matched by the file name History saved.
  await page.goto('/reports/', { waitUntil: 'networkidle' });
  const row = page.locator('li').filter({ hasText: 'end-to-end.wav' });
  await expect(row).toBeVisible();
  const download = page.waitForEvent('download');
  await row.getByRole('button', { name: 'Download PDF' }).click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(/\.pdf$/);
  await expect(row.getByRole('alert')).toHaveCount(0);
});
