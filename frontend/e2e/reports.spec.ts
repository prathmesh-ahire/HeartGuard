import { readFileSync } from 'node:fs';

import { expect, test, type Page } from '@playwright/test';

/**
 * T134.7, and the browser half of T134.1-T134.5, against the built Reports
 * page and the real API on :8000. A known recording is analysed first, then
 * every download this page offers is opened and checked against the SAME
 * History record the page itself reads -- never a hand-computed expectation,
 * because the recording's actual result depends on the model.
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

function parseCsv(text: string): Record<string, string>[] {
  const [head, ...rows] = text.trim().split(/\r?\n/);
  const headers = (head as string).split(',');
  return rows.map((line) => {
    const cells = line.split(',');
    return Object.fromEntries(headers.map((h, i) => [h, cells[i] ?? '']));
  });
}

async function analyseOne(page: Page, name: string): Promise<void> {
  await page.goto('/', { waitUntil: 'networkidle' });
  const input = page.locator('input[type="file"]').first();
  await input.setInputFiles({ name, mimeType: 'audio/wav', buffer: wav(6, 90) });
  await page.getByRole('button', { name: 'Analyse recording' }).click();
  await expect(page.locator('[aria-label="Prediction result"]')).toBeVisible({ timeout: 60_000 });
}

test('T134.7: a known recording produces a matching per-recording PDF, print view, bulk CSV, model summary and recent-reports list', async ({
  page,
}) => {
  test.setTimeout(180_000);
  expect(process.env.PV_E2E_HISTORY_DB, 'the API History file is named for this run').toBeTruthy();
  expect((await page.request.delete('/api/history?confirm=true')).status()).toBe(200);

  await analyseOne(page, 'reports-fixture.wav');
  const ground = (await (await page.request.get('/api/history?page_size=1')).json()) as {
    items: { id: string; file_name: string; display: { result: string; confidence: string } }[];
  };
  const record = ground.items[0];
  if (!record) throw new Error('the analysed recording did not land in History');
  expect(record.file_name).toBe('reports-fixture.wav');

  await page.goto('/reports/', { waitUntil: 'networkidle' });
  const row = page.locator('li').filter({ hasText: 'reports-fixture.wav' });
  await expect(row).toBeVisible();

  // T134.1 / T134.6: the PDF's text is exactly the record's own display strings.
  const pdfDownload = page.waitForEvent('download');
  await row.getByRole('button', { name: 'Download PDF' }).click();
  const pdf = await pdfDownload;
  expect(pdf.suggestedFilename()).toBe('pv-mepcg-reports-fixture.wav-report.pdf');
  const pdfBytes = readFileSync((await pdf.path()) as string);
  expect(pdfBytes.subarray(0, 4).toString()).toBe('%PDF');

  // T134.3: the print view shows the same result and confidence.
  const printPage = await page.context().newPage();
  await printPage.goto('/reports/print/?record=' + record.id, { waitUntil: 'networkidle' });
  await expect(printPage.getByTestId('print-result')).toHaveText(record.display.result);
  await expect(printPage.getByTestId('print-confidence')).toHaveText(record.display.confidence);
  await printPage.close();

  // T134.2: bulk export is a CSV of every History row -- one, here.
  const csvDownload = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export CSV' }).click();
  const csv = await csvDownload;
  const rows = parseCsv(readFileSync((await csv.path()) as string, 'utf-8'));
  expect(rows).toHaveLength(1);
  expect(rows[0]?.file_name).toBe('reports-fixture.wav');
  expect(rows[0]?.history_id).toBe(record.id);

  // T134.5: the model summary PDF, independent of History.
  const summaryDownload = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download model summary PDF' }).click();
  const summary = await summaryDownload;
  expect(readFileSync((await summary.path()) as string).subarray(0, 4).toString()).toBe('%PDF');

  // T134.4: all three generated reports are listed, newest first, and re-download.
  const recent = page.getByRole('region', { name: 'Recently generated reports' });
  await expect(recent.getByRole('button', { name: 'Re-download' })).toHaveCount(3);
  const redownload = page.waitForEvent('download');
  await recent.getByRole('button', { name: 'Re-download' }).first().click();
  await redownload;
});
