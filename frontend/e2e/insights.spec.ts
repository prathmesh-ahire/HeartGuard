import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * T133.7, and the browser half of T133.1-T133.6, against the built Insights
 * page and the real API on :8000. A known batch of recordings is analysed
 * first, then every tile and chart section is checked against the SAME
 * `/api/history/insights` call the page itself makes -- this is a
 * screen-matches-its-own-source check, not a hand-computed expectation,
 * because the batch's actual normal/abnormal split depends on the model.
 */

/** A real mono 16-bit PCM WAV of `seconds` of a tone at `hz`. */
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

const NAMES = ['pitch-a.wav', 'pitch-b.wav', 'pitch-c.wav', 'pitch-d.wav'];

interface InsightsPayload {
  total: number;
  total_display: string;
  low_confidence: { count_display: string; percent_display: string };
  result_split: { title: string; classes: { class: string; count: number; percent_display: string }[] }[];
  trend: { in_window_display: string };
}

async function fetchInsights(page: Page, params: string): Promise<InsightsPayload> {
  return (await (await page.request.get('/api/history/insights?' + params)).json()) as InsightsPayload;
}

function tile(page: Page, label: string): Locator {
  return page.getByRole('region', { name: 'Summary' }).locator('> div').filter({ hasText: label });
}

test('T133.7: seed a known batch; every tile and chart on Insights matches the aggregate endpoint', async ({
  page,
}) => {
  test.setTimeout(180_000);
  expect(process.env.PV_E2E_HISTORY_DB, 'the API History file is named for this run').toBeTruthy();
  expect((await page.request.delete('/api/history?confirm=true')).status()).toBe(200);

  // T133.6: no history yet -- a clear prompt, not empty charts.
  await page.goto('/insights/', { waitUntil: 'networkidle' });
  await expect(page.getByText('No trends to show yet')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Analyse a recording' })).toHaveAttribute('href', '/');

  // Seed: four recordings analysed as one batch, on the binary task (the only
  // task batch mode offers -- see frontend/app/page.tsx).
  await page.goto('/', { waitUntil: 'networkidle' });
  await page.getByRole('group', { name: 'How many recordings' }).getByRole('button', { name: 'Several recordings' }).click();
  const batch = page.getByRole('region', { name: 'Batch analysis' });
  await batch.locator('input[type="file"]').setInputFiles(
    NAMES.map((name, index) => ({ name, mimeType: 'audio/wav', buffer: wav(5 + index, 60 + index * 40) })),
  );
  await batch.getByRole('button', { name: 'Analyse 4 recordings' }).click();
  await expect(batch.locator('tr[data-status="scored"]')).toHaveCount(4, { timeout: 120_000 });

  const tzOffsetMinutes = -new Date().getTimezoneOffset();
  const ground = await fetchInsights(page, 'days=30&tz_offset_minutes=' + String(tzOffsetMinutes));
  expect(ground.total).toBe(4);

  // T133.1: summary tiles read exactly the API's own display strings.
  await page.goto('/insights/', { waitUntil: 'networkidle' });
  await expect(tile(page, 'Total analyses')).toContainText(ground.total_display);
  await expect(tile(page, 'Analyses — last 30 days')).toContainText(ground.trend.in_window_display);
  await expect(tile(page, 'Low-confidence analyses')).toContainText(ground.low_confidence.percent_display);
  const binarySplit = ground.result_split.find((s) => s.title.toLowerCase().includes('binary'));
  const topClass = binarySplit?.classes.reduce<(typeof binarySplit.classes)[number] | null>(
    (best, c) => (c.count > (best?.count ?? -1) ? c : best),
    null,
  );
  if (topClass && binarySplit) {
    await expect(tile(page, 'Most common result')).toContainText(topClass.class);
    await expect(tile(page, 'Most common result')).toContainText(topClass.percent_display);
  }

  // T133.2 / T133.3: trend and outcome/confidence sections render charts, not
  // their empty-window fallbacks, once there is data in range.
  await expect(page.getByRole('region', { name: 'Trend over time' }).getByText('No analyses in this window')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Results by check type' }).locator('canvas, svg')).toHaveCount(1);
  await expect(page.getByRole('region', { name: 'Confidence distribution' }).locator('canvas, svg')).toHaveCount(1);

  // T133.4: model reliability text is the generated task list, not typed here.
  const reliability = page.getByRole('region', { name: 'Model reliability summary' });
  await expect(reliability.getByText('Binary screening (PhysioNet 2016)')).toBeVisible();

  // T133.5: the trend-window filter drives the trend tile against a fresh call.
  const week = await fetchInsights(page, 'days=7&tz_offset_minutes=' + String(tzOffsetMinutes));
  await page.getByLabel('Trend window').selectOption('7');
  await expect(tile(page, 'Analyses — last 7 days')).toContainText(week.trend.in_window_display);

  // T133.5: the task filter drives every chart together -- a task nothing was
  // analysed under shows the "no match" state, not an empty chart.
  await page.getByLabel('Check type', { exact: true }).selectOption('pascal_a');
  await expect(page.getByText('No analyses match this filter')).toBeVisible();
  await page.getByLabel('Check type', { exact: true }).selectOption('');
  await expect(tile(page, 'Total analyses')).toContainText(ground.total_display);
});
