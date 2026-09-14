import { existsSync, readFileSync } from 'node:fs';

import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * T132.7, and the browser half of T132.1-T132.6, against the built History page
 * and the real API on :8000. After every change the test reads the TinyDB FILE
 * the API writes (Playwright points it at a temp folder, never the operator's
 * own History), then reloads the page and checks the change is still shown.
 */

interface StoredRow {
  id: string;
  created_at: string;
  file_name: string;
  result: string;
  notes: string;
  tags: string[];
  batch_id: string | null;
}

/** Every row in the History database file, by file name. */
function onDisk(): Map<string, StoredRow> {
  const path = process.env.PV_E2E_HISTORY_DB as string;
  const rows = new Map<string, StoredRow>();
  if (!existsSync(path)) return rows;
  const table = (JSON.parse(readFileSync(path, 'utf-8')) as { records?: Record<string, StoredRow> }).records ?? {};
  for (const row of Object.values(table)) rows.set(row.file_name, row);
  return rows;
}

/** A real mono 16-bit PCM WAV of `seconds` of a 40 Hz tone. */
function wav(seconds: number, fs = 2000): Buffer {
  const frames = Math.round(seconds * fs);
  const data = Buffer.alloc(frames * 2);
  for (let n = 0; n < frames; n += 1) {
    data.writeInt16LE(Math.round(3000 * Math.sin((2 * Math.PI * 40 * n) / fs)), n * 2);
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

const NAMES = ['tone-alpha.wav', 'tone-beta.wav', 'tone-gamma.wav'];

function rowsOf(page: Page): Locator {
  return page.getByRole('region', { name: 'Saved analyses' }).locator('tbody tr');
}

async function shownFiles(page: Page): Promise<string[]> {
  const names = await rowsOf(page).evaluateAll((rows) => rows.map((row) => row.getAttribute('data-file') ?? ''));
  return names.sort();
}

async function reload(page: Page): Promise<void> {
  await page.reload({ waitUntil: 'networkidle' });
}

/** The viewer's today, as the date input takes it. */
function today(): string {
  const now = new Date();
  return [now.getFullYear(), String(now.getMonth() + 1).padStart(2, '0'), String(now.getDate()).padStart(2, '0')].join('-');
}

test('T132.7: create, filter, edit, compare and delete through the page; every change is in TinyDB and survives a reload', async ({
  page,
}) => {
  test.setTimeout(300_000);
  expect(process.env.PV_E2E_HISTORY_DB, 'the API History file is named for this run').toBeTruthy();
  // A fresh install: earlier specs in this run saved their own analyses.
  expect((await page.request.delete('/api/history?confirm=true')).status()).toBe(200);

  // T132.6: the empty state links to Analyse.
  await page.goto('/history/', { waitUntil: 'networkidle' });
  await expect(page.getByText('No analyses yet')).toBeVisible();
  await expect(page.getByRole('main').getByRole('link', { name: 'Analyse a recording' }).last()).toHaveAttribute('href', '/');

  // Create: three recordings analysed through the Analyse page as one batch.
  await page.goto('/', { waitUntil: 'networkidle' });
  await page.getByRole('group', { name: 'How many recordings' }).getByRole('button', { name: 'Several recordings' }).click();
  const batch = page.getByRole('region', { name: 'Batch analysis' });
  await batch.locator('input[type="file"]').setInputFiles(
    NAMES.map((name, index) => ({ name, mimeType: 'audio/wav', buffer: wav(6 + index, index === 1 ? 4000 : 2000) })),
  );
  await batch.getByRole('button', { name: 'Analyse 3 recordings' }).click();
  await expect(batch.locator('tr[data-status="scored"]')).toHaveCount(3, { timeout: 120_000 });
  const batchId = (await batch.getAttribute('data-batch-id')) as string;
  await batch.getByRole('link', { name: 'View batch in History' }).click();

  await expect(page).toHaveURL(new RegExp('/history/\\?batch=' + batchId));
  await expect(rowsOf(page)).toHaveCount(3);
  expect(await shownFiles(page)).toEqual(NAMES);
  let stored = onDisk();
  expect([...stored.keys()].sort()).toEqual(NAMES);
  expect([...stored.values()].every((row) => row.batch_id === batchId)).toBe(true);
  // T132.1: the table shows the stored result and the API's confidence string.
  const alpha = stored.get('tone-alpha.wav') as StoredRow;
  const alphaRow = rowsOf(page).and(page.locator('[data-file="tone-alpha.wav"]'));
  await expect(alphaRow.locator('[data-cell="result"]')).toContainText(alpha.result);
  const api = (await (await page.request.get('/api/history/' + alpha.id)).json()) as { display: { confidence: string } };
  await expect(alphaRow.locator('[data-cell="confidence"]')).toHaveText(api.display.confidence);

  // Filter: search, then a reload keeps it.
  await page.getByRole('button', { name: 'Clear filters' }).click();
  await page.getByLabel('Search file names, notes and tags').fill('beta');
  await expect(rowsOf(page)).toHaveCount(1);
  await reload(page);
  await expect(page.getByLabel('Search file names, notes and tags')).toHaveValue('beta');
  expect(await shownFiles(page)).toEqual(['tone-beta.wav']);
  await page.getByRole('button', { name: 'Clear filters' }).click();

  // Filter: result, and a date range of today and of a day long gone.
  await page.getByLabel('Result').selectOption(alpha.result);
  const sameResult = NAMES.filter((name) => stored.get(name)?.result === alpha.result);
  await expect(rowsOf(page)).toHaveCount(sameResult.length);
  expect(await shownFiles(page)).toEqual(sameResult);
  await page.getByRole('button', { name: 'Clear filters' }).click();
  await page.getByLabel('From', { exact: true }).fill(today());
  await page.getByLabel('To', { exact: true }).fill(today());
  await expect(rowsOf(page)).toHaveCount(3);
  await page.getByLabel('From', { exact: true }).fill('2020-01-01');
  await page.getByLabel('To', { exact: true }).fill('2020-01-02');
  await expect(page.getByText('Nothing matches these filters')).toBeVisible();
  await page.getByRole('button', { name: 'Clear filters' }).first().click();
  await expect(rowsOf(page)).toHaveCount(3);

  // Edit: notes and tags in the drawer, in the file, and still there after a reload.
  await page.getByRole('button', { name: 'Open tone-alpha.wav' }).click();
  let drawer = page.getByRole('dialog', { name: 'tone-alpha.wav' });
  await expect(drawer.getByText('Audio not kept')).toBeVisible();
  await expect(drawer.getByTestId('record-confidence')).toHaveText(api.display.confidence);
  await drawer.getByLabel('Notes', { exact: true }).fill('Repeat at the next visit');
  await drawer.getByLabel('Tags', { exact: true }).fill('clinic-a, repeat, Clinic-A');
  await drawer.getByRole('button', { name: 'Save notes and tags' }).click();
  await expect(drawer.getByRole('status')).toHaveText('Saved');
  stored = onDisk();
  expect(stored.get('tone-alpha.wav')?.notes).toBe('Repeat at the next visit');
  expect(stored.get('tone-alpha.wav')?.tags).toEqual(['clinic-a', 'repeat']);
  await reload(page); // the address names the open record
  drawer = page.getByRole('dialog', { name: 'tone-alpha.wav' });
  await expect(drawer).toBeVisible();
  await expect(drawer.getByLabel('Notes', { exact: true })).toHaveValue('Repeat at the next visit');
  await expect(drawer.getByLabel('Tags', { exact: true })).toHaveValue('clinic-a, repeat');
  await drawer.getByRole('button', { name: 'Close' }).click();
  await expect(drawer).toHaveCount(0);
  await page.getByLabel('Tag', { exact: true }).fill('CLINIC-A');
  await expect(rowsOf(page)).toHaveCount(1);
  await page.getByLabel('Search file names, notes and tags').fill('next visit');
  await expect(page).toHaveURL(/q=next\+visit/);
  await expect.poll(() => shownFiles(page)).toEqual(['tone-alpha.wav']);
  await page.getByRole('button', { name: 'Clear filters' }).click();
  await expect(rowsOf(page)).toHaveCount(3);

  // Compare: two entries through the Phase 131 view.
  await expect(page.getByRole('button', { name: 'Compare selected' })).toBeDisabled();
  await page.getByRole('checkbox', { name: 'Select tone-alpha.wav to compare' }).check();
  await page.getByRole('checkbox', { name: 'Select tone-beta.wav to compare' }).check();
  await page.getByRole('button', { name: 'Compare selected' }).click();
  const view = page.getByRole('region', { name: 'Comparison' });
  await expect(view.getByText('Audio not kept')).toHaveCount(2);
  await expect(view.getByTestId('compare-result')).toHaveText([
    stored.get('tone-alpha.wav')?.result as string,
    stored.get('tone-beta.wav')?.result as string,
  ]);
  await view.getByRole('button', { name: 'Close comparison' }).click();

  // Delete one, undo it: the same entry is back in the file.
  const gamma = stored.get('tone-gamma.wav') as StoredRow;
  await page.getByRole('button', { name: 'Open tone-gamma.wav' }).click();
  await page.getByRole('dialog', { name: 'tone-gamma.wav' }).getByRole('button', { name: 'Delete this analysis' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Deleted tone-gamma.wav.' })).toBeVisible();
  await expect(rowsOf(page)).toHaveCount(2);
  expect(onDisk().has('tone-gamma.wav')).toBe(false);
  await page.getByRole('button', { name: 'Undo' }).click();
  await expect(rowsOf(page)).toHaveCount(3);
  const back = onDisk().get('tone-gamma.wav');
  expect([back?.id, back?.created_at]).toEqual([gamma.id, gamma.created_at]);

  // Delete one for good; a reload agrees.
  await page.getByRole('button', { name: 'Open tone-beta.wav' }).click();
  await page.getByRole('dialog', { name: 'tone-beta.wav' }).getByRole('button', { name: 'Delete this analysis' }).click();
  await expect(rowsOf(page)).toHaveCount(2);
  await reload(page);
  expect(await shownFiles(page)).toEqual(['tone-alpha.wav', 'tone-gamma.wav']);
  expect([...onDisk().keys()].sort()).toEqual(['tone-alpha.wav', 'tone-gamma.wav']);

  // Delete all: refused until confirmed, then empty in the file and after a reload.
  await page.getByRole('button', { name: 'Delete all' }).click();
  const confirm = page.getByRole('dialog', { name: 'Delete all history?' });
  await confirm.getByRole('button', { name: 'Keep history' }).click();
  await expect(rowsOf(page)).toHaveCount(2);
  expect(onDisk().size).toBe(2);
  await page.getByRole('button', { name: 'Delete all' }).click();
  await confirm.getByRole('button', { name: 'Delete everything' }).click();
  await expect(page.getByText('No analyses yet')).toBeVisible();
  expect(onDisk().size).toBe(0);
  await reload(page);
  await expect(page.getByText('No analyses yet')).toBeVisible();
});

test('T132.3: a single analysis opens from its Analyse link, with its waveform when the recording is a sample', async ({
  page,
}) => {
  await page.goto('/', { waitUntil: 'networkidle' });
  const button = page.locator('button[data-sample-id="binary-normal"]');
  test.skip(await button.isDisabled(), 'the sample is not reachable on this machine');
  await button.click();
  await page.getByRole('button', { name: 'Analyse recording' }).click();
  const card = page.locator('[aria-label="Prediction result"]');
  await expect(card).toBeVisible({ timeout: 120_000 });
  const shown = await card.getByTestId('result-class').textContent();
  await card.getByRole('link', { name: 'View in History' }).click();

  const drawer = page.getByRole('dialog', { name: 'binary-normal' });
  await expect(drawer.getByTestId('record-result')).toHaveText(shown ?? '');
  await expect(drawer.locator('[data-player-status="ready"]')).toBeVisible();
  await expect(drawer.getByText('Audio not kept')).toHaveCount(0);
});
