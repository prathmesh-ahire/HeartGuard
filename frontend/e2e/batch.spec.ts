import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * T131.7, and the browser half of T131.1-T131.6, against the built Analyse page
 * and the real API on :8000 (whose History file Playwright points at a temp
 * folder, so these runs never write into the operator's own History).
 */

const TRAINING_A = resolve(__dirname, '..', '..', 'dataset', 'archive (3)', 'training-a');
const CORRUPT = Buffer.from('this is not a recording, only text with a .wav name. '.repeat(40));

interface Payload {
  name: string;
  mimeType: string;
  buffer: Buffer;
}

/** The first `count` PhysioNet training-a recordings, if the dataset is on this machine. */
function realRecordings(count: number): Payload[] | null {
  if (!existsSync(TRAINING_A)) return null;
  const names = readdirSync(TRAINING_A)
    .filter((name) => name.endsWith('.wav'))
    .sort()
    .slice(0, count);
  if (names.length < count) return null;
  return names.map((name) => ({ name, mimeType: 'audio/wav', buffer: readFileSync(join(TRAINING_A, name)) }));
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

function tone(name: string, seconds: number, fs = 2000): Payload {
  return { name, mimeType: 'audio/wav', buffer: wav(seconds, fs) };
}

/** RFC 4180: quoted fields may hold commas, quotes and line breaks. */
function parseCsv(text: string): Record<string, string>[] {
  const records: string[][] = [];
  let field = '';
  let record: string[] = [];
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i] as string;
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') {
      record.push(field);
      field = '';
    } else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      record.push(field);
      records.push(record);
      record = [];
      field = '';
    } else field += char;
  }
  if (field !== '' || record.length > 0) records.push([...record, field]);
  const [header, ...body] = records;
  return body.map((cells) => Object.fromEntries((header ?? []).map((name, index) => [name, cells[index] ?? ''])));
}

async function openBatch(page: Page): Promise<Locator> {
  await page.goto('/', { waitUntil: 'networkidle' });
  await page.getByRole('group', { name: 'How many recordings' }).getByRole('button', { name: 'Several recordings' }).click();
  return page.getByRole('region', { name: 'Batch analysis' });
}

async function runBatch(panel: Locator, files: Payload[], timeout = 120_000): Promise<void> {
  await panel.locator('input[type="file"]').setInputFiles(files);
  await panel.getByRole('button', { name: /^Analyse \d+ recordings?$/ }).click();
  await expect(panel.locator('tr[data-status="queued"], tr[data-status="running"]')).toHaveCount(0, { timeout });
}

/** The table as shown, top to bottom. */
async function tableRows(panel: Locator): Promise<Record<string, string>[]> {
  return panel.locator('tbody tr').evaluateAll((rows) =>
    rows.map((row) => ({
      file_name: row.getAttribute('data-file') ?? '',
      status: row.getAttribute('data-status') ?? '',
      result: row.querySelector('[data-cell="result"]')?.textContent?.trim() ?? '',
      confidence: row.querySelector('[data-cell="confidence"]')?.textContent?.trim() ?? '',
    })),
  );
}

async function historyTotal(page: Page, panel: Locator): Promise<{ total: number; names: string[] }> {
  const batchId = await panel.getAttribute('data-batch-id');
  expect(batchId).toMatch(/^[0-9a-f]{32}$/);
  const response = await page.request.get('/api/history?page_size=100&batch_id=' + String(batchId));
  expect(response.status()).toBe(200);
  const body = (await response.json()) as { total: number; items: { file_name: string }[] };
  return { total: body.total, names: body.items.map((item) => item.file_name).sort() };
}

test('T131.7: a 10-file batch with one corrupt file gives 9 results, 1 visible error, 9 History rows and a CSV that matches the table', async ({
  page,
}) => {
  test.setTimeout(600_000);
  const real = realRecordings(9);
  test.skip(real === null, 'dataset/ is not on this machine');
  const files = [...(real as Payload[])];
  files.splice(4, 0, { name: 'corrupt.wav', mimeType: 'audio/wav', buffer: CORRUPT });

  const panel = await openBatch(page);
  await runBatch(panel, files, 540_000);

  await expect(panel.locator('tr[data-status="scored"]')).toHaveCount(9);
  const failed = panel.locator('tr[data-status="failed"]');
  await expect(failed).toHaveCount(1);
  await expect(failed).toHaveAttribute('data-file', 'corrupt.wav');
  await expect(failed.locator('[data-cell="note"]')).toContainText('could not be read as audio');
  await expect(failed.locator('[data-cell="note"]')).toBeVisible();
  await expect(panel.locator('[data-count="total"] dd')).toHaveText('10');
  await expect(panel.locator('[data-count="failed"] dd')).toHaveText('1');
  await expect(panel.getByTestId('batch-progress')).toHaveText('10 of 10 finished');

  const history = await historyTotal(page, panel);
  expect(history.total).toBe(9);
  expect(history.names).toEqual((real as Payload[]).map((file) => file.name).sort());

  // Sorted first, so the CSV is shown to follow the table's order, not upload order.
  await panel.getByRole('button', { name: /^Confidence/ }).click();
  const table = await tableRows(panel);
  expect(table).toHaveLength(10);

  const download = page.waitForEvent('download');
  await panel.getByRole('button', { name: 'Export CSV' }).click();
  const saved = await download;
  expect(saved.suggestedFilename()).toMatch(/^pv-mepcg-batch-binary-[0-9a-f]{8}\.csv$/);
  const csv = parseCsv(readFileSync((await saved.path()) as string, 'utf-8'));
  expect(
    csv.map((row) => ({ file_name: row.file_name, status: row.status, result: row.result, confidence: row.confidence })),
  ).toEqual(table);
  expect(csv.filter((row) => row.status === 'scored').every((row) => row.history_id !== '')).toBe(true);
  await expect(panel.getByRole('alert').filter({ hasText: 'The CSV was not produced' })).toHaveCount(0);
});

test('T131.5: empty, wrong-format, very long and corrupt files keep their row with the reason, and the rest still score', async ({
  page,
}) => {
  test.setTimeout(240_000);
  const panel = await openBatch(page);
  await runBatch(panel, [
    { name: 'empty.wav', mimeType: 'audio/wav', buffer: Buffer.alloc(0) },
    { ...tone('notes.mp3', 3), mimeType: 'audio/mpeg' },
    tone('tone-1.wav', 8),
    tone('long.wav', 160),
    { name: 'corrupt.wav', mimeType: 'audio/wav', buffer: CORRUPT },
    tone('tone-2.wav', 6, 4000),
  ]);

  const note = (name: string) => panel.locator('tr[data-file="' + name + '"] [data-cell="note"]');
  await expect(note('empty.wav')).toContainText('is empty (0 bytes)');
  await expect(note('notes.mp3')).toContainText('is not a WAV file');
  await expect(note('long.wav')).toContainText('beyond the 150 s');
  await expect(note('corrupt.wav')).toContainText('could not be read as audio');
  await expect(panel.locator('tr[data-status="failed"]')).toHaveCount(4);
  await expect(panel.locator('tr[data-status="scored"]')).toHaveCount(2);
  expect((await historyTotal(page, panel)).total).toBe(2);
});

test('T131.1: cancel lets the file in flight finish, cancels the rest, and History holds only what finished', async ({
  page,
}) => {
  const panel = await openBatch(page);
  await panel.locator('input[type="file"]').setInputFiles([
    tone('a.wav', 8),
    tone('b.wav', 8),
    tone('c.wav', 8),
    tone('d.wav', 8),
  ]);
  await expect(panel.locator('tr[data-status="queued"]')).toHaveCount(4);
  await panel.getByRole('button', { name: 'Analyse 4 recordings' }).click();
  await expect(panel.locator('tr[data-status="running"]')).toHaveCount(1);
  await panel.getByRole('button', { name: 'Cancel batch' }).click();
  await expect(panel.locator('tr[data-status="queued"], tr[data-status="running"]')).toHaveCount(0, { timeout: 60_000 });

  const scored = await panel.locator('tr[data-status="scored"]').count();
  const cancelled = await panel.locator('tr[data-status="cancelled"]').count();
  expect(cancelled).toBeGreaterThanOrEqual(2);
  expect(scored + cancelled).toBe(4);
  expect((await historyTotal(page, panel)).total).toBe(scored);
  await expect(panel.getByTestId('batch-progress')).toHaveText('4 of 4 finished');
});

test('T131.4: two scored recordings open side by side with their waveforms, results and probabilities', async ({ page }) => {
  const panel = await openBatch(page);
  await runBatch(panel, [tone('tone-a.wav', 8), tone('tone-b.wav', 6, 4000)]);
  await expect(panel.locator('tr[data-status="scored"]')).toHaveCount(2);

  await expect(panel.getByRole('button', { name: 'Compare selected' })).toBeDisabled();
  await panel.getByRole('checkbox', { name: 'Select tone-a.wav to compare' }).check();
  await panel.getByRole('checkbox', { name: 'Select tone-b.wav to compare' }).check();
  await panel.getByRole('button', { name: 'Compare selected' }).click();

  const view = panel.getByRole('region', { name: 'Comparison' });
  await expect(view.locator('[data-player-status="ready"]')).toHaveCount(2);
  const table = await tableRows(panel);
  await expect(view.getByTestId('compare-result')).toHaveText(table.map((row) => row.result ?? ''));
  await expect(view.getByTestId('compare-confidence')).toHaveText(table.map((row) => row.confidence ?? ''));
  await expect(view.getByText('Probability of each category')).toHaveCount(2);

  await view.getByRole('button', { name: 'Close comparison' }).click();
  await expect(view).toHaveCount(0);
});
