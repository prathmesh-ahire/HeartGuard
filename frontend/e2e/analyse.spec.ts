import { existsSync, readFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * T130.7, and the browser half of T130.1-T130.6, against the built Analyse page
 * and the real API on :8000 (whose History file Playwright points at a temp
 * folder, so these runs never write into the operator's own History).
 *
 * The known records are the two binary samples whose five out-of-fold repeats
 * in EXP-A2 all agree. What is compared is the CLASS, never the probability:
 * the deployed bundle is refit on all 3,240 records, so its probability for a
 * training record is in-sample and matching an out-of-fold one would be
 * asserting leakage (the same reasoning as T121.4).
 */

interface Sample {
  sample_id: string;
  record_uid: string;
  reference: null | { predicted_classes: string[]; folds_agree: boolean; true_class: string };
}

const prediction = JSON.parse(
  readFileSync(join(__dirname, '..', 'lib', 'generated', 'prediction.json'), 'utf-8'),
) as { samples: Sample[] };

const RUN = 'Analyse recording';
const KNOWN = prediction.samples.filter((sample) => sample.reference?.folds_agree === true);

/** `D1_training-b_b0033` -> `dataset/archive (3)/training-b/b0033.wav`, if it is on this machine. */
function corpusFile(recordUid: string): string | null {
  const match = /^D1_(training-[a-f])_(\w+)$/.exec(recordUid);
  if (match === null) return null;
  const path = resolve(__dirname, '..', '..', 'dataset', 'archive (3)', match[1] as string, match[2] + '.wav');
  return existsSync(path) ? path : null;
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

async function analyse(page: Page): Promise<Locator> {
  await page.getByRole('button', { name: RUN }).click();
  const card = page.locator('[aria-label="Prediction result"]');
  // Waited on positively: the absence of the empty state is not a result.
  await expect(card).toBeVisible({ timeout: 120_000 });
  await expect(page.getByRole('alert').filter({ hasText: 'No prediction was produced' })).toHaveCount(0);
  return card;
}

interface HistoryRow {
  file_name: string;
  task: string;
  result: string;
  source_kind: string;
}

/** Follow the card's "View in History" link to the row the API stored. */
async function historyRow(page: Page, card: Locator): Promise<HistoryRow> {
  const link = card.getByRole('link', { name: 'View in History' });
  await expect(link).toBeVisible();
  const href = (await link.getAttribute('href')) ?? '';
  const id = new URL(href, 'http://127.0.0.1').searchParams.get('record');
  expect(id, 'the link names the History row').toBeTruthy();
  const response = await page.request.get('/api/history/' + encodeURIComponent(id as string));
  expect(response.status()).toBe(200);
  return (await response.json()) as HistoryRow;
}

test.beforeEach(async ({ page }) => {
  await page.goto('/', { waitUntil: 'networkidle' });
});

test('the known records are the two whose out-of-fold repeats agree', () => {
  expect(KNOWN.map((sample) => sample.sample_id).sort()).toEqual(['binary-abnormal', 'binary-normal']);
});

for (const sample of KNOWN) {
  test('T130.7: uploading ' + sample.sample_id + ' matches its stored out-of-fold class and lands in History', async ({
    page,
  }) => {
    const file = corpusFile(sample.record_uid);
    test.skip(file === null, 'dataset/ is not on this machine');
    const stored = sample.reference?.predicted_classes[0] as string;

    await page.locator('input[type="file"]').first().setInputFiles(file as string);
    await expect(page.locator('[data-player-status="ready"]')).toBeVisible();
    const card = await analyse(page);

    await expect(card.getByTestId('result-class')).toHaveText(stored);
    const row = await historyRow(page, card);
    expect(row.result).toBe(stored);
    expect(row.task).toBe('binary');
    expect(row.source_kind).toBe('upload');
    expect(row.file_name).toBe(basename(file as string));
  });
}

test('a built-in sample is scored, shows its dataset label, and is saved as a sample', async ({ page }) => {
  const sample = KNOWN[0] as Sample;
  const button = page.locator('button[data-sample-id="' + sample.sample_id + '"]');
  test.skip(await button.isDisabled(), 'the sample is not reachable on this machine');
  const stored = sample.reference?.predicted_classes[0] as string;

  await button.click();
  await expect(page.locator('[data-player-status="ready"]')).toBeVisible();
  const card = await analyse(page);

  await expect(card.getByTestId('result-class')).toHaveText(stored);
  await expect(card.getByText('Dataset label: ' + stored)).toBeVisible();
  const row = await historyRow(page, card);
  expect(row.result).toBe(stored);
  expect(row.source_kind).toBe('sample');
  expect(row.file_name).toBe(sample.sample_id);
});

test('T130.4: each check has a plain description, and a two-task check asks which', async ({ page }) => {
  const checks = page.getByRole('group', { name: 'What to check' }).getByRole('button');
  await expect(checks).toHaveCount(3);
  await expect(page.getByRole('group', { name: 'Which set of categories' })).toHaveCount(0);

  await checks.filter({ hasText: 'Sound type' }).click();
  const which = page.getByRole('group', { name: 'Which set of categories' });
  await expect(which.getByRole('button')).toHaveText(['PASCAL A — four classes', 'PASCAL B — three classes']);
  await expect(which.getByRole('button', { name: 'PASCAL A — four classes' })).toHaveAttribute('aria-pressed', 'true');
  // The recording-quality caveat travels with the task, from the payload.
  await expect(page.getByText(/not a four-class cardiac classifier/)).toBeVisible();
});

test('T130.3: the waveform plays and scrubs, and the playhead follows', async ({ page }) => {
  await page.locator('input[type="file"]').first().setInputFiles({
    name: 'tone.wav',
    mimeType: 'audio/wav',
    buffer: wav(8),
  });
  await expect(page.locator('[data-player-status="ready"]')).toBeVisible();
  const readout = page.getByTestId('player-clock');
  await expect(readout).toHaveText('0:00 / 0:08');

  // Click three quarters of the way along the waveform. Through the locator,
  // never `page.mouse.click` at measured coordinates: the waveform sits below
  // the fold, a raw click there lands outside the viewport and reaches nothing
  // (`elementFromPoint` returned null), while a locator click scrolls it in.
  const waveform = page.getByRole('img', { name: 'Waveform of tone.wav' });
  const box = await waveform.boundingBox();
  expect(box).not.toBeNull();
  await waveform.click({ position: { x: (box?.width ?? 0) * 0.75, y: (box?.height ?? 0) / 2 } });
  await expect(readout).toHaveText(/^0:0[56] \/ 0:08$/);

  // The keyboard-reachable scrubber moves the same playhead.
  await page.getByRole('slider', { name: 'Playback position' }).fill('2');
  await expect(readout).toHaveText('0:02 / 0:08');

  await page.getByRole('button', { name: 'Play' }).click();
  await expect(page.getByRole('button', { name: 'Pause' })).toBeVisible();
  await expect(readout).not.toHaveText('0:02 / 0:08', { timeout: 10_000 });
});

test('T130.6: the result offers its report as a download', async ({ page }) => {
  await page.locator('input[type="file"]').first().setInputFiles({
    name: 'tone.wav',
    mimeType: 'audio/wav',
    buffer: wav(8),
  });
  const card = await analyse(page);
  await expect(card.getByText('Saved to History')).toBeVisible();

  const download = page.waitForEvent('download', { timeout: 180_000 });
  await card.getByRole('button', { name: 'Download report' }).click();
  expect((await download).suggestedFilename()).toMatch(/\.docx$/);
  await expect(card.getByRole('alert').filter({ hasText: 'The report was not produced' })).toHaveCount(0);
});
