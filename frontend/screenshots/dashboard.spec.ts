import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { expect, test, type ConsoleMessage, type Page } from '@playwright/test';

/**
 * The thirteen dashboard screenshots (T120.1 - T120.6).
 *
 * Nothing about WHICH pages are captured lives in this file. The plan --
 * number, route, interaction steps, caption, and what must and must not be on
 * screen -- is declared in `src/reporting/screenshots.py` and written to
 * `outputs/15_dashboard_screenshots/capture_plan.json` before this runs. This
 * spec is the browser half of that plan and nothing else.
 *
 * The run is gated: `playwright.screenshots.config.ts` calls
 * `scripts/45_audit_displayed_values.py --gate` in `globalSetup`, so no
 * screenshot can be taken of a page the displayed-value audit has not passed on
 * (T119.4).
 *
 * Six of the thirteen drive the real inference API. They need `dataset/` on
 * this machine, because the built-in samples are resolved from the operator's
 * own corpus rather than committed -- see `src/reporting/samples.py`.
 */

interface Step {
  action:
    | 'goto'
    | 'sample'
    | 'click'
    | 'select'
    | 'await_result'
    | 'await_patient'
    | 'await_saved';
  sample_id?: string;
  name?: string;
  label?: string;
  value?: string;
  route?: string;
}

interface Shot {
  number: number;
  shot_id: string;
  title: string;
  route: string;
  filename: string;
  caption: string;
  steps: Step[];
  must_contain: string[];
  must_not_contain: string[];
  full_page: boolean;
  height: number;
}

const OUT_DIR = resolve(__dirname, '..', '..', 'outputs', '15_dashboard_screenshots');
const plan = JSON.parse(readFileSync(join(OUT_DIR, 'capture_plan.json'), 'utf-8')) as {
  min_bytes: number;
  disclaimer: string;
  shots: Shot[];
};

/** Every ECharts container on the page has finished its first paint. */
async function chartsSettled(page: Page): Promise<void> {
  await page.waitForFunction(() => {
    const nodes = Array.from(document.querySelectorAll('[data-chart-status]'));
    return nodes.every((node) => node.getAttribute('data-chart-status') !== 'loading');
  }, undefined, { timeout: 60_000 });
}

async function runStep(page: Page, step: Step): Promise<void> {
  switch (step.action) {
    case 'goto':
      await page.goto(step.route as string, { waitUntil: 'networkidle' });
      break;
    case 'sample':
      // The sample buttons carry the record uid, not the sample id, so the id
      // is matched on the control's own value attribute set by the panel.
      await page.locator('button[data-sample-id="' + step.sample_id + '"]').click();
      break;
    case 'click':
      await page.getByRole('button', { name: step.name as string }).first().click();
      break;
    case 'select':
      await page
        .getByLabel(step.label as string)
        .selectOption(step.value as string);
      break;
    case 'await_result':
      // Waited on POSITIVELY. The first version of this asserted the ABSENCE of
      // the "nothing scored yet" panel, which is absent the instant scoring
      // starts -- so it returned immediately and SS-08 photographed a spinner.
      await expect(page.locator('[aria-label="Prediction result"]')).toBeVisible({
        timeout: 180_000,
      });
      await expect(page.getByText('Preprocessing, extracting 138 features')).toHaveCount(0);
      await expect(
        page.getByRole('alert').filter({ hasText: 'No prediction was produced' }),
      ).toHaveCount(0);
      break;
    case 'await_patient':
      await expect(page.getByText('Patient-level collapse')).toBeVisible({
        timeout: 300_000,
      });
      await expect(
        page.getByRole('button', { name: 'Scoring all four recordings…' }),
      ).toHaveCount(0);
      await expect(
        page.getByRole('alert').filter({ hasText: 'was not produced' }),
      ).toHaveCount(0);
      break;
    case 'await_saved':
      await expect(page.getByText(/^Saved /)).toBeVisible({ timeout: 180_000 });
      break;
  }
}

for (const shot of plan.shots) {
  test(shot.shot_id + ' ' + shot.title, async ({ page }, testInfo) => {
    testInfo.setTimeout(420_000);

    const errors: string[] = [];
    page.on('console', (message: ConsoleMessage) => {
      if (message.type() === 'error') errors.push(message.text());
    });
    page.on('pageerror', (error) => errors.push('uncaught: ' + error.message));

    await page.setViewportSize({ width: 1440, height: shot.height });
    const response = await page.goto(shot.route, { waitUntil: 'networkidle' });
    expect(response?.status(), shot.route).toBe(200);

    for (const step of shot.steps) await runStep(page, step);

    await chartsSettled(page);
    // Every `Reveal` section has to have entered the viewport before a
    // full-page capture, or the bottom of a long page photographs mid-fade.
    await page.evaluate(async () => {
      const step = window.innerHeight;
      for (let y = 0; y < document.body.scrollHeight; y += step) {
        window.scrollTo(0, y);
        await new Promise((done) => setTimeout(done, 60));
      }
      window.scrollTo(0, 0);
      await new Promise((done) => setTimeout(done, 250));
    });
    await chartsSettled(page);

    // Nothing visible may still be transparent. Playwright's `toBeVisible`
    // ignores opacity, so every assertion below -- and the whole first capture
    // run -- passed over a page whose sections were ALL at opacity 0 because a
    // reveal never completed. The screenshot was blank and nothing objected.
    //
    // Only INLINE opacity is checked. Framer Motion animates by writing
    // `style="opacity: …"` on the element, so a stalled reveal shows up here;
    // a disabled button faded by Tailwind's `disabled:opacity-50` class does
    // not, and is a deliberate state rather than a rendering failure.
    const faded = await page.evaluate(() => {
      const bad: string[] = [];
      for (const node of Array.from(document.querySelectorAll('main [style*="opacity"]'))) {
        const text = (node.textContent ?? '').trim();
        if (text.length === 0) continue;
        const opacity = Number.parseFloat((node as HTMLElement).style.opacity);
        if (Number.isFinite(opacity) && opacity < 0.9) {
          bad.push(node.tagName.toLowerCase() + ' @ ' + opacity + ': ' + text.slice(0, 60));
        }
      }
      return bad.slice(0, 5);
    });
    expect(faded, shot.route + ' has text rendered at opacity < 0.9').toEqual([]);

    // T120.7's "no placeholder or empty chart", asserted before the shutter.
    await expect(page.getByRole('note', { name: 'Scope and safety notice' })).toContainText(
      plan.disclaimer,
    );
    for (const text of shot.must_contain) {
      await expect(page.getByText(text, { exact: false }).first()).toBeVisible();
    }
    for (const text of shot.must_not_contain) {
      await expect(page.getByText(text, { exact: false })).toHaveCount(0);
    }

    await page.screenshot({
      path: join(OUT_DIR, shot.filename),
      fullPage: shot.full_page,
      animations: 'disabled',
    });

    expect(errors, shot.route + ' logged console errors').toEqual([]);
  });
}
