import { expect, test } from '@playwright/test';

/**
 * T136.6: responsiveness across breakpoints, and a real measured frame rate
 * for the Analyse hero rather than a claim about one.
 *
 * The frame-rate check counts `requestAnimationFrame` callbacks for real over
 * one second while the hero is on screen -- it is a measurement of this
 * machine's actual browser (Chromium, headless, whatever GPU path Playwright
 * gets on this box), not an assertion that `HeartScene`'s `FrameLimiter`
 * constant is set to some value. If WebGL is unavailable in this environment
 * the hero renders its designed fallback instead of a canvas, which is itself
 * the "degrade gracefully" half of T136.6 and is asserted directly.
 */

const ROUTES = ['/', '/history/', '/insights/', '/reports/', '/about/'] as const;
const VIEWPORTS = [
  { name: 'mobile', width: 375, height: 812 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1440, height: 900 },
] as const;

for (const viewport of VIEWPORTS) {
  test(`T136.6: no horizontal overflow at ${viewport.name} (${viewport.width}px)`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    for (const route of ROUTES) {
      await page.goto(route, { waitUntil: 'domcontentloaded' });
      const overflow = await page.evaluate(() => ({
        scrollWidth: document.documentElement.scrollWidth,
        clientWidth: document.documentElement.clientWidth,
      }));
      expect(
        overflow.scrollWidth,
        route + ' at ' + viewport.width + 'px: scrollWidth ' + overflow.scrollWidth + ' vs clientWidth ' + overflow.clientWidth,
      ).toBeLessThanOrEqual(overflow.clientWidth + 1);
    }
  });
}

test('T136.5/T136.6: the Analyse hero renders a canvas or its designed fallback, never neither', async ({ page }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  const canvas = page.locator('canvas').first();
  const fallback = page.getByText('Preparing the model').or(page.getByText('WebGL is unavailable'));
  await expect(canvas.or(fallback)).toBeVisible({ timeout: 15_000 });
});

test('T136.6: measured frame rate while the Analyse hero is on screen', async ({ page }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  const canvas = page.locator('canvas').first();
  const hasCanvas = await canvas.isVisible().catch(() => false);
  if (!hasCanvas) {
    test.info().annotations.push({
      type: 'note',
      description: 'No WebGL canvas in this environment (fallback shown instead) -- nothing to measure.',
    });
    return;
  }

  const frames = await page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        let count = 0;
        const start = performance.now();
        function tick() {
          count += 1;
          if (performance.now() - start < 1000) requestAnimationFrame(tick);
          else resolve(count);
        }
        requestAnimationFrame(tick);
      }),
  );

  test.info().annotations.push({ type: 'measured-fps', description: String(frames) });
  // Not a pass/fail band on the exact number -- headless Chromium's rAF cadence
  // on CI/dev hardware is not this project's target device. The floor here
  // only catches the failure mode T136.6 cares about: the page hard-stalling
  // (0-1 fps) while the hero is mounted.
  expect(frames).toBeGreaterThan(5);
});
