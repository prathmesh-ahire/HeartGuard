import { expect, test } from '@playwright/test';

/**
 * T135.7: all six About the Model tabs render, and the expandable sections
 * T135.5 added behave correctly -- collapsed by default except the one or two
 * "key charts" a tab opens with, expandable by a click, and a deep link into
 * a collapsed section (the Performance tab's own "jump to" nav) lands open
 * through the native `<details>` fragment-navigation behaviour, not JS.
 *
 * The metrics-match-source half of T135.7 is the Python display audit
 * (`scripts/45_audit_displayed_values.py`, run in `npm run build`); this file
 * covers only what a Python script cannot check -- what renders in a browser.
 */

const TABS = [
  ['/about/', 'How it works'],
  ['/about/datasets/', 'Datasets'],
  ['/about/features/', 'Features'],
  ['/about/models/', 'Models'],
  ['/about/performance/', 'Performance'],
  ['/about/limitations/', 'Limitations'],
] as const;

test('T135.1: all six About the Model tabs render with a heading and the tab bar', async ({ page }) => {
  for (const [href, label] of TABS) {
    await page.goto(href, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('main h1, main h2').first()).toBeVisible();
    await expect(page.getByRole('navigation', { name: 'About the Model' }).getByRole('link', { name: label })).toHaveAttribute(
      'aria-current',
      'page',
    );
  }
});

test('T135.5: secondary detail is collapsed by default and opens on click', async ({ page }) => {
  await page.goto('/about/features/', { waitUntil: 'domcontentloaded' });
  const registry = page.locator('details', { hasText: 'Full registry' });
  await expect(registry).toBeVisible();
  expect(await registry.evaluate((el) => (el as HTMLDetailsElement).open)).toBe(false);
  const table = registry.locator('table');
  await expect(table).toBeHidden();

  await registry.locator('summary').click();
  expect(await registry.evaluate((el) => (el as HTMLDetailsElement).open)).toBe(true);
  await expect(table.first()).toBeVisible();
});

/** Whether the closed `<details>` a fragment id sits inside is open. */
function detailsOpenFor(page: import('@playwright/test').Page, fragmentId: string) {
  return page
    .locator('#' + fragmentId)
    .evaluate((el) => (el.closest('details') as HTMLDetailsElement | null)?.open ?? null);
}

test('T135.3/T135.5: Robustness opens with its first block only, the rest collapsed', async ({ page }) => {
  await page.goto('/about/performance/', { waitUntil: 'domcontentloaded' });
  expect(await detailsOpenFor(page, 'noise')).toBe(true);
  expect(await detailsOpenFor(page, 'duration')).toBe(false);
});

test('T135.5: a deep link into a collapsed section opens it, native details behaviour', async ({ page }) => {
  await page.goto('/about/performance/#location', { waitUntil: 'domcontentloaded' });
  await expect(page.locator('#location')).toBeInViewport();
  expect(await detailsOpenFor(page, 'location')).toBe(true);
});
