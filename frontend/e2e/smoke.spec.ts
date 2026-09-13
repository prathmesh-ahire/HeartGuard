import { expect, test, type ConsoleMessage } from '@playwright/test';

import { SHOW_SCREENING_NOTICE } from '../lib/flags';
import { ABOUT_TABS, LEGACY_REDIRECTS, ROUTES } from '../lib/routes';

/**
 * T118.2 / T127.7: every declared route renders in a real browser with no
 * console error, the navigation holds exactly the five product pages, and every
 * retired route redirects to its new home.
 *
 * The route lists are imported from `lib/routes.ts`, the same declaration the
 * navigation is built from, so a page added there is smoke-tested without
 * anyone remembering to add it here.
 */

test('the navigation holds exactly the five product pages', async ({ page }) => {
  // The rail is static HTML. Waiting for network idle here would wait on the
  // Analyse page's first calls to a cold API, which is what the next test is for.
  await page.goto('/history/', { waitUntil: 'domcontentloaded' });
  const links = page.getByRole('navigation', { name: 'Primary' }).getByRole('link');
  await expect(links).toHaveCount(5);
  await expect(links).toHaveText(ROUTES.map((route) => route.label));
});

const PAGES = [...ROUTES, ...ABOUT_TABS.filter((tab) => tab.href !== '/about/')];

for (const route of PAGES) {
  test('renders ' + route.href + ' with no console error', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', (message: ConsoleMessage) => {
      if (message.type() === 'error') errors.push(message.text());
    });
    page.on('pageerror', (error) => errors.push('uncaught: ' + error.message));

    const response = await page.goto(route.href, { waitUntil: 'networkidle' });
    expect(response?.status(), route.href + ' status').toBe(200);

    await expect(page.locator('main h1').first()).toBeVisible();

    const notice = page.getByRole('note', { name: 'Scope and safety notice' });
    await expect(notice).toHaveCount(SHOW_SCREENING_NOTICE ? 1 : 0);

    expect(errors, route.href + ' logged console errors').toEqual([]);
  });
}

for (const [from, to] of Object.entries(LEGACY_REDIRECTS)) {
  test('redirects ' + from + ' to ' + to, async ({ page }) => {
    await page.goto(from);
    await page.waitForURL((url) => url.pathname === to);
    await expect(page.locator('main h1').first()).toBeVisible();
  });
}
