import { expect, test, type ConsoleMessage } from '@playwright/test';

import { ALL_ROUTES, ROUTES } from '../lib/routes';

/**
 * T118.2 / T118.3: every declared route renders in a real browser with no
 * console error, and carries the screening-only disclaimer.
 *
 * The route list is imported from `lib/routes.ts`, the same declaration the
 * navbar is built from, so a page added to the navigation is smoke-tested
 * without anyone remembering to add it here.
 */

test('the twelve document routes are the declared ones', () => {
  expect(ROUTES).toHaveLength(12);
});

for (const route of ALL_ROUTES) {
  test('renders ' + route.href + ' with no console error and the disclaimer', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', (message: ConsoleMessage) => {
      if (message.type() === 'error') errors.push(message.text());
    });
    page.on('pageerror', (error) => errors.push('uncaught: ' + error.message));

    const response = await page.goto(route.href, { waitUntil: 'networkidle' });
    expect(response?.status(), route.href + ' status').toBe(200);

    await expect(page.locator('main h1').first()).toBeVisible();
    await expect(page.getByText('Route scaffolded; content not built yet.')).toHaveCount(0);

    const notice = page.getByRole('note', { name: 'Scope and safety notice' });
    await expect(notice).toBeVisible();
    await expect(notice).toContainText('it does not diagnose');
    await expect(notice).toContainText('Screening and research use only.');

    expect(errors, route.href + ' logged console errors').toEqual([]);
  });
}
