import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * T128.7, browser half: every Phase 128 primitive renders in both themes with
 * the scheme's own colours, and reduced motion switches the animation off.
 *
 * The colour assertions read computed styles, so they test what the browser
 * painted rather than which class a component asked for. The one expected
 * value per ground is the token from `app/globals.css`, restated here as the
 * RGB the browser reports -- `tests/test_design_refresh.py` is what ties those
 * tokens to the user's four colours.
 */

const GROUND = { light: 'rgb(242, 224, 210)', dark: 'rgb(23, 13, 16)' } as const;
const PANEL = { light: 'rgb(255, 255, 255)', dark: 'rgb(34, 20, 24)' } as const;
const MODES = ['light', 'dark'] as const;
const OVERLAYS = ['modal', 'drawer', 'sheet'] as const;

async function openDesign(page: Page): Promise<void> {
  await page.goto('/design/', { waitUntil: 'domcontentloaded' });
  await expect(page.locator('[data-refresh-preview]')).toBeVisible();
  // The server HTML is on screen before React has attached a single handler,
  // and on a cold `next dev` that gap is long: the first run of this spec
  // clicked "Open modal", focused the button, and got no dialog. The theme
  // button's label changes only in an effect, so it is proof of hydration.
  await expect(page.getByRole('button', { name: /^Switch to (light|dark) theme$/ })).toBeVisible();
}

function preview(page: Page, mode: (typeof MODES)[number]): Locator {
  return page.locator(`[data-theme-preview="${mode}"]`);
}

async function style(locator: Locator, property: string): Promise<string> {
  return locator.evaluate(
    (element, name) => getComputedStyle(element).getPropertyValue(name),
    property,
  );
}

for (const pageTheme of MODES) {
  test(`every primitive renders in both themes, page theme ${pageTheme}`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.addInitScript((theme) => window.localStorage.setItem('theme', theme), pageTheme);

    await openDesign(page);
    await expect(page.locator('html')).toHaveClass(new RegExp('\\b' + pageTheme + '\\b'));

    for (const mode of MODES) {
      const panel = preview(page, mode);
      await panel.scrollIntoViewIfNeeded();
      expect(await style(panel, 'background-color')).toBe(GROUND[mode]);

      for (const text of [
        'No analyses yet',
        'No trends to show yet',
        'Nothing matches these filters',
        'The analysis service is not responding',
      ]) {
        await expect(panel.getByText(text)).toBeVisible();
      }
      await expect(panel.getByRole('alert')).toHaveCount(1);
      await expect(panel.getByRole('status')).toHaveCount(2);
      expect(await panel.locator('[data-skeleton]').count()).toBeGreaterThan(0);
      await expect(panel.getByRole('button', { name: 'Upload' })).toBeVisible();

      for (const kind of OVERLAYS) {
        await panel.getByTestId(`open-${kind}-${mode}`).click();
        const dialog = page.getByRole('dialog');
        await expect(dialog).toBeVisible();
        expect(await style(dialog.locator('[data-panel]'), 'background-color')).toBe(PANEL[mode]);
        await page.keyboard.press('Escape');
        await expect(page.getByRole('dialog')).toHaveCount(0);
        await expect(panel.getByTestId(`open-${kind}-${mode}`)).toBeFocused();
      }
    }

    expect(errors).toEqual([]);
  });
}

test.describe('with reduced motion', () => {
  test.use({ contextOptions: { reducedMotion: 'reduce' } });

  test('no primitive animates', async ({ page }) => {
    await openDesign(page);
    const panel = preview(page, 'light');
    await panel.scrollIntoViewIfNeeded();

    expect(await style(panel.locator('[data-skeleton]').first(), 'animation-name')).toBe('none');

    for (const kind of OVERLAYS) {
      await panel.getByTestId(`open-${kind}-light`).click();
      const dialog = page.getByRole('dialog');
      await expect(dialog).toBeVisible();
      expect(await style(dialog.locator('[data-panel]'), 'animation-name')).toBe('none');
      expect(await style(dialog.locator('[data-scrim]'), 'animation-name')).toBe('none');
      await page.keyboard.press('Escape');
      await expect(page.getByRole('dialog')).toHaveCount(0);
    }

    // Nothing is still moving: CSS and Web Animations both report through here.
    await expect
      .poll(() =>
        page.evaluate(
          () => document.getAnimations().filter((animation) => animation.playState === 'running').length,
        ),
      )
      .toBe(0);
  });
});

test.describe('without reduced motion', () => {
  test.use({ contextOptions: { reducedMotion: 'no-preference' } });

  test('the same primitives do animate, so the check above is not vacuous', async ({ page }) => {
    await openDesign(page);
    const panel = preview(page, 'light');
    await panel.scrollIntoViewIfNeeded();

    expect(await style(panel.locator('[data-skeleton]').first(), 'animation-name')).toBe('skeleton');

    await panel.getByTestId('open-drawer-light').click();
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    expect(await style(dialog.locator('[data-panel]'), 'animation-name')).toBe('drawer-in-right');
    await page.keyboard.press('Escape');
  });
});
