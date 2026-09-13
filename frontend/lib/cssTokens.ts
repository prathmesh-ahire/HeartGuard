/**
 * Read a colour token out of the stylesheet, for the things CSS cannot style.
 *
 * An ECharts canvas, a WaveSurfer waveform and a WebGL material are drawn by
 * script, so a Tailwind class never reaches them. Reading the custom property
 * instead of restating a hex value keeps `app/globals.css` the only place a
 * colour is defined (T128.1), and the value follows the theme because the
 * callers re-read it whenever the resolved theme changes.
 *
 * Before the stylesheet is available (server render, jsdom) it returns a
 * neutral mid grey, which reads on either ground and is not a brand colour.
 */
export type ColourToken =
  | 'ink'
  | 'ink-2'
  | 'ink-3'
  | 'line'
  | 'panel'
  | 'raised'
  | 'surface'
  | 'accent';

const FALLBACK = 'rgb(128, 128, 128)';

export function tokenColour(name: ColourToken, alpha = 1): string {
  if (typeof document === 'undefined' || typeof getComputedStyle !== 'function') return FALLBACK;
  const channels = getComputedStyle(document.documentElement)
    .getPropertyValue('--' + name)
    .trim()
    .split(/\s+/);
  if (channels.length !== 3) return FALLBACK;
  return alpha >= 1
    ? 'rgb(' + channels.join(', ') + ')'
    : 'rgba(' + channels.join(', ') + ', ' + String(alpha) + ')';
}
