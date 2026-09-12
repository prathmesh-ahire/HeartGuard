import { theme } from '@/lib/generated';

/**
 * The design tokens (T111.1).
 *
 * **Colour is not defined here.** It comes from `generated/theme.json`, which
 * `src/reporting/plot_style.py` writes -- the same module every matplotlib
 * figure draws through. That is the point of T111.1: a bar chart rendered in
 * the browser and the 300 dpi PNG sitting next to it must colour the same
 * series the same way, and they cannot if the palette is typed out twice.
 *
 * The type scale and spacing ARE defined here, because the figures and the
 * pages do not share them: a figure is 3.5 to 10 inches wide on paper, a page
 * is whatever the viewport is.
 */

/** Okabe-Ito, in the fixed order every figure uses. Index 0 is series 1. */
export const SERIES_COLORS: readonly string[] = theme.palette.series;

/**
 * The brand accent, as a literal.
 *
 * Every other surface takes its colour from a CSS variable, but a WebGL
 * material is not styled by CSS and needs a value it can pass to Three. This
 * is decoration -- the hero heart -- and never a data mark: a mark that
 * encodes a value still comes from `SERIES_COLORS`, which the figures share.
 */
export const BRAND = {
  accent: '#e11d48',
  accentDark: '#fb7185',
} as const;

/** Semantic roles, so a page never indexes the palette by a magic number. */
export const ROLE_COLORS = theme.palette.roles;

export function seriesColor(index: number): string {
  const palette = SERIES_COLORS;
  return palette[index % palette.length] ?? palette[0] ?? '#0072B2';
}

/**
 * The type scale. Ratio 1.2 (minor third) from a 14px base -- tight enough that
 * a dense results table and a page heading sit in the same document, wide
 * enough that hierarchy is legible without weight changes.
 */
export const TYPE_SCALE = {
  micro: 'font-mono text-label-sm uppercase',
  /** A provenance line: monospace for the paths, but not shouted. */
  provenance: 'font-mono text-body-sm normal-case tracking-normal',
  caption: 'text-body-sm',
  body: 'text-body-md',
  lead: 'text-body-lg',
  h3: 'text-headline-sm',
  h2: 'text-headline-md',
  h1: 'text-headline-lg',
  display: 'text-headline-xl',
  /** Metrics are tabular so digits line up column to column. */
  stat: 'stat font-mono text-telemetry',
} as const;

/** A 4px base step. Named so a card and a section cannot drift apart. */
export const SPACING = {
  tight: 'gap-2',
  normal: 'gap-4',
  section: 'gap-8',
  page: 'space-y-10',
} as const;

export const SURFACE = {
  card: 'rounded-xl border border-line bg-panel shadow-panel',
  glass: 'rounded-xl border border-line bg-panel/80 backdrop-blur-md shadow-panel',
  sunken: 'rounded-lg border border-line bg-sunken',
  muted: 'text-ink-2',
  subtle: 'text-ink-3',
} as const;

/**
 * Status colours. Deliberately NOT the series palette: a badge saying
 * "abnormal" must not borrow the colour a chart is using for series 2 in the
 * same viewport, or the reader will read a legend that is not there.
 */
export const STATUS = {
  neutral: 'bg-sunken text-ink-2 border-line',
  info: 'bg-sky-50 text-sky-800 border-sky-200 dark:bg-sky-950/50 dark:text-sky-200 dark:border-sky-900',
  good: 'bg-emerald-50 text-emerald-800 border-emerald-200 dark:bg-emerald-950/50 dark:text-emerald-200 dark:border-emerald-900',
  warn: 'bg-amber-50 text-amber-900 border-amber-200 dark:bg-amber-950/50 dark:text-amber-200 dark:border-amber-900',
  danger: 'bg-accent-soft text-accent-deep border-accent-line',
} as const;

export type StatusTone = keyof typeof STATUS;

/**
 * Measured WCAG contrast for each series colour on each page ground (T111.5).
 *
 * Computed in Python and exported, so the numbers a page shows and the rule the
 * chart layer applies come from one measurement rather than two opinions.
 */
export const PALETTE_CONTRAST = theme.contrast;

/**
 * True when a fill of this colour is too close to the page ground to be seen
 * without a stroke.
 *
 * Okabe-Ito guarantees the eight hues stay *distinguishable from each other*
 * under the common colour-vision deficiencies. It says nothing about luminance
 * contrast against a white page, and four of the eight fail 3:1 on one ground
 * or the other -- the yellow is nearly as bright as white. Dropping a colour
 * would break the fixed series order every figure depends on, so the chart
 * layer strokes the mark instead.
 */
export function needsOutlineOn(colorIndex: number, ground: 'light' | 'dark'): boolean {
  const entry = PALETTE_CONTRAST.series[colorIndex % PALETTE_CONTRAST.series.length];
  return entry?.needs_outline_on.includes(ground) ?? false;
}

/** The measured ratio, for display beside the swatch. */
export function contrastOn(colorIndex: number, ground: 'light' | 'dark'): number | null {
  const entry = PALETTE_CONTRAST.series[colorIndex % PALETTE_CONTRAST.series.length];
  if (entry === undefined) return null;
  return ground === 'light' ? entry.on_light : entry.on_dark;
}
