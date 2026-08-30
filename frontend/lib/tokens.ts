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
  micro: 'text-[10px] leading-4 tracking-widest uppercase',
  caption: 'text-xs leading-5',
  body: 'text-sm leading-6',
  lead: 'text-base leading-7',
  h3: 'text-lg font-semibold leading-7 tracking-tight',
  h2: 'text-xl font-semibold leading-8 tracking-tight',
  h1: 'text-2xl font-semibold leading-9 tracking-tight',
  display: 'text-3xl font-semibold leading-10 tracking-tight',
  /** Metrics are tabular so digits line up column to column. */
  stat: 'text-3xl font-semibold tabular-nums leading-none tracking-tight',
} as const;

/** A 4px base step. Named so a card and a section cannot drift apart. */
export const SPACING = {
  tight: 'gap-2',
  normal: 'gap-4',
  section: 'gap-8',
  page: 'space-y-10',
} as const;

export const SURFACE = {
  card: 'rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900',
  glass:
    'rounded-lg border border-slate-200/70 bg-white/70 backdrop-blur-md dark:border-slate-700/60 dark:bg-slate-900/60',
  sunken: 'rounded-lg border border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-950',
  muted: 'text-slate-600 dark:text-slate-400',
  subtle: 'text-slate-500 dark:text-slate-500',
} as const;

/**
 * Status colours. Deliberately NOT the series palette: a badge saying
 * "abnormal" must not borrow the colour a chart is using for series 2 in the
 * same viewport, or the reader will read a legend that is not there.
 */
export const STATUS = {
  neutral:
    'bg-slate-100 text-slate-700 border-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700',
  info: 'bg-sky-50 text-sky-800 border-sky-300 dark:bg-sky-950/50 dark:text-sky-200 dark:border-sky-800',
  good: 'bg-emerald-50 text-emerald-800 border-emerald-300 dark:bg-emerald-950/50 dark:text-emerald-200 dark:border-emerald-800',
  warn: 'bg-amber-50 text-amber-900 border-amber-300 dark:bg-amber-950/50 dark:text-amber-200 dark:border-amber-800',
  danger:
    'bg-rose-50 text-rose-800 border-rose-300 dark:bg-rose-950/50 dark:text-rose-200 dark:border-rose-800',
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
