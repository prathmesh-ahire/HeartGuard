import type { GeneratedColumn, GeneratedFigure, GeneratedTable } from '@/lib/generated';

/**
 * The shapes a chart accepts, and the one rule they all enforce.
 *
 * A chart is handed a **generated table or figure**, never loose numbers. That
 * is not ceremony: it means the chart cannot be fed a literal, because there is
 * no parameter that would take one. `numericColumn` returns the `values` array
 * for geometry and `displayColumn` returns the pre-formatted strings for any
 * text -- the two never cross.
 */

export type Source = GeneratedTable | GeneratedFigure;

export function column(source: Source | undefined, name: string): GeneratedColumn | undefined {
  return source?.columns.find((candidate) => candidate.name === name);
}

/** Numbers, for positioning marks. Never rendered as text. */
export function numericColumn(source: Source | undefined, name: string): (number | null)[] {
  return column(source, name)?.values ?? [];
}

/** Pre-formatted strings, for axis labels, tooltips and tables. */
export function displayColumn(source: Source | undefined, name: string): string[] {
  return column(source, name)?.display ?? [];
}

/** True when the source is present and has at least one row. */
export function hasRows(source: Source | undefined): boolean {
  return source !== undefined && source.n_rows > 0 && source.columns.length > 0;
}

/** Pairs `(x, y)` with any row where either is null dropped, not zeroed. */
export function points(xs: (number | null)[], ys: (number | null)[]): [number, number][] {
  const out: [number, number][] = [];
  for (let index = 0; index < Math.min(xs.length, ys.length); index += 1) {
    const x = xs[index];
    const y = ys[index];
    // A missing value is missing. Substituting zero would draw a mark at the
    // origin and the reader would have no way to tell it from a real one.
    if (typeof x === 'number' && typeof y === 'number') out.push([x, y]);
  }
  return out;
}
