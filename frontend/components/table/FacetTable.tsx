'use client';

import { useMemo, useState } from 'react';

import { cn } from '@/lib/cn';
import { ResultsTable } from '@/components/table/ResultsTable';
import type { GeneratedColumn, GeneratedTable } from '@/lib/generated/types';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * A results table with a drop-down filter per chosen column (T117.1).
 *
 * Filtering is row SELECTION and nothing else: the rows kept are the exported
 * rows, and every cell is still the display string Python formatted. No value
 * is recomputed, re-aggregated or re-rounded when a filter changes -- which is
 * why the filter narrows the committed table rather than summarising it.
 *
 * The first render is the whole table, so the static HTML (and the displayed-
 * value audit that reads it) sees every row.
 */

const ALL = '__all__';

export function FacetTable({
  table,
  facets,
  caption,
  className,
}: {
  table?: GeneratedTable;
  /** Column names offered as filters, in the order shown. */
  facets: readonly string[];
  caption?: string;
  className?: string;
}) {
  const [chosen, setChosen] = useState<Record<string, string>>({});

  const facetColumns = useMemo<GeneratedColumn[]>(() => {
    if (table === undefined) return [];
    return facets
      .map((name) => table.columns.find((column) => column.name === name))
      .filter((column): column is GeneratedColumn => column !== undefined);
  }, [table, facets]);

  const subset = useMemo<GeneratedTable | undefined>(() => {
    if (table === undefined) return undefined;
    const keep: number[] = [];
    for (let row = 0; row < table.n_rows; row += 1) {
      const matches = facetColumns.every((column) => {
        const wanted = chosen[column.name];
        return wanted === undefined || wanted === ALL || column.display[row] === wanted;
      });
      if (matches) keep.push(row);
    }
    if (keep.length === table.n_rows) return table;
    return {
      ...table,
      n_rows: keep.length,
      columns: table.columns.map((column) => ({
        ...column,
        display: keep.map((row) => column.display[row] ?? ''),
        values: column.values === null ? null : keep.map((row) => column.values?.[row] ?? null),
      })),
    };
  }, [table, facetColumns, chosen]);

  if (table === undefined) return <ResultsTable className={className} />;

  const active = Object.values(chosen).some((value) => value !== ALL);

  return (
    <div className={cn('space-y-3', className)}>
      <div role="group" aria-label={'Filter ' + table.id} className="flex flex-wrap items-end gap-3">
        {facetColumns.map((column) => {
          const options = Array.from(new Set(column.display));
          return (
            <label key={column.name} className={cn(TYPE_SCALE.caption, 'flex flex-col gap-1')}>
              <span className={SURFACE.muted}>{column.header ?? column.name}</span>
              <select
                value={chosen[column.name] ?? ALL}
                onChange={(event) =>
                  setChosen((previous) => ({ ...previous, [column.name]: event.target.value }))
                }
                className={cn(
                  'rounded border px-2 py-1',
                  'border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900',
                )}
              >
                <option value={ALL}>all</option>
                {options.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          );
        })}
        {active ? (
          <button
            type="button"
            onClick={() => setChosen({})}
            className={cn(
              TYPE_SCALE.caption,
              'rounded border px-2.5 py-1',
              'border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800',
            )}
          >
            Clear filters
          </button>
        ) : null}
      </div>
      <ResultsTable table={subset} caption={caption} />
    </div>
  );
}
