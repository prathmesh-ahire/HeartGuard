'use client';

import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table';
import { useMemo, useState } from 'react';

import { cn } from '@/lib/cn';
import type { GeneratedTable } from '@/lib/generated';
import { EmptyState } from '@/components/ui/States';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * The results table (T113.3): sort, filter, CSV download.
 *
 * ## It renders `display`, and only `display`
 *
 * Every cell is a string Python already formatted under the T85.6 rounding
 * rules. The numeric `values` array is used for **sorting only** -- so that
 * "0.9" sorts below "0.85" numerically rather than lexically -- and never
 * reaches the DOM. That is the codegen boundary applied to a table: the client
 * may order numbers, it may not format or compute them.
 *
 * ## The CSV is the displayed text, not a re-serialisation
 *
 * Downloading writes exactly the strings on screen. Re-serialising `values`
 * would hand the reader a file whose numbers differ in the last decimal from
 * the table they were looking at, and they would have no way to know which one
 * the thesis used.
 */

interface Row {
  index: number;
  [key: string]: string | number;
}

export function ResultsTable({
  table: source,
  className,
  caption,
  initialSort,
}: {
  table?: GeneratedTable;
  className?: string;
  caption?: string;
  initialSort?: string;
}) {
  const [sorting, setSorting] = useState<SortingState>(
    initialSort === undefined ? [] : [{ id: initialSort, desc: true }],
  );
  const [filter, setFilter] = useState('');

  const rows = useMemo<Row[]>(() => {
    if (source === undefined) return [];
    return Array.from({ length: source.n_rows }, (_, index) => {
      const row: Row = { index };
      for (const column of source.columns) {
        row[column.name] = column.display[index] ?? '';
        const numeric = column.values?.[index];
        // A parallel sort key, never rendered. See the docstring.
        if (typeof numeric === 'number') row['__sort__' + column.name] = numeric;
      }
      return row;
    });
  }, [source]);

  const columns = useMemo<ColumnDef<Row>[]>(() => {
    if (source === undefined) return [];
    return source.columns.map((column) => ({
      id: column.name,
      header: column.header ?? column.name,
      accessorFn: (row: Row) => row[column.name],
      sortingFn: (a, b) => {
        const left = a.original['__sort__' + column.name];
        const right = b.original['__sort__' + column.name];
        if (typeof left === 'number' && typeof right === 'number') return left - right;
        return String(a.original[column.name]).localeCompare(String(b.original[column.name]));
      },
    }));
  }, [source]);

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting, globalFilter: filter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  if (source === undefined) {
    return (
      <EmptyState
        className={className}
        title="This table has not been exported"
        description="Nothing is hidden — the run that produces it has not been made yet."
      />
    );
  }

  const visible = table.getRowModel().rows;

  return (
    <div className={cn('space-y-3', className)}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <label className={cn(TYPE_SCALE.caption, 'flex items-center gap-2')}>
          <span className={SURFACE.muted}>Filter</span>
          <input
            type="search"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder={'search ' + source.id}
            className={cn(
              'rounded border px-2 py-1',
              'border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900',
            )}
          />
        </label>
        <DownloadCsv source={source} rows={visible.map((row) => row.original)} />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <caption className={cn(TYPE_SCALE.caption, SURFACE.muted, 'caption-bottom pt-2')}>
            {caption ?? source.caption}
          </caption>
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => {
                  const direction = header.column.getIsSorted();
                  return (
                    <th
                      key={header.id}
                      scope="col"
                      aria-sort={
                        direction === 'asc'
                          ? 'ascending'
                          : direction === 'desc'
                            ? 'descending'
                            : 'none'
                      }
                      className={cn(
                        TYPE_SCALE.caption,
                        'border-b border-slate-300 px-3 py-2 font-semibold dark:border-slate-700',
                      )}
                    >
                      <button
                        type="button"
                        onClick={header.column.getToggleSortingHandler()}
                        className="inline-flex items-center gap-1 hover:underline"
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        <span aria-hidden="true" className={SURFACE.subtle}>
                          {direction === 'asc' ? '▲' : direction === 'desc' ? '▼' : '↕'}
                        </span>
                      </button>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {visible.map((row) => (
              <tr key={row.id} className="border-b border-slate-200 dark:border-slate-800">
                {row.getVisibleCells().map((cell) => (
                  <td
                    key={cell.id}
                    className={cn(TYPE_SCALE.caption, 'px-3 py-1.5 tabular-nums')}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext()) ??
                      String(cell.getValue())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {visible.length === 0 ? (
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
          No rows match that filter. The table itself has {source.n_rows} rows.
        </p>
      ) : (
        <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>
          {visible.length} of {source.n_rows} rows · source {source.source_csv}
        </p>
      )}
    </div>
  );
}

function DownloadCsv({ source, rows }: { source: GeneratedTable; rows: Row[] }) {
  const download = (): void => {
    const header = source.columns.map((column) => column.header ?? column.name);
    const body = rows.map((row) => source.columns.map((column) => String(row[column.name] ?? '')));
    const csv = [header, ...body]
      .map((line) => line.map((cell) => '"' + cell.replaceAll('"', '""') + '"').join(','))
      .join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = source.id + '_as_displayed.csv';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={download}
        className={cn(
          TYPE_SCALE.caption,
          'rounded border px-2.5 py-1',
          'border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800',
        )}
      >
        Download CSV as displayed
      </button>
      <span className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>
        filtered rows, rounded exactly as shown
      </span>
    </div>
  );
}
