'use client';

import { useMemo, useState } from 'react';

import { EmptyState } from '@/components/ui/States';
import { records } from '@/lib/generated/records';

/**
 * Record-level filtering and drill-down (T114.3).
 *
 * ## What this component is allowed to do
 *
 * Select rows and show them. That is the whole contract. It reads the columnar
 * record index from `generated/records.json`, where every cell already exists as
 * a **pre-formatted string** in `display`; the numeric `values` array is used
 * only to compare a duration against a slider bound. Nothing is rounded here,
 * nothing is summed, and no rate, share or average is derived — a "3.4% of
 * records" computed in a browser is exactly the kind of number that ends up in a
 * screenshot with nothing behind it.
 *
 * The one number this component produces is **how many rows the filter matched**,
 * which is a property of the reader's own selection and not a result. It is
 * labelled as such.
 *
 * ## Why the whole corpus is here
 *
 * All 7,536 records ship, not a sample. A filter over a truncated table shows a
 * count the reader will believe. Columnar JSON is what makes that affordable:
 * the same data as rows-of-objects is fifteen times larger.
 */

const PAGE_SIZE = 50;

type Filters = Record<string, string>;

export function DatasetExplorer() {
  const [filters, setFilters] = useState<Filters>({});
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(0);

  const byName = useMemo(
    () => new Map(records.columns.map((column) => [column.name, column])),
    [],
  );

  const matching = useMemo(() => {
    const active = Object.entries(filters).filter(([, value]) => value !== '');
    const needle = query.trim().toLowerCase();
    const uids = byName.get('record_uid')?.display ?? [];
    const subjects = byName.get('subject_id')?.display ?? [];

    const kept: number[] = [];
    for (let index = 0; index < records.n_records; index += 1) {
      let ok = true;
      for (const [name, value] of active) {
        if (byName.get(name)?.display[index] !== value) {
          ok = false;
          break;
        }
      }
      if (ok && needle !== '') {
        const uid = uids[index]?.toLowerCase() ?? '';
        const subject = subjects[index]?.toLowerCase() ?? '';
        ok = uid.includes(needle) || subject.includes(needle);
      }
      if (ok) kept.push(index);
    }
    return kept;
  }, [byName, filters, query]);

  const shown = matching.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);
  const lastPage = Math.max(0, Math.ceil(matching.length / PAGE_SIZE) - 1);

  const tableColumns = records.columns.filter(
    (column) => column.name !== 'dataset_name' && column.name !== 'subject_derived',
  );

  function setFilter(name: string, value: string) {
    setPage(0);
    setFilters((current) => ({ ...current, [name]: value }));
  }

  return (
    <div>
      {/* ---------------------------------------------------------------- */}
      <div className="flex flex-wrap items-end gap-3 rounded-lg border border-slate-200 p-4 dark:border-slate-800">
        <label className="flex flex-col gap-1 text-xs">
          <span className="uppercase tracking-widest text-slate-500">Record or subject</span>
          <input
            type="search"
            value={query}
            onChange={(event) => {
              setPage(0);
              setQuery(event.target.value);
            }}
            placeholder="a0005, 85197&hellip;"
            className="w-56 rounded border border-slate-300 bg-white px-2 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900"
          />
        </label>

        {records.facets.map((facet) => (
          <label key={facet.name} className="flex flex-col gap-1 text-xs">
            <span className="uppercase tracking-widest text-slate-500">{facet.label}</span>
            <select
              value={filters[facet.name] ?? ''}
              onChange={(event) => setFilter(facet.name, event.target.value)}
              className="rounded border border-slate-300 bg-white px-2 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900"
            >
              <option value="">any</option>
              {facet.values.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        ))}

        <button
          type="button"
          onClick={() => {
            setFilters({});
            setQuery('');
            setPage(0);
          }}
          className="rounded border border-slate-300 px-3 py-1.5 text-sm dark:border-slate-700"
        >
          Reset
        </button>
      </div>

      {/* ---------------------------------------------------------------- */}
      <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
        <span className="tabular-nums font-medium text-slate-800 dark:text-slate-200">
          {matching.length.toLocaleString('en-US')}
        </span>{' '}
        of {records.n_records.toLocaleString('en-US')} audited recordings match this
        selection. This is a count of your own filter, not a reported result.
      </p>

      {/* ---------------------------------------------------------------- */}
      {shown.length === 0 ? (
        <EmptyState
          className="mt-4"
          title="No recording matches this selection"
          description="Loosen a filter, or reset. Nothing is hidden: the whole audited corpus is loaded."
        />
      ) : (
        <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase tracking-widest text-slate-500 dark:bg-slate-900/60">
              <tr>
                {tableColumns.map((column) => (
                  <th key={column.name} scope="col" className="whitespace-nowrap px-3 py-2">
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {shown.map((index) => (
                <tr
                  key={index}
                  className="border-t border-slate-100 odd:bg-white even:bg-slate-50/60 dark:border-slate-800 dark:odd:bg-transparent dark:even:bg-slate-900/40"
                >
                  {tableColumns.map((column) => (
                    <td
                      key={column.name}
                      className={
                        column.values === null
                          ? 'whitespace-nowrap px-3 py-1.5'
                          : 'whitespace-nowrap px-3 py-1.5 tabular-nums'
                      }
                    >
                      {column.display[index]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ---------------------------------------------------------------- */}
      {lastPage > 0 ? (
        <div className="mt-3 flex items-center gap-3 text-sm">
          <button
            type="button"
            disabled={page === 0}
            onClick={() => setPage((current) => Math.max(0, current - 1))}
            className="rounded border border-slate-300 px-3 py-1 disabled:opacity-40 dark:border-slate-700"
          >
            Previous
          </button>
          <span className="tabular-nums text-slate-600 dark:text-slate-400">
            Page {page + 1} of {lastPage + 1}
          </span>
          <button
            type="button"
            disabled={page >= lastPage}
            onClick={() => setPage((current) => Math.min(lastPage, current + 1))}
            className="rounded border border-slate-300 px-3 py-1 disabled:opacity-40 dark:border-slate-700"
          >
            Next
          </button>
        </div>
      ) : null}

      <p className="mt-4 text-xs text-slate-500 dark:text-slate-500">
        Source: {records.source}. Every cell is a string formatted in Python; this view
        selects rows and renders them.
      </p>
    </div>
  );
}
