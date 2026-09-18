'use client';

import { useMemo, useState } from 'react';

import { EmptyState } from '@/components/ui/States';
import { features } from '@/lib/generated/features';

/**
 * The 138-feature registry and one record's vector (T115.1).
 *
 * ## The order is the data, and this component never sorts it away
 *
 * The 138-vector's column order is a literal in
 * `src/feature_extraction/registry.py`, is fingerprinted, and two runs that
 * disagree on that fingerprint are not comparable. note.md records T05 coming
 * out alphabetical once: every count was correct and the ordering told a reader
 * the order was arbitrary. So this table is always in `index` order. Filtering
 * hides rows; it never reorders them, and the index column stays visible so a
 * filtered view is still locatable in the full vector.
 *
 * ## What is displayed and what is only compared
 *
 * `display` renders. `value` and `abs_cohens_d` exist so a filter can compare
 * and a bar can be positioned. Nothing here rounds, sums or ranks — the
 * separation ranking was computed in Python and arrives already ordered.
 */
export function FeatureExplorer() {
  const [family, setFamily] = useState('');
  const [query, setQuery] = useState('');
  const [showVector, setShowVector] = useState(true);

  const vector = features.example_vector;
  const valueByName = useMemo(() => {
    const map = new Map<string, string>();
    for (const item of vector.values) map.set(item.name, item.display);
    return map;
  }, [vector.values]);

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return features.features.filter((item) => {
      if (family !== '' && item.family !== family) return false;
      if (needle === '') return true;
      return (
        item.name.toLowerCase().includes(needle) ||
        item.description.toLowerCase().includes(needle)
      );
    });
  }, [family, query]);

  return (
    <div>
      <div className="flex flex-wrap items-end gap-3 rounded-lg border border-line p-4">
        <label className="flex flex-col gap-1.5">
          <span className="label-micro">Family</span>
          <select
            value={family}
            onChange={(event) => setFamily(event.target.value)}
            className="rounded border border-line bg-sunken px-2 py-1.5 text-label-md text-ink focus:border-accent focus:outline-none"
          >
            <option value="">all families</option>
            {features.families.map((item) => (
              <option key={item.family} value={item.family}>
                {item.family}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="label-micro">Name or description</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="mfcc, entropy, centroid&hellip;"
            className="w-64 rounded border border-line bg-panel px-2 py-1.5 text-sm"
          />
        </label>

        {vector.available ? (
          <label className="flex items-center gap-2 text-body-md text-ink-2">
            <input
              type="checkbox"
              checked={showVector}
              onChange={(event) => setShowVector(event.target.checked)}
              className="h-4 w-4 accent-accent"
            />
            Show one record&rsquo;s values
          </label>
        ) : null}
      </div>

      <p className="mt-3 text-sm text-ink-2">
        Showing{' '}
        <span className="tabular-nums font-medium text-ink">
          {rows.length}
        </span>{' '}
        of {features.n_features} features, in registry order. This is a count of your
        own filter, not a reported result.
      </p>

      {rows.length === 0 ? (
        <EmptyState
          className="mt-4"
          title="No feature matches this filter"
          description="Loosen the filter or reset it. The whole registry is loaded."
        />
      ) : (
        <div className="mt-4 max-h-[36rem] overflow-auto rounded-lg border border-line">
          <table className="min-w-full text-left text-sm">
            <thead className="sticky top-0 border-b-2 border-line bg-sunken text-label-sm uppercase text-ink-3">
              <tr>
                <th scope="col" className="px-3 py-2">
                  #
                </th>
                <th scope="col" className="px-3 py-2">
                  Feature
                </th>
                <th scope="col" className="px-3 py-2">
                  Family
                </th>
                <th scope="col" className="px-3 py-2">
                  Extractor
                </th>
                <th scope="col" className="px-3 py-2">
                  |Cohen&rsquo;s d|
                </th>
                {showVector && vector.available ? (
                  <th scope="col" className="px-3 py-2">
                    Value
                  </th>
                ) : null}
              </tr>
            </thead>
            <tbody>
              {rows.map((item) => (
                <tr
                  key={item.index}
                  className="border-t border-line"
                >
                  <td className="px-3 py-1.5 tabular-nums text-ink-3">{item.index}</td>
                  <td className="px-3 py-1.5 font-mono">{item.name}</td>
                  <td className="px-3 py-1.5">{item.family}</td>
                  <td className="px-3 py-1.5 text-ink-3">{item.extractor}</td>
                  <td className="px-3 py-1.5 tabular-nums">{item.abs_cohens_d_display}</td>
                  {showVector && vector.available ? (
                    <td className="px-3 py-1.5 tabular-nums">
                      {valueByName.get(item.name) ?? 'n/a'}
                    </td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showVector && vector.available ? (
        <p className="mt-3 text-xs text-ink-3">
          Values are for record <span className="font-mono">{vector.record_uid}</span>.{' '}
          {vector.note}
        </p>
      ) : null}

      <p className="mt-2 text-xs text-ink-3">
        Cohen&rsquo;s d is the class separation between normal and abnormal recordings.
        It describes one feature in
        isolation and is not a model result: a feature with a small d can still matter
        inside a model, and a large one is not evidence on its own.
      </p>
    </div>
  );
}
