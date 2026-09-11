'use client';

import { useEffect, useState } from 'react';

import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';

/**
 * A search box over the server-rendered evidence browser (T117.4).
 *
 * The browser's rows are rendered in the static HTML, so every link exists for
 * a crawler and for the audit without JavaScript. This island only toggles
 * `hidden` on groups whose text does not match -- it holds no copy of the
 * evidence index, which keeps several hundred rows out of the page's bundle.
 * It also opens the group named in the URL hash, so a "evidence" link from a
 * table lands on that table's group already expanded.
 */
export function EvidenceFilter({ total }: { total: number }) {
  const [query, setQuery] = useState('');
  const [shown, setShown] = useState(total);

  useEffect(() => {
    const target = window.location.hash.slice(1);
    if (target.startsWith('evidence-')) {
      const group = document.getElementById(target);
      if (group instanceof HTMLDetailsElement) {
        group.open = true;
        group.scrollIntoView({ block: 'start' });
      }
    }
  }, []);

  useEffect(() => {
    const needle = query.trim().toLowerCase();
    let count = 0;
    document.querySelectorAll<HTMLElement>('[data-evidence-group]').forEach((group) => {
      const match = needle === '' || (group.dataset.evidenceText ?? '').includes(needle);
      group.hidden = !match;
      if (match) count += Number(group.dataset.evidenceCount ?? '0');
      if (match && needle !== '' && group instanceof HTMLDetailsElement) group.open = true;
    });
    setShown(count);
  }, [query]);

  return (
    <label className={cn(TYPE_SCALE.caption, 'flex flex-wrap items-center gap-2')}>
      <span className={SURFACE.muted}>Search the evidence index</span>
      <input
        type="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="table id, figure id, file name…"
        className="w-72 rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-900"
      />
      <span className={SURFACE.subtle}>
        {shown} of {total} entries
      </span>
    </label>
  );
}
