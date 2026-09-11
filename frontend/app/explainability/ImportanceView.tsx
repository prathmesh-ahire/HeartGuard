'use client';

import { useState } from 'react';

import { cn } from '@/lib/cn';
import type { GeneratedExplainability } from '@/lib/generated/types';
import { seriesColor, SURFACE, TYPE_SCALE } from '@/lib/tokens';

type Group = GeneratedExplainability['importance'][number];

/**
 * Global importance for one model at a time (T117.2).
 *
 * Bar LENGTH comes from the numeric `importance` -- geometry, which the
 * codegen rule allows. Every number a reader can read is `importance_display`,
 * formatted in Python.
 */
export function ImportanceView({ groups }: { groups: Group[] }) {
  const [index, setIndex] = useState(0);
  const group = groups[index];
  if (group === undefined) return null;

  const largest = Math.max(...group.rows.map((row) => row.importance ?? 0), 0);
  const families = Array.from(new Set(group.rows.map((row) => row.family)));

  return (
    <div className="space-y-4">
      {groups.length > 1 ? (
        <label className={cn(TYPE_SCALE.caption, 'flex items-center gap-2')}>
          <span className={SURFACE.muted}>Model</span>
          <select
            value={index}
            onChange={(event) => setIndex(Number(event.target.value))}
            className="rounded border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-900"
          >
            {groups.map((item, position) => (
              <option key={item.task + item.model_id + item.kind} value={position}>
                {item.task} · {item.model_id} · {item.kind}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
        {group.task} task, model {group.model_id}, {group.kind} importance: the top{' '}
        {group.rows.length} of {group.n_features_ranked_display} ranked features, averaged over{' '}
        {group.n_folds} folds.
      </p>

      <ol className="space-y-1.5">
        {group.rows.map((row) => {
          const width =
            largest > 0 && row.importance !== null ? Math.max(0, row.importance / largest) : 0;
          return (
            <li key={row.feature} className="grid grid-cols-[2.5rem_14rem_1fr] items-center gap-2">
              <span className={cn(TYPE_SCALE.caption, SURFACE.subtle, 'tabular-nums')}>
                {row.rank}
              </span>
              <span className={cn(TYPE_SCALE.caption, 'truncate font-mono')} title={row.family}>
                {row.feature}
              </span>
              <span className="flex items-center gap-2">
                <span
                  aria-hidden="true"
                  className="h-3 rounded-sm"
                  style={{
                    width: String(width * 70) + '%',
                    backgroundColor: seriesColor(Math.max(0, families.indexOf(row.family))),
                  }}
                />
                <span className={cn(TYPE_SCALE.caption, 'tabular-nums')}>
                  {row.importance_display} ± {row.importance_sd_display}
                </span>
                <span className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>
                  {row.family} · positive in {row.folds_positive_display} folds
                </span>
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
