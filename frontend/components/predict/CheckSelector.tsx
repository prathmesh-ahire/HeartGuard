'use client';

import { cn } from '@/lib/cn';
import { TYPE_SCALE } from '@/lib/tokens';

/**
 * What to check (T130.4).
 *
 * Three checks in plain words, each with one line saying what it does. Two of
 * them hold two tasks -- PASCAL A and PASCAL B under "Sound type", murmur and
 * outcome under "Murmur and outcome" -- and the second row picks between them.
 * They stay separate tasks with separate models (research rule 4): the check is
 * a way to find a task, never a merged label space, so exactly one task is
 * ever selected and sent.
 */

export interface AnalyseTask {
  task: string;
  label: string;
}

export interface AnalyseCheck {
  id: string;
  label: string;
  /** One plain line. */
  description: string;
  tasks: readonly AnalyseTask[];
}

export function CheckSelector({
  checks,
  task,
  onChange,
  disabled = false,
  className,
}: {
  checks: readonly AnalyseCheck[];
  task: string;
  onChange: (task: string) => void;
  disabled?: boolean;
  className?: string;
}) {
  const active = checks.find((check) => check.tasks.some((entry) => entry.task === task)) ?? checks[0];

  return (
    <div className={className}>
      <div role="group" aria-label="What to check" className="grid gap-2 sm:grid-cols-3">
        {checks.map((check) => {
          const selected = check.id === active?.id;
          return (
            <button
              key={check.id}
              type="button"
              aria-pressed={selected}
              disabled={disabled}
              onClick={() => {
                const first = check.tasks[0];
                if (!selected && first !== undefined) onChange(first.task);
              }}
              className={cn(
                'rounded-xl border p-3 text-left transition-colors disabled:opacity-60',
                selected
                  ? 'border-accent bg-accent-soft shadow-panel'
                  : 'border-line bg-panel hover:border-accent-line',
              )}
            >
              <span className={cn(TYPE_SCALE.body, 'block font-medium text-ink')}>{check.label}</span>
              <span className={cn(TYPE_SCALE.caption, 'mt-1 block text-ink-2')}>{check.description}</span>
            </button>
          );
        })}
      </div>

      {active !== undefined && active.tasks.length > 1 ? (
        <div role="group" aria-label="Which set of categories" className="mt-3 flex flex-wrap items-center gap-2">
          <span className={cn(TYPE_SCALE.caption, 'text-ink-3')}>Categories from</span>
          {active.tasks.map((entry) => (
            <button
              key={entry.task}
              type="button"
              aria-pressed={entry.task === task}
              disabled={disabled}
              onClick={() => onChange(entry.task)}
              className={cn(
                'rounded-lg border px-3 py-1.5 font-mono text-label-md uppercase transition-colors',
                entry.task === task
                  ? 'border-accent-strong bg-accent text-on-accent'
                  : 'border-line bg-panel text-ink-2 hover:border-accent-line hover:text-accent-strong',
              )}
            >
              {entry.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
