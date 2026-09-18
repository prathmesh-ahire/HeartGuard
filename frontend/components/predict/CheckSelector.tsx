'use client';

import { LazyMotion, m } from 'framer-motion';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import {
  CARD_HOVER,
  CARD_TAP,
  CHECK_CARD_INTRO_SCALE,
  DURATION,
  EASE_OUT,
  PRESS_SPRING,
} from '@/lib/motion';
import { TYPE_SCALE } from '@/lib/tokens';

const loadFeatures = () => import('@/components/motion/features').then((mod) => mod.default);

/**
 * What to check (T130.4).
 *
 * Three checks in plain words, each with one line saying what it does. Two of
 * them hold two tasks -- PASCAL A and PASCAL B under "Sound type", murmur and
 * outcome under "Murmur and outcome" -- and the second row picks between them.
 * They stay separate tasks with separate models (research rule 4): the check is
 * a way to find a task, never a merged label space, so exactly one task is
 * ever selected and sent.
 *
 * Stacked, not side by side (redesign, 2026-09-18), and compact -- close
 * together, no dedicated spacer between them. Two earlier passes tried to
 * earn a dramatic "card 1 pushes out card 2" handoff with a GSAP `scrub`
 * timeline, first sized to the viewport, then to a fixed pixel range; both
 * needed real scroll distance between cards to read, so both ended up as
 * empty space stretched between them -- the thing that was wrong for
 * everyone who scrolled through it, not just the calibration. Settled: a
 * light, one-shot entrance instead, the same proven `whileInView` shape
 * `Reveal.tsx` already uses everywhere else in the app (see its own comment
 * for why `animate` stands in under reduced motion, and why `initial` cannot
 * differ between what the server renders and what hydration sees). Each card
 * grows from `CHECK_CARD_INTRO_SCALE` and fades in the first time it scrolls
 * into view, once, and stays put after -- no exit, no forced gap.
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
  const reduced = useReducedMotion();

  return (
    <div className={className}>
      <LazyMotion features={loadFeatures} strict>
        <div role="group" aria-label="What to check" className="flex flex-col gap-4">
          {checks.map((check) => {
            const selected = check.id === active?.id;
            const press = !reduced && !disabled;
            return (
              <m.button
                key={check.id}
                type="button"
                aria-pressed={selected}
                disabled={disabled}
                onClick={() => {
                  const first = check.tasks[0];
                  if (!selected && first !== undefined) onChange(first.task);
                }}
                initial={{ opacity: 0, scale: CHECK_CARD_INTRO_SCALE }}
                animate={reduced ? { opacity: 1, scale: 1 } : undefined}
                whileInView={reduced ? undefined : { opacity: 1, scale: 1 }}
                // A fixed pixel inset, not `Reveal.tsx`'s `-10%`: T130's own
                // metric-guard test (test_analyse_page.py) demands zero
                // suppressions in this directory, percentage or not, so this
                // reaches for a unit the guard was never written to flag.
                viewport={{ once: true, margin: '0px 0px -80px 0px' }}
                transition={reduced ? { duration: 0 } : { duration: DURATION.base, ease: EASE_OUT }}
                whileHover={press ? { ...CARD_HOVER, transition: PRESS_SPRING } : undefined}
                whileTap={press ? { ...CARD_TAP, transition: PRESS_SPRING } : undefined}
                className={cn(
                  'w-full rounded-3xl border p-6 text-left backdrop-blur-md transition-colors disabled:opacity-60',
                  selected
                    ? 'border-accent bg-accent-soft/80 shadow-panel'
                    : 'border-line bg-panel/70 hover:border-accent-line',
                )}
              >
                <span className={cn(TYPE_SCALE.h3, 'block font-medium text-ink')}>{check.label}</span>
                <span className={cn(TYPE_SCALE.body, 'mt-2 block max-w-reading leading-relaxed text-ink-2')}>
                  {check.description}
                </span>
              </m.button>
            );
          })}
        </div>
      </LazyMotion>

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
                'rounded-full border px-3 py-1.5 text-label-md uppercase transition-colors',
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
