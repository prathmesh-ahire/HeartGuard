'use client';

import { useEffect, useRef, useState } from 'react';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { pipeline } from '@/lib/generated';
import { SURFACE, TYPE_SCALE, needsOutlineOn, seriesColor } from '@/lib/tokens';

/**
 * The scroll-driven walkthrough of the twelve architecture steps (T112.3).
 *
 * **The twelve steps are not written here.** They come from
 * `generated/pipeline.json`, which `src/reporting/architecture.py` emits after
 * checking that every step's module and evidence directory actually exist in
 * the repository. So a step on this page is a claim the build verified, not a
 * caption somebody typed. If a module is renamed and the list is not updated,
 * the export fails and this component never renders the stale text.
 *
 * ## Reduced motion is a different component, not a slower one
 *
 * With `prefers-reduced-motion: reduce` the whole thing renders as a plain
 * ordered list -- no pinning, no scroll-linked progress, no GSAP loaded at all.
 * That is deliberate: a scroll-driven narrative degraded to "the same thing but
 * quicker" is still a scroll-driven narrative, and a pinned section that moves
 * content while the page is stationary is one of the most reliable triggers
 * there is.
 */

export function PipelineWalkthrough({ className }: { className?: string }) {
  const reduced = useReducedMotion();
  const container = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    if (reduced) return;
    const root = container.current;
    if (root === null) return;

    let cleanup: (() => void) | undefined;

    void (async () => {
      const [{ default: gsap }, { ScrollTrigger }] = await Promise.all([
        import('gsap'),
        import('gsap/ScrollTrigger'),
      ]);
      gsap.registerPlugin(ScrollTrigger);

      const steps = gsap.utils.toArray<HTMLElement>('[data-pipeline-step]', root);
      const triggers = steps.map((step, index) =>
        ScrollTrigger.create({
          trigger: step,
          // metric-guard: allow -- scroll offsets, not measurements
          start: 'top 70%',
          // metric-guard: allow -- scroll offsets, not measurements
          end: 'bottom 30%',
          onEnter: () => setActive(index),
          onEnterBack: () => setActive(index),
        }),
      );

      const reveals = steps.map((step) =>
        gsap.fromTo(
          step,
          { opacity: 0.35, y: 18 },
          {
            opacity: 1,
            y: 0,
            duration: 0.45,
            ease: 'power2.out',
            // metric-guard: allow -- scroll offset, not a measurement
            scrollTrigger: { trigger: step, start: 'top 80%', once: true },
          },
        ),
      );

      cleanup = () => {
        triggers.forEach((trigger) => trigger.kill());
        reveals.forEach((tween) => tween.scrollTrigger?.kill());
        reveals.forEach((tween) => tween.kill());
      };
    })();

    return () => cleanup?.();
  }, [reduced]);

  const steps = pipeline.steps;

  return (
    <div ref={container} className={cn('relative', className)}>
      <p className={cn(TYPE_SCALE.caption, SURFACE.subtle, 'mb-6')}>{pipeline.note}</p>

      <div className="lg:grid lg:grid-cols-[13rem_1fr] lg:gap-10">
        <ProgressRail steps={steps} active={reduced ? null : active} />

        <ol className="space-y-4">
          {steps.map((step, index) => (
            <li
              key={step.key}
              data-pipeline-step
              aria-current={!reduced && index === active ? 'step' : undefined}
              className={cn(
                SURFACE.card,
                'p-5 transition-colors',
                !reduced && index === active ? 'ring-2 ring-offset-2 dark:ring-offset-slate-950' : '',
              )}
              style={
                !reduced && index === active
                  ? ({ '--tw-ring-color': seriesColor(index) } as React.CSSProperties)
                  : undefined
              }
            >
              <div className="flex items-baseline gap-3">
                <StepNumber index={index} number={step.index} />
                <h3 className={cn(TYPE_SCALE.h3)}>{step.title}</h3>
              </div>
              <p className={cn(TYPE_SCALE.body, SURFACE.muted, 'mt-2')}>{step.summary}</p>
              {step.rule === null ? null : (
                <p
                  className={cn(
                    TYPE_SCALE.caption,
                    'mt-3 border-l-2 pl-3 italic',
                    'border-slate-300 dark:border-slate-700',
                  )}
                >
                  {step.rule}
                </p>
              )}
              <dl className={cn(TYPE_SCALE.micro, SURFACE.subtle, 'mt-3 flex flex-wrap gap-x-6')}>
                <div className="flex gap-1">
                  <dt>module</dt>
                  <dd className="font-mono normal-case tracking-normal">{step.module}</dd>
                </div>
                <div className="flex gap-1">
                  <dt>evidence</dt>
                  <dd className="font-mono normal-case tracking-normal">{step.evidence_dir}</dd>
                </div>
              </dl>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

/**
 * The step marker. Filled with the series colour, stroked when that colour is
 * too close to the page ground to be seen -- the same `needs_outline_on` rule
 * the chart layer applies, read from the measurement rather than restated.
 */
function StepNumber({ index, number }: { index: number; number: number }) {
  const fill = seriesColor(index);
  const outlined = needsOutlineOn(index, 'light');
  return (
    <span
      aria-hidden="true"
      className={cn(
        'inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full',
        'text-xs font-semibold tabular-nums text-white',
        outlined ? 'text-slate-900 ring-1 ring-slate-900/60 dark:ring-white/60' : '',
      )}
      style={{ backgroundColor: fill }}
    >
      {number}
    </span>
  );
}

function ProgressRail({
  steps,
  active,
}: {
  steps: typeof pipeline.steps;
  active: number | null;
}) {
  return (
    <nav aria-label="Pipeline steps" className="mb-8 hidden lg:sticky lg:top-24 lg:mb-0 lg:block">
      <ol className="space-y-1">
        {steps.map((step, index) => (
          <li key={step.key}>
            <span
              className={cn(
                TYPE_SCALE.caption,
                'flex items-center gap-2 rounded px-2 py-1',
                active === index ? 'bg-slate-100 font-medium dark:bg-slate-800' : SURFACE.subtle,
              )}
            >
              <span
                aria-hidden="true"
                className="h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ backgroundColor: seriesColor(index) }}
              />
              {step.title}
            </span>
          </li>
        ))}
      </ol>
    </nav>
  );
}
