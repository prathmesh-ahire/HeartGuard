'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { AnimatePresence, LazyMotion, m } from 'framer-motion';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import {
  CARD_SCROLL_REVEAL_BLUR_PX,
  CARD_SCROLL_REVEAL_OPACITY,
  CARD_SCROLL_REVEAL_SCALE,
  LIFT_SPRING,
  STEP1_CARD_BLUR_SCROLL_PX,
  STEP1_CARD_EXIT_BLUR_DELAY_PX,
  STEP1_CARD_EXIT_BLUR_PX,
  STEP1_CARD_SCROLL_PX,
} from '@/lib/motion';
import { useCardScrollReveal } from '@/lib/useCardScrollReveal';
import { GlassCard } from '@/components/ui/GlassCard';
import { Button } from '@/components/ui/Button';
import { EmptyState, ErrorState } from '@/components/ui/States';
import { AnalyseProgress } from '@/components/predict/AnalyseProgress';
import { FileUpload, type UploadPhase } from '@/components/ui/FileUpload';
import { MicRecorder } from '@/components/audio/MicRecorder';
import { WaveformPlayer } from '@/components/audio/WaveformPlayer';
import { BatchPanel } from '@/components/predict/BatchPanel';
import { CheckSelector, type AnalyseCheck } from '@/components/predict/CheckSelector';
import { ResultCard } from '@/components/predict/ResultCard';
import { TYPE_SCALE } from '@/lib/tokens';
import { prediction } from '@/lib/generated/prediction';
import { rememberPrediction } from '@/lib/lastPrediction';
import type { GeneratedSample, GeneratedTaskSpec } from '@/lib/generated/types';
import {
  ApiError,
  predictFile,
  predictSample,
  sampleAudioUrl,
  sampleReport,
  samples as fetchSamples,
  saveDocument,
  tasks as fetchTasks,
  type PredictResult,
  type SampleStatus,
  type TaskStatus,
} from '@/lib/api';

/**
 * The Analyse workspace (T116.1, rebuilt in T130.1-T130.6).
 *
 * Choose a check, add a recording -- dropped, recorded or a built-in sample --
 * hear and scrub it, analyse it, and read the result beside it. One primary
 * action: "Analyse recording".
 *
 * ## One path for every recording
 *
 * A dropped file and a microphone recording both arrive through `acceptFile`,
 * as a validated WAV `File`. From there they are the same thing: the same
 * preview, the same `POST /predict`, the same History row, the same report.
 * Only a built-in sample differs, because its audio is on the server already.
 *
 * ## Availability is a runtime question, asked at runtime
 *
 * `generated/prediction.json` records whether each task had a saved model when
 * the export ran. That goes stale the moment a model is saved, so the panel
 * re-reads `GET /tasks` on mount and prefers the live answer.
 *
 * ## Nothing is rounded here
 *
 * Every number rendered downstream comes from `result.display.*`, formatted in
 * Python.
 *
 * ## The Step 1 card's own scroll intro and exit
 *
 * Small, faint and blurred at the top of the page, resolving to its normal
 * size/opacity/sharpness over `STEP1_CARD_SCROLL_PX` of scroll -- see the
 * constant's own comment in `lib/motion.ts`. This wraps just the Step 1
 * `GlassCard` in a plain ref'd `div`, rather than reaching into `GlassCard`
 * itself, so the shared card primitive (used everywhere else in the app)
 * stays untouched.
 *
 * The exit mirrors it as the card scrolls past the top of the viewport --
 * but as one combined timeline per property (scale+opacity, and separately
 * filter) spanning intro, a held resting state and exit together, not as
 * separate intro/exit tweens. Two tweens both writing `filter` fought each
 * other wherever their ranges met, which on this page is close: the card
 * sits high enough that intro-finished and exit-starting are not far apart
 * in scroll terms. One timeline per property means one owner at every
 * scroll position. Both timelines share one body-anchored `ScrollTrigger`
 * range, `elTop`/`elBottom` (the card's own absolute position, read once at
 * mount) standing in for the card-relative positions a second `ScrollTrigger`
 * would otherwise need.
 *
 * ## Step 2 and the Result card get the same reveal, the easy way
 *
 * Both start below the fold, so unlike Step 1 they have a natural "scrolling
 * up into view" distance to trigger from -- their whole intro-hold-exit
 * cycle runs off `useCardScrollReveal`, one hook call each, rather than the
 * hand-built effect above. Step 1 could not use it: it is visible from the
 * first paint, with no such distance to read the intro from.
 */

const loadFeatures = () => import('@/components/motion/features').then((mod) => mod.default);

function taskSpec(name: string): GeneratedTaskSpec | undefined {
  return prediction.tasks.find((entry) => entry.task === name);
}

function samplesFor(name: string): GeneratedSample[] {
  return prediction.samples.filter((entry) => entry.tasks.includes(name));
}

export function PredictionPanel({
  checks,
  className,
}: {
  /** The checks offered. Each task inside a check is still its own label space. */
  checks: readonly AnalyseCheck[];
  className?: string;
}) {
  const reduced = useReducedMotion();
  const [task, setTask] = useState(checks[0]?.tasks[0]?.task ?? 'binary');
  const [live, setLive] = useState<TaskStatus[] | null>(null);
  const [servable, setServable] = useState<SampleStatus[] | null>(null);
  const [probeFailed, setProbeFailed] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [chosenSample, setChosenSample] = useState<string | null>(null);
  const [phase, setPhase] = useState<UploadPhase>('idle');
  const [result, setResult] = useState<PredictResult | null>(null);
  // Bumped on every successful run, purely to key the result reveal's
  // animation -- `PredictResult` carries no request id to key on instead.
  const [resultKey, setResultKey] = useState(0);
  const [failure, setFailure] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // T131: one recording, or a batch. A running batch locks the check and the mode.
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [batchBusy, setBatchBusy] = useState(false);

  const step1Ref = useRef<HTMLDivElement>(null);
  const step2Ref = useRef<HTMLDivElement>(null);
  const resultRef = useRef<HTMLElement>(null);
  useCardScrollReveal(step2Ref);
  useCardScrollReveal(resultRef);

  useEffect(() => {
    const el = step1Ref.current;
    if (el === null) return;

    // Same race `useCardScrollReveal` guards against (see that hook's own
    // comment): `reduced` starts `false` and only flips after its own effect
    // reads `matchMedia`, so on a page where reduced motion is already on
    // from the first paint this effect could still run once with `reduced`
    // stale at `false` -- and GSAP's `fromTo` below writes its "from" opacity
    // to `el`'s inline style synchronously on creation, before the later
    // re-render tears the timeline down. `.kill()` never reverts a style it
    // already wrote. Checking `matchMedia` directly here closes the race;
    // Step 1 has its own copy of this effect (see the doc comment above) so
    // it needs its own copy of the guard, not just `useCardScrollReveal`'s.
    const prefersReduced =
      reduced ||
      (typeof window.matchMedia === 'function' &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    if (prefersReduced) {
      el.style.removeProperty('opacity');
      el.style.removeProperty('transform');
      el.style.removeProperty('filter');
      return;
    }
    let cleanup: (() => void) | undefined;

    void (async () => {
      const [{ default: gsap }, { ScrollTrigger }] = await Promise.all([
        import('gsap'),
        import('gsap/ScrollTrigger'),
      ]);
      gsap.registerPlugin(ScrollTrigger);

      // One timeline per property, covering the card's whole journey --
      // intro, a held resting state, then the exit -- rather than four
      // independent tweens. Two tweens both targeting `filter` (one for the
      // intro, one for the exit) fought over the same property wherever
      // their ranges came near each other, which on this page is close: the
      // card sits high enough that "intro finished" and "exit starting" are
      // not far apart in scroll terms. A single timeline has exactly one
      // owner per property at every scroll position, so there is nothing
      // left to fight.
      //
      // `elTop`/`elBottom` are the card's own absolute position on the page,
      // read once here rather than re-derived from `ScrollTrigger`'s own
      // element-relative positions, so both the page-anchored intro and the
      // card-anchored exit can sit on the same body-anchored timeline.
      const elTop = el.getBoundingClientRect().top + window.scrollY;
      const elBottom = elTop + el.getBoundingClientRect().height;

      const sizeTl = gsap.timeline({
        defaults: { ease: 'none' },
        scrollTrigger: {
          trigger: document.body,
          start: 'top top',
          end: `+=${elBottom}`,
          scrub: true,
        },
      });
      sizeTl
        .fromTo(
          el,
          { scale: CARD_SCROLL_REVEAL_SCALE, opacity: CARD_SCROLL_REVEAL_OPACITY },
          { scale: 1, opacity: 1, duration: STEP1_CARD_SCROLL_PX },
          0,
        )
        // No tween between here and `elTop`: the timeline simply holds
        // whatever the intro left it at, for as long as the card is not yet
        // near the top of the viewport.
        .to(
          el,
          { scale: CARD_SCROLL_REVEAL_SCALE, opacity: CARD_SCROLL_REVEAL_OPACITY, duration: elBottom - elTop },
          elTop,
        );

      const blurTl = gsap.timeline({
        defaults: { ease: 'none' },
        scrollTrigger: {
          trigger: document.body,
          start: 'top top',
          end: `+=${elBottom}`,
          scrub: true,
        },
      });
      blurTl
        .fromTo(
          el,
          { filter: `blur(${CARD_SCROLL_REVEAL_BLUR_PX}px)` },
          { filter: 'blur(0px)', duration: STEP1_CARD_BLUR_SCROLL_PX },
          0,
        )
        // Delayed start, same idea as the hold above: nothing touches
        // `filter` again until STEP1_CARD_EXIT_BLUR_DELAY_PX past `elTop`,
        // so the shrink is already under way before the blur follows it.
        //
        // Duration is `elBottom - elTop` -- the shrink's own full span, not
        // the short STEP1_CARD_BLUR_SCROLL_PX the intro uses -- so the blur
        // keeps climbing for the whole exit instead of maxing out in the
        // first ~90px and then holding flat while the card is still visibly
        // shrinking the rest of the way out. That flat hold was why the exit
        // read as weaker than the intro even after raising the target
        // strength: most of the exit showed no change at all.
        //
        // `power2.in` (2026-09-18, replacing the linear default just for
        // this tween): linear still read as too intense right after the
        // delayed start -- a straight ramp spends as much blur on its first
        // pixels of scroll as its last. An in-ease spends the early part of
        // the exit barely blurring at all and saves most of the climb to
        // `STEP1_CARD_EXIT_BLUR_PX` for the back half, so it reads as normal
        // just after the delay and only gets dramatic as the card actually
        // gets small. (`power2.out`, tried earlier for the opposite reason --
        // to front-load the climb -- was reverted; this is not that.)
        .to(
          el,
          { filter: `blur(${STEP1_CARD_EXIT_BLUR_PX}px)`, duration: elBottom - elTop, ease: 'power2.in' },
          elTop + STEP1_CARD_EXIT_BLUR_DELAY_PX,
        );

      cleanup = () => {
        sizeTl.scrollTrigger?.kill();
        sizeTl.kill();
        blurTl.scrollTrigger?.kill();
        blurTl.kill();
      };
    })();

    return () => cleanup?.();
  }, [reduced]);

  useEffect(() => {
    let disposed = false;
    void (async () => {
      try {
        const [taskRows, sampleRows] = await Promise.all([fetchTasks(), fetchSamples()]);
        if (disposed) return;
        setLive(taskRows);
        setServable(sampleRows);
        setProbeFailed(null);
      } catch (error) {
        if (disposed) return;
        setProbeFailed(error instanceof ApiError ? error.message : String(error));
      }
    })();
    return () => {
      disposed = true;
    };
  }, []);

  const spec = taskSpec(task);
  const liveStatus = live?.find((entry) => entry.task === task) ?? null;
  const available = liveStatus?.available ?? spec?.model_available_at_export ?? false;
  const unavailableReason =
    liveStatus?.reason ?? spec?.reason_at_export ?? 'No model is deployed for this task.';

  const pageSamples = useMemo(() => samplesFor(task), [task]);
  const servableIds = useMemo(
    () => new Set((servable ?? []).filter((row) => row.available).map((row) => row.sample_id)),
    [servable],
  );
  const activeSample = pageSamples.find((entry) => entry.sample_id === chosenSample) ?? null;

  const reset = useCallback(() => {
    setResult(null);
    setFailure(null);
    setPhase('idle');
  }, []);

  /** T130.1 and T130.2: an upload and a recording both land here. */
  const acceptFile = useCallback(
    (chosen: File) => {
      setFile(chosen);
      setChosenSample(null);
      reset();
    },
    [reset],
  );

  const run = useCallback(async () => {
    setBusy(true);
    setFailure(null);
    setResult(null);
    setPhase('uploading');
    try {
      const outcome =
        chosenSample !== null
          ? await predictSample(chosenSample, task)
          : file !== null
            ? await predictFile(file, task)
            : null;
      if (outcome === null) {
        setFailure('Choose a recording first.');
        setPhase('error');
        return;
      }
      setResult(outcome);
      setResultKey((key) => key + 1);
      // T117.2: hand the response to About the Model's Performance tab.
      rememberPrediction(outcome);
      setPhase('done');
    } catch (error) {
      setFailure(error instanceof ApiError ? error.message : String(error));
      setPhase('error');
    } finally {
      setBusy(false);
    }
  }, [chosenSample, file, task]);

  /** T130.6: the report is rebuilt by the API from the same recording. */
  const downloadReport = useCallback(async () => {
    const source =
      chosenSample !== null ? { sampleId: chosenSample } : file !== null ? { file } : null;
    if (source === null) throw new Error('The recording is no longer selected.');
    saveDocument(await sampleReport(task, source));
  }, [chosenSample, file, task]);

  const source = file ?? (chosenSample !== null ? sampleAudioUrl(chosenSample) : null);

  return (
    <div className={cn('space-y-6', className)}>
      <div ref={step1Ref}>
        <GlassCard eyebrow="Step 1" title="Choose a check" className="rounded-3xl">
          <CheckSelector
            checks={checks}
            task={task}
            disabled={busy || batchBusy}
            onChange={(next) => {
              setTask(next);
              setChosenSample(null);
              reset();
            }}
          />
          {spec !== undefined ? (
            <p className={cn(TYPE_SCALE.caption, 'mt-3 max-w-prose text-ink-3')}>
              {spec.description} Categories: {spec.classes.join(', ')}.
            </p>
          ) : null}
        </GlassCard>
      </div>

      {!available ? (
        <EmptyState
          title="No model is deployed for this task yet"
          description={<p>{unavailableReason}</p>}
        />
      ) : (
        <>
          <div role="group" aria-label="How many recordings" className="flex flex-wrap gap-2">
            {(['single', 'batch'] as const).map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={mode === value}
                disabled={busy || batchBusy}
                onClick={() => setMode(value)}
                className={cn(
                  'rounded-lg border px-3 py-1.5 text-label-md uppercase transition-colors',
                  mode === value
                    ? 'border-accent bg-accent-soft text-accent-deep'
                    : 'border-line bg-panel text-ink-2 hover:border-accent-line',
                  (busy || batchBusy) && 'cursor-not-allowed opacity-60',
                )}
              >
                {value === 'single' ? 'One recording' : 'Several recordings'}
              </button>
            ))}
          </div>
          {mode === 'batch' ? (
            <div ref={step2Ref}>
              <GlassCard eyebrow="Step 2" title="Add several recordings" className="rounded-3xl">
                {/* Keyed by task: a batch's rows belong to the check they were scored for. */}
                <BatchPanel
                  key={task}
                  task={task}
                  taskTitle={spec?.title ?? task}
                  classes={spec?.classes ?? []}
                  onRunningChange={setBatchBusy}
                />
              </GlassCard>
            </div>
          ) : (
          <>
          <div ref={step2Ref}>
          <GlassCard eyebrow="Step 2" title="Add a recording" className="rounded-3xl">
            <FileUpload onFile={acceptFile} phase={phase} fileName={file?.name ?? null} disabled={busy} />

            <MicRecorder
              onFile={acceptFile}
              className="mt-4"
              maxSeconds={prediction.upload.max_duration_seconds}
              minSeconds={prediction.upload.min_duration_seconds}
              minDisplay={prediction.upload.min_duration_display}
              disabled={busy}
            />

            <p className={cn(TYPE_SCALE.caption, 'mt-4 text-ink-3')}>
              Accepted: {prediction.upload.accepted_suffixes.join(', ')}, between{' '}
              {prediction.upload.min_duration_display} and {prediction.upload.max_duration_display}.{' '}
              {prediction.upload.duration_note}
            </p>

            <WaveformPlayer
              className="mt-5"
              source={source}
              label={file?.name ?? (activeSample !== null ? activeSample.dataset_name + ' sample' : 'recording')}
            />

            <Button
              tone="primary"
              size="lg"
              icon="pulse"
              className="mt-5 w-full"
              onClick={() => void run()}
              disabled={busy || (file === null && chosenSample === null)}
            >
              {busy ? 'Analysing…' : 'Analyse recording'}
            </Button>

            {pageSamples.length > 0 ? (
              <div className="mt-5">
                <p className="label-micro">Or try a sample from the datasets</p>
                <ul className="mt-2 grid gap-2 sm:grid-cols-2">
                  {pageSamples.map((entry) => {
                    const reachable = servable === null || servableIds.has(entry.sample_id);
                    const active = entry.sample_id === chosenSample;
                    const label = entry.labels[task];
                    return (
                      <li key={entry.sample_id}>
                        <button
                          type="button"
                          // Addressed by sample id (the capture plan and the
                          // browser tests), not by its visible text.
                          data-sample-id={entry.sample_id}
                          disabled={!reachable || busy}
                          aria-pressed={active}
                          title={entry.selection}
                          onClick={() => {
                            setChosenSample(active ? null : entry.sample_id);
                            setFile(null);
                            reset();
                          }}
                          className={cn(
                            'w-full rounded-lg border px-3 py-2 text-left transition-colors',
                            active
                              ? 'border-accent bg-accent-soft shadow-panel'
                              : 'border-line bg-sunken hover:border-accent-line',
                            !reachable && 'cursor-not-allowed opacity-55',
                          )}
                        >
                          <span className={cn(TYPE_SCALE.body, 'block font-medium capitalize text-ink')}>
                            {label ?? entry.dataset_name}
                          </span>
                          <span className={cn(TYPE_SCALE.caption, 'mt-0.5 block text-ink-3')}>
                            {reachable
                              ? entry.dataset_name + ' · ' + entry.duration_display
                              : 'The datasets are not on this machine.'}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ) : null}

            {probeFailed !== null ? (
              <p className={cn(TYPE_SCALE.caption, 'mt-3 rounded border border-warn-line bg-warn-soft p-2 text-warn')}>
                {probeFailed}
              </p>
            ) : null}
          </GlassCard>
          </div>

          <section ref={resultRef} aria-label="Result" className="space-y-3">
            <h2 className={cn(TYPE_SCALE.h2, 'text-ink')}>Result</h2>
            <LazyMotion features={loadFeatures} strict>
              <AnimatePresence mode="wait">
                {busy ? (
                  <m.div
                    key="busy"
                    initial={reduced ? undefined : { opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={reduced ? undefined : { opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <AnalyseProgress className="rounded-3xl" />
                  </m.div>
                ) : failure !== null ? (
                  <m.div
                    key="failure"
                    initial={reduced ? undefined : { opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={LIFT_SPRING}
                  >
                    <ErrorState
                      title="No prediction was produced"
                      detail={failure}
                      onRetry={() => void run()}
                      className="rounded-3xl"
                    />
                  </m.div>
                ) : result !== null ? (
                  // Keyed by result identity so a second analysis on the same task
                  // re-triggers the reveal rather than reusing the first mount.
                  <m.div
                    key={'result-' + resultKey}
                    initial={reduced ? undefined : { opacity: 0, scale: 0.96, y: 10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={LIFT_SPRING}
                  >
                    <ResultCard
                      result={result}
                      classes={spec?.classes ?? []}
                      taskTitle={spec?.title}
                      sample={activeSample}
                      onDownloadReport={downloadReport}
                      className="rounded-3xl"
                    />
                  </m.div>
                ) : (
                  <m.div
                    key="empty"
                    initial={reduced ? undefined : { opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.2 }}
                  >
                    <EmptyState
                      icon="pulse"
                      title="Nothing scored yet"
                      description="Add a recording and select Analyse recording. The result appears here."
                      className="rounded-3xl"
                    />
                  </m.div>
                )}
              </AnimatePresence>
            </LazyMotion>
          </section>
          </>
        )}
        </>
      )}
    </div>
  );
}
