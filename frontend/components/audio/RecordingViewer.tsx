'use client';

import { useEffect, useRef, useState } from 'react';

import { cn } from '@/lib/cn';
import { segmentation } from '@/lib/generated';
import { EmptyState } from '@/components/ui/States';
import { SURFACE, TYPE_SCALE, seriesColor } from '@/lib/tokens';

/**
 * Waveform, spectrogram and cardiac-cycle overlay (T113.6).
 *
 * ## The overlay is the dataset's, not ours
 *
 * Every S1 / systole / S2 / diastole band on the waveform is one row of
 * `85197_TV.tsv` — CirCor's own expert-reviewed segmentation, exported by
 * `src/reporting/segmentation.py` with the TSV's sha256 recorded beside it. The
 * task says "not a synthetic animation" and that is the difference: no boundary
 * here was estimated, smoothed or generated. Label 0 is drawn as **unannotated**
 * rather than as a fifth phase, because the segmentation declining to label a
 * span is information and painting it as diastole would invent an observation.
 *
 * ## Attribution is rendered, not filed
 *
 * ODC-By section 4.3 requires a notice associated with the Produced Work
 * wherever it is publicly used. The dashboard is the Produced Work, so the
 * attribution line sits under the player — visible to anyone who plays the
 * audio — as well as in `public/NOTICE.md` and the README.
 *
 * ## It is a dataset sample
 *
 * Never "patient", never "case". De-identified paediatric screening data, in a
 * prototype that does not diagnose.
 */

const PHASE_COLOR: Record<string, number> = {
  s1: 0,
  systole: 1,
  s2: 2,
  diastole: 3,
};

export function RecordingViewer({ className }: { className?: string }) {
  const waveform = useRef<HTMLDivElement>(null);
  const spectrogram = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'failed'>('loading');
  const [message, setMessage] = useState('');
  const [playing, setPlaying] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const playerRef = useRef<{
    playPause: () => void;
    destroy: () => void;
    on: (event: string, handler: (...args: unknown[]) => void) => void;
  } | null>(null);

  const available = segmentation.available !== false && segmentation.audio_url !== undefined;

  useEffect(() => {
    if (!available) return;
    const node = waveform.current;
    const spectrogramNode = spectrogram.current;
    if (node === null) return;

    let disposed = false;

    void (async () => {
      try {
        const [{ default: WaveSurfer }, regionsModule, spectrogramModule] = await Promise.all([
          import('wavesurfer.js'),
          import('wavesurfer.js/dist/plugins/regions.esm.js'),
          import('wavesurfer.js/dist/plugins/spectrogram.esm.js'),
        ]);
        if (disposed) return;

        const RegionsPlugin = regionsModule.default;
        const SpectrogramPlugin = spectrogramModule.default;
        const regions = RegionsPlugin.create();

        const player = WaveSurfer.create({
          container: node,
          height: 96,
          waveColor: '#94a3b8',
          progressColor: seriesColor(0),
          cursorColor: seriesColor(1),
          normalize: true,
          plugins: [regions],
        });

        if (spectrogramNode !== null) {
          player.registerPlugin(
            SpectrogramPlugin.create({
              container: spectrogramNode,
              labels: true,
              height: 160,
            }),
          );
        }

        player.on('decode', () => {
          for (const segment of segmentation.segments) {
            const index = PHASE_COLOR[segment.key];
            regions.addRegion({
              start: segment.start,
              end: segment.end,
              drag: false,
              resize: false,
              color:
                index === undefined
                  ? 'rgba(148,163,184,0.18)'
                  : hexToRgba(seriesColor(index), 0.22),
            });
          }
        });
        player.on('play', () => setPlaying(true));
        player.on('pause', () => setPlaying(false));
        player.on('finish', () => setPlaying(false));
        player.on('timeupdate', (time: number) => {
          const current = segmentation.segments.find(
            (segment) => time >= segment.start && time < segment.end,
          );
          setPhase(current?.name ?? null);
        });
        player.on('ready', () => setStatus('ready'));
        player.on('error', (error: unknown) => {
          setMessage(error instanceof Error ? error.message : String(error));
          setStatus('failed');
        });

        void player.load(segmentation.audio_url as string);
        playerRef.current = player as unknown as typeof playerRef.current;
      } catch (error) {
        if (disposed) return;
        setMessage(error instanceof Error ? error.message : String(error));
        setStatus('failed');
      }
    })();

    return () => {
      disposed = true;
      playerRef.current?.destroy();
      playerRef.current = null;
    };
  }, [available]);

  if (!available) {
    return (
      <EmptyState
        className={className}
        title="No recording sample in this build"
        description={
          segmentation.reason ??
          'The corpus was not present when this dashboard was exported, so no audio sample was copied out of it.'
        }
      />
    );
  }

  return (
    <section className={cn(SURFACE.card, 'p-5', className)}>
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className={cn(TYPE_SCALE.h3)}>{segmentation.label}</h3>
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted)}>
          {segmentation.duration_display} · {segmentation.sample_rate_hz} Hz ·{' '}
          {segmentation.n_segments} segments
        </p>
      </header>

      <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>{segmentation.scope_note}</p>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => playerRef.current?.playPause()}
          disabled={status !== 'ready'}
          className={cn(
            TYPE_SCALE.body,
            'rounded border px-3 py-1.5 disabled:opacity-50',
            'border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800',
          )}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <span className={cn(TYPE_SCALE.caption, SURFACE.muted)} aria-live="polite">
          {status === 'loading'
            ? 'Decoding the recording…'
            : phase === null
              ? 'Stopped'
              : 'Now in: ' + phase}
        </span>
      </div>

      <div ref={waveform} className="mt-4" />
      <div ref={spectrogram} className="mt-2" />

      {status === 'failed' ? (
        <p
          role="alert"
          className={cn(
            TYPE_SCALE.caption,
            'mt-3 rounded border border-rose-300 bg-rose-50 p-3 dark:border-rose-800 dark:bg-rose-950/50',
          )}
        >
          The recording could not be loaded — this is a failure, not an empty result. {message}
        </p>
      ) : null}

      <ul className="mt-4 flex flex-wrap gap-x-5 gap-y-2">
        {segmentation.legend
          .filter((entry) => entry.n_segments > 0)
          .map((entry) => (
            <li key={entry.key} className={cn(TYPE_SCALE.caption, 'flex items-center gap-2')}>
              <span
                aria-hidden="true"
                className="inline-block h-3 w-3 rounded-sm border border-slate-400/60"
                style={{
                  backgroundColor:
                    PHASE_COLOR[entry.key] === undefined
                      ? 'rgba(148,163,184,0.35)'
                      : hexToRgba(seriesColor(PHASE_COLOR[entry.key] as number), 0.45),
                }}
              />
              <span>
                {entry.name}
                <span className={cn(SURFACE.subtle, ' ml-1 tabular-nums')}>
                  ×{entry.n_segments} · {entry.seconds_display}
                </span>
              </span>
            </li>
          ))}
      </ul>

      {segmentation.provenance === undefined ? null : (
        <footer className={cn(SURFACE.sunken, 'mt-5 space-y-1 p-4')}>
          {/* ODC-By 4.3: the notice travels with the Produced Work. */}
          <p className={cn(TYPE_SCALE.caption)}>
            Contains information from{' '}
            <a
              href={segmentation.provenance.source_url}
              className="underline"
              rel="noreferrer noopener"
              target="_blank"
            >
              {segmentation.provenance.dataset}
            </a>{' '}
            (record {segmentation.record_id}), which is made available under the{' '}
            <a
              href={segmentation.provenance.licence_uri}
              className="underline"
              rel="noreferrer noopener"
              target="_blank"
            >
              {segmentation.provenance.licence}
            </a>
            .
          </p>
          <p className={cn(TYPE_SCALE.micro, SURFACE.subtle, 'font-mono')}>
            overlay from {segmentation.provenance.tsv_source} · sha256{' '}
            {segmentation.provenance.tsv_sha256.slice(0, 16)}…
          </p>
          <p className={cn(TYPE_SCALE.micro, SURFACE.subtle)}>
            Full notice: <a href={segmentation.provenance.notice} className="underline">
              NOTICE.md
            </a>
          </p>
        </footer>
      )}
    </section>
  );
}

/** ECharts and WaveSurfer both want rgba; the palette is hex. */
function hexToRgba(hex: string, alpha: number): string {
  const value = hex.replace('#', '');
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}
