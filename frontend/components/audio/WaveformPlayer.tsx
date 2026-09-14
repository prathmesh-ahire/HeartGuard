'use client';

import { useEffect, useRef, useState } from 'react';
import { useTheme } from 'next-themes';

import { audioBytes } from '@/lib/api';
import { cn } from '@/lib/cn';
import { tokenColour } from '@/lib/cssTokens';
import {
  PLAYBACK_MIN_RATE,
  PLAYBACK_RATE,
  clock,
  decodeAudio,
  encodeWav,
  envelope,
  resampleLinear,
  type DecodedAudio,
} from '@/lib/wav';
import { Icon } from '@/components/ui/Icon';
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States';
import { TYPE_SCALE } from '@/lib/tokens';

/**
 * The chosen recording: waveform, playback, scrubbing and a playhead (T130.3).
 *
 * ## The waveform is drawn from peaks this page computed, not by WaveSurfer
 *
 * WaveSurfer decodes through Web Audio, which refuses PhysioNet's 2 kHz files
 * (see `lib/wav.ts`). So the file is parsed here, its envelope is handed to
 * WaveSurfer as `peaks` with the `duration`, and WaveSurfer never decodes: it
 * draws those peaks and plays through an `<audio>` element.
 *
 * ## A low-rate recording is PLAYED from a resampled copy
 *
 * The `<audio>` element refuses a 2 kHz WAV too ("no supported source", measured
 * in T130.3), so below `PLAYBACK_MIN_RATE` the player gets a 44.1 kHz copy made
 * here. The copy exists only to be heard: the waveform is drawn from the
 * original samples, and analysis, History and the report all receive the
 * original file, untouched.
 *
 * ## It is the raw file, and it says so
 *
 * The model scores a resampled, band-pass filtered and normalized version of
 * this signal, so the caption states the difference.
 *
 * ## WaveSurfer loads lazily
 *
 * It is imported inside the effect, so it is fetched only when a recording is
 * chosen and never sits in the first-load bundle (the T113 lazy rule, checked
 * by `scripts/20_check_bundle_budget.py`).
 */

const HEIGHT = 88;

interface Player {
  playPause: () => Promise<void>;
  setTime: (seconds: number) => void;
  getCurrentTime: () => number;
  setOptions: (options: Record<string, unknown>) => void;
  destroy: () => void;
}

function colours() {
  return {
    waveColor: tokenColour('ink-3', 0.55),
    progressColor: tokenColour('accent'),
    cursorColor: tokenColour('ink'),
  };
}

/** A URL the browser's `<audio>` element will actually play. */
function playableUrl(source: File | string, decoded: DecodedAudio): string {
  if (decoded.sampleRate < PLAYBACK_MIN_RATE) {
    const copy = encodeWav(
      [resampleLinear(decoded.samples, decoded.sampleRate, PLAYBACK_RATE)],
      PLAYBACK_RATE,
    );
    return URL.createObjectURL(new Blob([copy], { type: 'audio/wav' }));
  }
  return typeof source === 'string' ? source : URL.createObjectURL(source);
}

export function WaveformPlayer({
  source,
  label,
  className,
}: {
  /** A chosen File (an upload or a microphone recording), or a sample's URL. */
  source: File | string | null;
  label: string;
  className?: string;
}) {
  const container = useRef<HTMLDivElement>(null);
  const player = useRef<Player | null>(null);
  const { resolvedTheme } = useTheme();
  const [status, setStatus] = useState<'loading' | 'ready' | 'failed'>('loading');
  const [message, setMessage] = useState('');
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackProblem, setPlaybackProblem] = useState<string | null>(null);

  useEffect(() => {
    if (source === null) return;
    let disposed = false;
    let playUrl: string | null = null;
    setStatus('loading');
    setPlaying(false);
    setPosition(0);
    setPlaybackProblem(null);

    void (async () => {
      try {
        const bytes = typeof source === 'string' ? await audioBytes(source) : await source.arrayBuffer();
        const decoded = await decodeAudio(bytes);
        const { default: WaveSurfer } = await import('wavesurfer.js');
        if (disposed || container.current === null) return;

        playUrl = playableUrl(source, decoded);
        const instance = WaveSurfer.create({
          container: container.current,
          height: HEIGHT,
          ...colours(),
          cursorWidth: 2,
          barWidth: 2,
          barGap: 1,
          barRadius: 2,
          normalize: true,
          dragToSeek: true,
          url: playUrl,
          peaks: [envelope(decoded.samples)],
          duration: decoded.duration,
        });
        const follow = () => {
          const now = instance.getCurrentTime();
          if (Number.isFinite(now)) setPosition(now);
        };
        instance.on('play', () => setPlaying(true));
        instance.on('pause', () => setPlaying(false));
        instance.on('finish', () => setPlaying(false));
        instance.on('timeupdate', follow);
        instance.on('seeking', follow);
        instance.on('interaction', follow);
        instance.on('error', (error: unknown) => {
          setPlaying(false);
          setPlaybackProblem(
            'This browser could not play the file, but its waveform is shown. ' +
              (error instanceof Error ? error.message : ''),
          );
        });
        player.current = instance as unknown as Player;
        setDuration(decoded.duration);
        setStatus('ready');
      } catch (error) {
        if (disposed) return;
        setMessage(error instanceof Error ? error.message : 'The browser could not read that file as audio.');
        setStatus('failed');
      }
    })();

    return () => {
      disposed = true;
      player.current?.destroy();
      player.current = null;
      if (playUrl !== null && playUrl.startsWith('blob:')) URL.revokeObjectURL(playUrl);
    };
  }, [source]);

  // Canvas colours are drawn by script, so they are re-read when the theme flips.
  useEffect(() => {
    player.current?.setOptions(colours());
  }, [resolvedTheme, status]);

  if (source === null) {
    return (
      <EmptyState
        className={className}
        icon="waveform"
        title="No recording yet"
        description="Drop a file, record one, or pick a sample to see and hear it here."
      />
    );
  }

  return (
    <figure
      className={cn('rounded-xl border border-line bg-panel p-4', className)}
      data-player-status={status}
    >
      <figcaption className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="truncate font-mono text-label-lg text-ink">{label}</span>
        <span className={cn(TYPE_SCALE.caption, 'text-ink-3')}>
          The file as recorded. The model scores a cleaned copy of it.
        </span>
      </figcaption>

      {status === 'loading' ? <LoadingState className="mt-3" label="Reading the recording" rows={2} /> : null}
      {status === 'failed' ? (
        <ErrorState className="mt-3" title="That file could not be read as audio" detail={message} />
      ) : null}

      <div
        ref={container}
        role="img"
        aria-label={'Waveform of ' + label}
        aria-hidden={status !== 'ready'}
        className={cn(
          'mt-3 cursor-pointer',
          status === 'loading' && 'invisible',
          status === 'failed' && 'h-0 overflow-hidden',
        )}
      />

      {status === 'ready' ? (
        <div className="mt-3 flex items-center gap-3">
          <button
            type="button"
            onClick={() => void player.current?.playPause()}
            aria-label={playing ? 'Pause' : 'Play'}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-accent-strong bg-accent text-on-accent shadow-accent hover:bg-accent-strong"
          >
            <Icon name={playing ? 'pause' : 'play'} className="h-4 w-4" strokeWidth={2} />
          </button>
          <input
            type="range"
            min={0}
            max={duration}
            step="any"
            value={position}
            aria-label="Playback position"
            onChange={(event) => {
              const seconds = Number(event.target.value);
              setPosition(seconds);
              player.current?.setTime(seconds);
            }}
            className="h-1 min-w-0 flex-1 cursor-pointer accent-accent"
          />
          <span className="shrink-0 font-mono text-label-md text-ink-2" data-testid="player-clock">
            {clock(position)} / {clock(duration)}
          </span>
        </div>
      ) : null}

      {playbackProblem !== null ? (
        <p role="status" className={cn(TYPE_SCALE.caption, 'mt-2 text-warn')}>
          {playbackProblem}
        </p>
      ) : null}
    </figure>
  );
}
