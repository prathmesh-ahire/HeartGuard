'use client';

import { useEffect, useRef, useState } from 'react';

import { cn } from '@/lib/cn';
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States';
import { SURFACE, TYPE_SCALE, seriesColor } from '@/lib/tokens';

/**
 * The uploaded recording, drawn (T116.1).
 *
 * ## Why this one component is allowed to compute
 *
 * The rule is that the client never computes a **metric**. This draws the peak
 * envelope of a file the visitor chose thirty seconds ago, which has no
 * precomputed form and never can: it did not exist when the exporter ran. It is
 * a picture of the user's own input, not a result — nothing here is scored,
 * compared, rounded or reported, and the duration and sample rate shown come
 * from the decoder rather than from any analysis.
 *
 * The signal the MODEL saw is a different thing again: resampled to 2 kHz,
 * band-pass filtered and normalized by `src/preprocessing/`. This is the raw
 * file as the browser decoded it, and the caption says so, because showing a
 * raw waveform under a prediction invites the reading that the model consumed
 * exactly this.
 *
 * ## Decoding can fail, and that is a rendered error
 *
 * `decodeAudioData` rejects on a truncated or non-audio file. That surfaces as
 * `ErrorState`, never as a blank canvas: an empty box is indistinguishable from
 * silence, and silence is a legitimate recording.
 */

const HEIGHT = 96;
const BUCKETS = 900;

interface Decoded {
  peaks: Float32Array;
  duration: number;
  sampleRate: number;
  channels: number;
}

async function decode(data: ArrayBuffer): Promise<Decoded> {
  const Context =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (Context === undefined) throw new Error('This browser has no Web Audio decoder.');
  const context = new Context();
  try {
    const buffer = await context.decodeAudioData(data);
    const channel = buffer.getChannelData(0);
    const step = Math.max(1, Math.floor(channel.length / BUCKETS));
    const peaks = new Float32Array(BUCKETS * 2);
    for (let bucket = 0; bucket < BUCKETS; bucket += 1) {
      let low = 0;
      let high = 0;
      const start = bucket * step;
      const stop = Math.min(channel.length, start + step);
      for (let index = start; index < stop; index += 1) {
        const value = channel[index] ?? 0;
        if (value < low) low = value;
        if (value > high) high = value;
      }
      peaks[bucket * 2] = low;
      peaks[bucket * 2 + 1] = high;
    }
    return {
      peaks,
      duration: buffer.duration,
      sampleRate: buffer.sampleRate,
      channels: buffer.numberOfChannels,
    };
  } finally {
    void context.close();
  }
}

export function WaveformPreview({
  source,
  label,
  className,
}: {
  /** A chosen File, or a URL the API serves a built-in sample from. */
  source: File | string | null;
  label: string;
  className?: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<'idle' | 'loading' | 'ready' | 'failed'>('idle');
  const [message, setMessage] = useState('');
  const [decoded, setDecoded] = useState<Decoded | null>(null);

  useEffect(() => {
    if (source === null) {
      setState('idle');
      setDecoded(null);
      return;
    }
    let disposed = false;
    setState('loading');
    void (async () => {
      try {
        const data =
          typeof source === 'string'
            ? await (await fetch(source)).arrayBuffer()
            : await source.arrayBuffer();
        const result = await decode(data);
        if (disposed) return;
        setDecoded(result);
        setState('ready');
      } catch (error) {
        if (disposed) return;
        setMessage(
          error instanceof Error
            ? error.message
            : 'The browser could not decode that file as audio.',
        );
        setState('failed');
      }
    })();
    return () => {
      disposed = true;
    };
  }, [source]);

  useEffect(() => {
    const node = canvas.current;
    if (node === null || decoded === null) return;
    const ratio = window.devicePixelRatio || 1;
    const width = node.clientWidth;
    node.width = Math.floor(width * ratio);
    node.height = Math.floor(HEIGHT * ratio);
    const context = node.getContext('2d');
    if (context === null) return;
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, HEIGHT);

    const middle = HEIGHT / 2;
    context.strokeStyle = 'rgba(148,163,184,0.55)';
    context.beginPath();
    context.moveTo(0, middle);
    context.lineTo(width, middle);
    context.stroke();

    context.fillStyle = seriesColor(0);
    const columns = decoded.peaks.length / 2;
    for (let column = 0; column < columns; column += 1) {
      const low = decoded.peaks[column * 2] ?? 0;
      const high = decoded.peaks[column * 2 + 1] ?? 0;
      const x = (column / columns) * width;
      const top = middle - high * middle;
      const bottom = middle - low * middle;
      context.fillRect(x, top, Math.max(1, width / columns), Math.max(1, bottom - top));
    }
  }, [decoded]);

  if (source === null) {
    return (
      <EmptyState
        className={className}
        title="No recording chosen yet"
        description="Choose a built-in sample or drop a WAV file to see its waveform here."
      />
    );
  }
  if (state === 'loading') return <LoadingState className={className} label="Decoding the recording" />;
  if (state === 'failed') {
    return (
      <ErrorState
        className={className}
        title="That file could not be decoded"
        detail={message}
      />
    );
  }

  return (
    <figure className={cn('rounded-lg border border-slate-200 p-4 dark:border-slate-800', className)}>
      <figcaption className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mb-2')}>
        {label} — the raw file as the browser decoded it. This is not the signal the model
        scored: preprocessing resamples, band-pass filters and normalizes it first.
      </figcaption>
      <canvas
        ref={canvas}
        role="img"
        aria-label={'Waveform of ' + label}
        style={{ width: '100%', height: HEIGHT }}
      />
    </figure>
  );
}
