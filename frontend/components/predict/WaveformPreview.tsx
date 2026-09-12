'use client';

import { useEffect, useRef, useState } from 'react';

import { audioBytes } from '@/lib/api';
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
 * ## Web Audio is the FALLBACK here, not the decoder
 *
 * `AudioContext.decodeAudioData` refuses any file whose sample rate is below
 * 3 kHz -- that is the `BaseAudioContext` sample-rate floor, not a property of
 * the file -- and **PhysioNet 2016 is recorded at 2 kHz**. So every built-in
 * sample on the binary prediction page, and every PhysioNet recording anybody
 * uploads, came back as "Unable to decode audio data" with no waveform at all.
 * Found by the Phase 120 screenshot run (SS-03/SS-08), which is the first thing
 * in the project to look at what that panel actually rendered for a corpus
 * recording; the Phase 118 upload test generates its own 2 kHz tone and asserts
 * the PREDICTION, never the preview.
 *
 * Uncompressed PCM WAV is therefore parsed here, from the RIFF header, which is
 * both what every recording in all four corpora is and what the upload control
 * accepts. Web Audio is kept for anything else a browser might hand this
 * component. Reading 44 bytes of header is not "the client computing": no
 * number produced here is reported, compared or rounded.
 *
 * ## Decoding can fail, and that is a rendered error
 *
 * A truncated or non-audio file rejects. That surfaces as `ErrorState`, never as
 * a blank canvas: an empty box is indistinguishable from silence, and silence is
 * a legitimate recording.
 */

const HEIGHT = 96;
const BUCKETS = 900;

interface Decoded {
  peaks: Float32Array;
  duration: number;
  sampleRate: number;
  channels: number;
}

/** Peak envelope of one channel, low and high per bucket. */
function envelope(samples: Float32Array): Float32Array {
  const step = Math.max(1, Math.floor(samples.length / BUCKETS));
  const peaks = new Float32Array(BUCKETS * 2);
  for (let bucket = 0; bucket < BUCKETS; bucket += 1) {
    let low = 0;
    let high = 0;
    const start = bucket * step;
    const stop = Math.min(samples.length, start + step);
    for (let index = start; index < stop; index += 1) {
      const value = samples[index] ?? 0;
      if (value < low) low = value;
      if (value > high) high = value;
    }
    peaks[bucket * 2] = low;
    peaks[bucket * 2 + 1] = high;
  }
  return peaks;
}

/**
 * The first channel of an uncompressed RIFF/WAVE file, or null if it is not one.
 *
 * Chunks are walked rather than assumed at offset 36: PASCAL's set_a files
 * carry a `LIST` chunk before `data`, so a parser that trusted the canonical
 * 44-byte layout would read metadata as audio.
 */
function decodePcmWav(data: ArrayBuffer): Decoded | null {
  const view = new DataView(data);
  const text = (offset: number): string =>
    String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
  if (data.byteLength < 44 || text(0) !== 'RIFF' || text(8) !== 'WAVE') return null;

  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let dataStart = -1;
  let dataLength = 0;

  let offset = 12;
  while (offset + 8 <= data.byteLength) {
    const id = text(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === 'fmt ' && body + 16 <= data.byteLength) {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
    } else if (id === 'data') {
      dataStart = body;
      dataLength = Math.min(size, data.byteLength - body);
    }
    offset = body + size + (size % 2); // chunks are word-aligned
  }

  const PCM = 1;
  const FLOAT = 3;
  if (dataStart < 0 || channels < 1 || sampleRate < 1) return null;
  if (!(format === PCM && (bits === 8 || bits === 16 || bits === 24 || bits === 32))) {
    if (!(format === FLOAT && bits === 32)) return null;
  }

  const bytesPerSample = bits / 8;
  const frames = Math.floor(dataLength / (bytesPerSample * channels));
  const samples = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    const at = dataStart + frame * bytesPerSample * channels; // channel 0 only
    if (format === FLOAT) samples[frame] = view.getFloat32(at, true);
    else if (bits === 8) samples[frame] = (view.getUint8(at) - 128) / 128;
    else if (bits === 16) samples[frame] = view.getInt16(at, true) / 32768;
    else if (bits === 24) {
      const raw =
        view.getUint8(at) | (view.getUint8(at + 1) << 8) | (view.getInt8(at + 2) << 16);
      samples[frame] = raw / 8388608;
    } else samples[frame] = view.getInt32(at, true) / 2147483648;
  }

  return {
    peaks: envelope(samples),
    duration: frames / sampleRate,
    sampleRate,
    channels,
  };
}

async function decode(data: ArrayBuffer): Promise<Decoded> {
  const pcm = decodePcmWav(data);
  if (pcm !== null) {
    if (pcm.duration <= 0) throw new Error('That WAV file contains no audio frames.');
    return pcm;
  }
  const Context =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (Context === undefined) throw new Error('This browser has no Web Audio decoder.');
  const context = new Context();
  try {
    const buffer = await context.decodeAudioData(data);
    return {
      peaks: envelope(buffer.getChannelData(0)),
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
            ? await audioBytes(source)
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
