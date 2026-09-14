/**
 * WAV bytes in and out of the browser (T130.2, T130.3).
 *
 * ## Why the page parses WAV itself
 *
 * `AudioContext.decodeAudioData` refuses any file whose sample rate is below
 * 3 kHz -- the `BaseAudioContext` sample-rate floor, not a property of the file
 * -- and **PhysioNet 2016 is recorded at 2 kHz**. Found in Phase 120: every
 * built-in binary sample came back as "Unable to decode audio data". So an
 * uncompressed PCM WAV, which is what every recording in all four corpora is
 * and what the upload control accepts, is parsed here from its RIFF chunks.
 * Web Audio stays the fallback for anything else.
 *
 * ## Why a microphone recording is encoded here
 *
 * `MediaRecorder` produces WebM/Opus or MP4/AAC, and the service accepts
 * uncompressed WAV only, because a lossy codec changes the spectral content the
 * features are computed from. So the recorder captures raw PCM and this module
 * writes it as a 16-bit mono WAV -- the same kind of file an upload is, sent
 * down the same path.
 *
 * Nothing here is a metric. It reads and writes the operator's own audio; no
 * value produced is reported, compared or rounded for display.
 */

export interface DecodedAudio {
  /** The first channel, as floats in [-1, 1]. */
  samples: Float32Array;
  duration: number;
  sampleRate: number;
  channels: number;
}

/** The first channel of an uncompressed RIFF/WAVE file, or null if it is not one. */
export function decodePcmWav(data: ArrayBuffer): DecodedAudio | null {
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

  // Chunks are walked rather than assumed at offset 36: PASCAL's set_a files
  // carry a `LIST` chunk before `data`.
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
      const raw = view.getUint8(at) | (view.getUint8(at + 1) << 8) | (view.getInt8(at + 2) << 16);
      samples[frame] = raw / 8388608;
    } else samples[frame] = view.getInt32(at, true) / 2147483648;
  }

  return { samples, duration: frames / sampleRate, sampleRate, channels };
}

/** PCM WAV first; Web Audio only for a file that is not one. */
export async function decodeAudio(data: ArrayBuffer): Promise<DecodedAudio> {
  const pcm = decodePcmWav(data);
  if (pcm !== null) {
    if (pcm.duration <= 0) throw new Error('That WAV file contains no audio frames.');
    return pcm;
  }
  const Context =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (Context === undefined) throw new Error('This browser has no audio decoder.');
  const context = new Context();
  try {
    const buffer = await context.decodeAudioData(data);
    return {
      samples: buffer.getChannelData(0),
      duration: buffer.duration,
      sampleRate: buffer.sampleRate,
      channels: buffer.numberOfChannels,
    };
  } finally {
    void context.close();
  }
}

/**
 * Low and high per bucket, interleaved. The waveform renderer takes the
 * extremes over each pixel, so interleaving keeps both halves of the envelope
 * at any zoom.
 */
export function envelope(samples: Float32Array, buckets = 2000): Float32Array {
  const step = Math.max(1, Math.floor(samples.length / buckets));
  const count = Math.min(buckets, Math.ceil(samples.length / step));
  const peaks = new Float32Array(count * 2);
  for (let bucket = 0; bucket < count; bucket += 1) {
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

/** Concatenated captured blocks as one 16-bit mono PCM WAV. */
export function encodeWav(blocks: readonly Float32Array[], sampleRate: number): ArrayBuffer {
  const frames = blocks.reduce((total, block) => total + block.length, 0);
  const buffer = new ArrayBuffer(44 + frames * 2);
  const view = new DataView(buffer);
  const write = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };
  write(0, 'RIFF');
  view.setUint32(4, 36 + frames * 2, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, frames * 2, true);

  let offset = 44;
  for (const block of blocks) {
    for (let index = 0; index < block.length; index += 1) {
      const clipped = Math.max(-1, Math.min(1, block[index] ?? 0));
      view.setInt16(offset, clipped < 0 ? clipped * 32768 : clipped * 32767, true);
      offset += 2;
    }
  }
  return buffer;
}

/**
 * Below this rate the recording is played from a resampled copy.
 *
 * Chromium's `<audio>` element refuses a 2 kHz WAV outright ("no supported
 * source was found", MediaError 4) -- measured in T130.3, where PhysioNet's
 * 2 kHz files could be drawn but not heard. 8 kHz and 44.1 kHz files played.
 * The threshold sits well clear of the refusal rather than at the lowest rate
 * seen to work.
 */
export const PLAYBACK_MIN_RATE = 22050;
export const PLAYBACK_RATE = 44100;

/**
 * Linear-interpolation resampling, for a LISTENING copy only.
 *
 * The file sent for analysis is never this copy: the service receives the
 * original bytes and does its own resampling. Linear interpolation is a weak
 * low-pass, which is adequate to hear heart sounds (energy below ~1 kHz) and
 * would not be adequate for anything that is measured.
 */
export function resampleLinear(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to || samples.length === 0) return samples.slice();
  const length = Math.max(1, Math.floor((samples.length * to) / from));
  const resampled = new Float32Array(length);
  const step = from / to;
  for (let index = 0; index < length; index += 1) {
    const position = index * step;
    const left = Math.floor(position);
    const right = Math.min(samples.length - 1, left + 1);
    const fraction = position - left;
    resampled[index] = (samples[left] ?? 0) * (1 - fraction) + (samples[right] ?? 0) * fraction;
  }
  return resampled;
}

function two(value: number): string {
  return String(value).padStart(2, '0');
}

/** `recording-20260914-101500.wav`, from the local clock. */
export function recordingFileName(at: Date): string {
  return (
    'recording-' +
    at.getFullYear() +
    two(at.getMonth() + 1) +
    two(at.getDate()) +
    '-' +
    two(at.getHours()) +
    two(at.getMinutes()) +
    two(at.getSeconds()) +
    '.wav'
  );
}

/**
 * A player clock, `m:ss`. Whole seconds, truncated like every media player's:
 * a position readout, not a measurement, and never a metric.
 */
export function clock(seconds: number): string {
  const whole = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
  return Math.floor(whole / 60) + ':' + two(whole % 60);
}
