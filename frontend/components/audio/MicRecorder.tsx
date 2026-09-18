'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import { cn } from '@/lib/cn';
import { clock, encodeWav, recordingFileName } from '@/lib/wav';
import { Button } from '@/components/ui/Button';
import { TYPE_SCALE } from '@/lib/tokens';

/**
 * Record a heart sound from the microphone (T130.2).
 *
 * ## Raw PCM, written as WAV, handed over as a File
 *
 * `MediaRecorder` only produces lossy formats, which the service refuses. So
 * the microphone is read through Web Audio as raw samples, written as a 16-bit
 * WAV by `lib/wav.ts`, and passed to `onFile` -- the same callback the upload
 * control calls, so a recording is validated, previewed, scored, saved to
 * History and reported exactly like a dropped file. There is no second path.
 *
 * `ScriptProcessorNode` rather than an `AudioWorklet`: a worklet needs a
 * separate module file served beside the page, and the capture here is a copy
 * of each block, which the older node does in every browser.
 *
 * Browser audio processing (echo cancellation, noise suppression, automatic
 * gain) is switched off: it is tuned for speech and would filter a heart sound.
 *
 * ## Stops itself at the longest recording the models were fitted on
 *
 * `maxSeconds` comes from the generated payload, so a recording cannot run past
 * what the service will accept. Too short is refused here with the bound.
 */

type Phase = 'idle' | 'starting' | 'recording' | 'failed';

interface Capture {
  stream: MediaStream;
  context: AudioContext;
  source: MediaStreamAudioSourceNode;
  processor: ScriptProcessorNode;
  blocks: Float32Array[];
  frames: number;
}

export function MicRecorder({
  onFile,
  maxSeconds,
  minSeconds,
  minDisplay,
  disabled = false,
  className,
}: {
  onFile: (file: File) => void;
  maxSeconds: number;
  minSeconds: number;
  /** The minimum as the server formatted it, for the refusal message. */
  minDisplay: string;
  disabled?: boolean;
  className?: string;
}) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [elapsed, setElapsed] = useState(0);
  const [problem, setProblem] = useState<string | null>(null);
  const capture = useRef<Capture | null>(null);

  const release = useCallback(() => {
    const active = capture.current;
    capture.current = null;
    if (active === null) return;
    active.processor.onaudioprocess = null;
    active.source.disconnect();
    active.processor.disconnect();
    for (const track of active.stream.getTracks()) track.stop();
    void active.context.close();
  }, []);

  const finish = useCallback(
    (keep: boolean) => {
      const active = capture.current;
      release();
      setPhase('idle');
      setElapsed(0);
      if (!keep || active === null) return;
      const seconds = active.frames / active.context.sampleRate;
      if (seconds < minSeconds) {
        setProblem('That recording is too short. Record at least ' + minDisplay + '.');
        return;
      }
      const bytes = encodeWav(active.blocks, active.context.sampleRate);
      onFile(new File([bytes], recordingFileName(new Date()), { type: 'audio/wav' }));
    },
    [minDisplay, minSeconds, onFile, release],
  );

  useEffect(() => release, [release]);

  const start = useCallback(async () => {
    setProblem(null);
    const media = typeof navigator === 'undefined' ? undefined : navigator.mediaDevices;
    if (media?.getUserMedia === undefined) {
      setProblem('This browser cannot record from a microphone here. Upload a file instead.');
      setPhase('failed');
      return;
    }
    setPhase('starting');
    try {
      const stream = await media.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      const context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const active: Capture = { stream, context, source, processor, blocks: [], frames: 0 };
      processor.onaudioprocess = (event) => {
        const block = new Float32Array(event.inputBuffer.getChannelData(0));
        active.blocks.push(block);
        active.frames += block.length;
        const seconds = active.frames / context.sampleRate;
        setElapsed(seconds);
        if (seconds >= maxSeconds) finish(true);
      };
      source.connect(processor);
      processor.connect(context.destination); // a processor only runs while connected
      capture.current = active;
      setPhase('recording');
    } catch (error) {
      release();
      const denied = error instanceof DOMException && error.name === 'NotAllowedError';
      setProblem(
        denied
          ? 'Microphone access was blocked. Allow it in the browser, or upload a file instead.'
          : 'The microphone could not be started. ' + (error instanceof Error ? error.message : ''),
      );
      setPhase('failed');
    }
  }, [finish, maxSeconds, release]);

  return (
    <div className={className}>
      {phase === 'recording' ? (
        <div className="flex flex-wrap items-center gap-3">
          <span className="flex items-center gap-2 text-label-lg text-danger" aria-live="polite">
            <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-danger motion-safe:animate-pulse" />
            Recording {clock(elapsed)} / {clock(maxSeconds)}
          </span>
          <Button tone="primary" size="md" icon="check" onClick={() => finish(true)}>
            Stop and use
          </Button>
          <Button tone="ghost" size="md" icon="close" onClick={() => finish(false)}>
            Discard
          </Button>
        </div>
      ) : (
        <Button
          icon="mic"
          onClick={() => void start()}
          disabled={disabled || phase === 'starting'}
        >
          {phase === 'starting' ? 'Starting microphone…' : 'Record from microphone'}
        </Button>
      )}
      <p className={cn(TYPE_SCALE.caption, 'mt-2 text-ink-3')}>
        Hold the microphone or a digital stethoscope against the chest in a quiet room.
      </p>
      {problem !== null ? (
        <p role="alert" className={cn(TYPE_SCALE.caption, 'mt-2 rounded border border-danger-line bg-danger-soft p-2 text-danger')}>
          {problem}
        </p>
      ) : null}
    </div>
  );
}
