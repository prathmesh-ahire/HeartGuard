'use client';

import { useCallback, useId, useRef, useState } from 'react';

import { AnimatePresence, LazyMotion, m } from 'framer-motion';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { LIFT_SPRING } from '@/lib/motion';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { Badge } from '@/components/ui/Badge';

const loadFeatures = () => import('@/components/motion/features').then((mod) => mod.default);

/**
 * The recording upload control (T111.3): drag-and-drop, format validation and
 * a progress state.
 *
 * Validation happens here so a wrong file is rejected before it reaches the
 * network, and the rejection SAYS WHAT WAS WRONG -- "that is a .mp3, this needs
 * a .wav" rather than a silent no-op. A control that quietly ignores a dropped
 * file reads as broken.
 *
 * The component owns no inference logic. It hands the caller a validated File
 * and renders whatever state the caller reports back.
 */

export const ACCEPTED_EXTENSIONS = ['.wav'] as const;
export const ACCEPTED_MIME = ['audio/wav', 'audio/x-wav', 'audio/wave'] as const;
export const MAX_BYTES = 50 * 1024 * 1024;

export type UploadPhase = 'idle' | 'validating' | 'uploading' | 'done' | 'error';

export function validateRecording(file: File): string | null {
  const name = file.name.toLowerCase();
  const extensionOk = ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension));
  if (!extensionOk) {
    return (
      `“${file.name}” is not a WAV file. This prototype reads uncompressed ` +
      `${ACCEPTED_EXTENSIONS.join(', ')} recordings only — a lossy format changes the ` +
      'spectral content the features are computed from.'
    );
  }
  if (file.size === 0) {
    return `“${file.name}” is empty (0 bytes).`;
  }
  if (file.size > MAX_BYTES) {
    return `“${file.name}” is larger than the ${MAX_BYTES / (1024 * 1024)} MB limit.`;
  }
  return null;
}

export function FileUpload({
  onFile,
  phase = 'idle',
  progress = null,
  error = null,
  fileName = null,
  className,
  disabled = false,
}: {
  onFile: (file: File) => void;
  phase?: UploadPhase;
  progress?: number | null;
  error?: string | null;
  fileName?: string | null;
  className?: string;
  disabled?: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const inputId = useId();

  const accept = useCallback(
    (file: File | undefined) => {
      if (file === undefined) return;
      const problem = validateRecording(file);
      setLocalError(problem);
      if (problem === null) onFile(file);
    },
    [onFile],
  );

  const shownError = error ?? localError;
  const busy = phase === 'uploading' || phase === 'validating';
  const reduced = useReducedMotion();

  return (
    <div className={className}>
      <LazyMotion features={loadFeatures} strict>
      <m.div
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!disabled) accept(event.dataTransfer.files[0]);
        }}
        // metric-guard: allow -- a drag-hover scale, not a measurement
        animate={reduced ? undefined : { scale: dragging ? 1.015 : 1 }}
        transition={LIFT_SPRING}
        className={cn(
          'rounded-lg border-2 border-dashed p-8 text-center transition-colors',
          dragging
            ? 'border-accent bg-accent-soft'
            : 'border-line',
          disabled && 'opacity-60',
        )}
      >
        <p className={cn(TYPE_SCALE.body, 'font-medium')}>
          Drop a heart-sound recording here
        </p>
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>
          Uncompressed WAV, up to {MAX_BYTES / (1024 * 1024)} MB. The audio is analysed on
          this computer and not kept; the result is saved to History.
        </p>

        <label
          htmlFor={inputId}
          className={cn(
            'mt-4 inline-block cursor-pointer rounded border border-line px-3 py-1.5',
            'text-sm font-medium hover:bg-sunken',
            '',
            disabled && 'pointer-events-none',
          )}
        >
          Choose a file
        </label>
        <input
          id={inputId}
          ref={input}
          type="file"
          className="sr-only"
          accept={[...ACCEPTED_EXTENSIONS, ...ACCEPTED_MIME].join(',')}
          disabled={disabled}
          onChange={(event) => accept(event.target.files?.[0])}
        />

        {fileName ? (
          <p className={cn(TYPE_SCALE.caption, 'mt-4 font-mono')}>{fileName}</p>
        ) : null}

        {busy ? (
          <div className="mt-4">
            <div
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progress ?? undefined}
              aria-label={phase === 'validating' ? 'Validating' : 'Uploading'}
              className="h-1.5 w-full overflow-hidden rounded bg-sunken"
            >
              {progress === null ? (
                // Indeterminate: the caller never reports a percentage for a
                // local inference call, so a fixed-width bar sat still for the
                // whole "Analysing…" phase. A short segment sweeps instead.
                <m.div
                  className="h-full w-1/3 rounded bg-accent"
                  // metric-guard: allow -- a sweep offset, not a measurement
                  animate={reduced ? { x: '100%' } : { x: ['-100%', '250%'] }}
                  transition={reduced ? { duration: 0 } : { duration: 1.1, ease: 'easeInOut', repeat: Infinity }}
                />
              ) : (
                <m.div
                  className="h-full bg-accent"
                  animate={{ width: `${progress}%` }}
                  transition={reduced ? { duration: 0 } : { duration: 0.25, ease: 'easeOut' }}
                />
              )}
            </div>
            <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>
              {phase === 'validating' ? 'Checking the file…' : 'Sending for inference…'}
            </p>
          </div>
        ) : null}

        <AnimatePresence>
          {phase === 'done' && !shownError ? (
            <m.p
              className="mt-4"
              initial={reduced ? undefined : { opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={reduced ? undefined : { opacity: 0, scale: 0.8 }}
              transition={LIFT_SPRING}
            >
              <Badge tone="good">Received</Badge>
            </m.p>
          ) : null}
        </AnimatePresence>
      </m.div>
      </LazyMotion>

      {shownError ? (
        <p
          role="alert"
          className={cn(
            TYPE_SCALE.caption,
            'mt-3 rounded border-2 border-danger-line bg-danger-soft p-3 text-danger',
          )}
        >
          {shownError}
        </p>
      ) : null}
    </div>
  );
}
