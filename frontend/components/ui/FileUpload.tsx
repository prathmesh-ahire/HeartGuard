'use client';

import { useCallback, useId, useRef, useState } from 'react';

import { cn } from '@/lib/cn';
import { SURFACE, TYPE_SCALE } from '@/lib/tokens';
import { Badge } from '@/components/ui/Badge';

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

  return (
    <div className={className}>
      <div
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
        className={cn(
          'rounded-lg border-2 border-dashed p-8 text-center transition-colors',
          dragging
            ? 'border-sky-500 bg-sky-50 dark:border-sky-400 dark:bg-sky-950/30'
            : 'border-slate-300 dark:border-slate-700',
          disabled && 'opacity-60',
        )}
      >
        <p className={cn(TYPE_SCALE.body, 'font-medium')}>
          Drop a heart-sound recording here
        </p>
        <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-1')}>
          Uncompressed WAV, up to {MAX_BYTES / (1024 * 1024)} MB. The file is sent to the
          local inference service and is not stored.
        </p>

        <label
          htmlFor={inputId}
          className={cn(
            'mt-4 inline-block cursor-pointer rounded border border-slate-300 px-3 py-1.5',
            'text-sm font-medium hover:bg-slate-100',
            'dark:border-slate-700 dark:hover:bg-slate-800',
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
              className="h-1.5 w-full overflow-hidden rounded bg-slate-200 dark:bg-slate-800"
            >
              <div
                className="h-full bg-sky-600 transition-[width] dark:bg-sky-500"
                style={{ width: progress === null ? '35%' : `${progress}%` }}
              />
            </div>
            <p className={cn(TYPE_SCALE.caption, SURFACE.muted, 'mt-2')}>
              {phase === 'validating' ? 'Checking the file…' : 'Sending for inference…'}
            </p>
          </div>
        ) : null}

        {phase === 'done' && !shownError ? (
          <p className="mt-4">
            <Badge tone="good">Received</Badge>
          </p>
        ) : null}
      </div>

      {shownError ? (
        <p
          role="alert"
          className={cn(
            TYPE_SCALE.caption,
            'mt-3 rounded border-2 border-rose-400 bg-rose-50 p-3 text-rose-800',
            'dark:border-rose-700 dark:bg-rose-950/40 dark:text-rose-200',
          )}
        >
          {shownError}
        </p>
      ) : null}
    </div>
  );
}
