import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { FileUpload, MAX_BYTES, validateRecording } from '@/components/ui/FileUpload';

/**
 * T118.1 / T118.4, client half: the upload control refuses a wrong format, an
 * empty file and an oversized file BEFORE anything reaches the network, and it
 * says which of the three it was. A corrupt WAV and a too-long recording pass
 * this control -- only the server can decode them -- and are covered end to end
 * in `e2e/upload.spec.ts` against the real API.
 */

function wav(name: string, bytes: number): File {
  return new File([new Uint8Array(bytes)], name, { type: 'audio/wav' });
}

function choose(file: File): void {
  const input = document.querySelector('input[type="file"]');
  if (!(input instanceof HTMLInputElement)) throw new Error('no file input rendered');
  fireEvent.change(input, { target: { files: [file] } });
}

describe('validateRecording', () => {
  it('refuses a lossy format and names the file', () => {
    const problem = validateRecording(new File(['x'], 'heart.mp3', { type: 'audio/mpeg' }));
    expect(problem).toMatch(/heart\.mp3/);
    expect(problem).toMatch(/not a WAV file/);
  });

  it('refuses an empty WAV', () => {
    expect(validateRecording(wav('empty.wav', 0))).toMatch(/empty \(0 bytes\)/);
  });

  it('refuses a WAV over the size limit', () => {
    const big = wav('big.wav', 1);
    Object.defineProperty(big, 'size', { value: MAX_BYTES + 1 });
    expect(validateRecording(big)).toMatch(/larger than the/);
  });

  it('accepts a non-empty WAV regardless of case', () => {
    expect(validateRecording(wav('Record.WAV', 44))).toBeNull();
  });
});

describe('FileUpload', () => {
  it('shows a visible alert and does not hand over a wrong-format file', () => {
    const onFile = vi.fn();
    render(<FileUpload onFile={onFile} />);
    choose(new File(['x'], 'clip.ogg', { type: 'audio/ogg' }));
    expect(screen.getByRole('alert').textContent).toMatch(/not a WAV file/);
    expect(onFile).not.toHaveBeenCalled();
  });

  it('shows a visible alert for an empty file', () => {
    const onFile = vi.fn();
    render(<FileUpload onFile={onFile} />);
    choose(wav('empty.wav', 0));
    expect(screen.getByRole('alert').textContent).toMatch(/0 bytes/);
    expect(onFile).not.toHaveBeenCalled();
  });

  it('hands a valid WAV to the caller and shows no alert', () => {
    const onFile = vi.fn();
    render(<FileUpload onFile={onFile} />);
    const file = wav('ok.wav', 128);
    choose(file);
    expect(onFile).toHaveBeenCalledWith(file);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('renders a server error it is given, so a failed upload is never silent', () => {
    render(<FileUpload onFile={() => undefined} phase="error" error="the recording is too long" />);
    expect(screen.getByRole('alert').textContent).toBe('the recording is too long');
  });

  it('shows progress while uploading', () => {
    render(<FileUpload onFile={() => undefined} phase="uploading" />);
    expect(screen.getByRole('progressbar', { name: 'Uploading' })).toBeTruthy();
  });
});
