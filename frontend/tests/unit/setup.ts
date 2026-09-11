import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

/*
 * jsdom lacks three browser APIs the components use. Each stub is the smallest
 * thing that lets the component run; none of them fakes a result.
 */

if (typeof window.matchMedia !== 'function') {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    }) as MediaQueryList;
}

if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
}

if (typeof URL.createObjectURL !== 'function') {
  URL.createObjectURL = () => 'blob:stub';
  URL.revokeObjectURL = () => undefined;
}

afterEach(() => cleanup());
