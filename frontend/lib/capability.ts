'use client';

import { useEffect, useState } from 'react';

/**
 * What the visitor's machine can actually do, and what they asked for (T112.6).
 *
 * Two separate questions, deliberately kept apart:
 *
 * * **Can it?** WebGL may be missing, blocklisted by the driver, or refused
 *   because the browser is out of GPU contexts. Integrated graphics on a
 *   laptop -- which is what this project is developed and marked on -- will run
 *   the scene but not at sixty frames.
 * * **Should it?** `prefers-reduced-motion: reduce` is an accessibility
 *   setting, not a performance hint. Someone who sets it may have a fast
 *   machine and still get vestibular symptoms from a scroll-driven animation.
 *
 * Both answers are `false` during server rendering and during the first client
 * render, so the markup the server produced and the markup React first produces
 * are identical. The scene appears on the effect pass. This is the difference
 * between a component that hydrates and one that logs a hydration mismatch and
 * throws away the server's HTML.
 */

export interface Capability {
  /** Resolved on the client only; false while server-rendering. */
  ready: boolean;
  webgl: boolean;
  reducedMotion: boolean;
  /** navigator.hardwareConcurrency, when the browser reports it. */
  cores: number | null;
  /** navigator.deviceMemory in GB, Chromium only. */
  memoryGb: number | null;
  /**
   * The scene is worth mounting: WebGL is available, motion was not refused,
   * and the machine is not obviously too small for it.
   */
  allow3d: boolean;
}

const INITIAL: Capability = {
  ready: false,
  webgl: false,
  reducedMotion: false,
  cores: null,
  memoryGb: null,
  allow3d: false,
};

/**
 * Probe for a WebGL context and release it immediately.
 *
 * The probe canvas is never attached to the document and the context is freed
 * with `WEBGL_lose_context`, because a browser allows a small number of live
 * contexts per page (often sixteen) and a probe that keeps one is a probe that
 * eventually prevents the thing it was testing for.
 */
export function detectWebgl(): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    const context =
      canvas.getContext('webgl2') ??
      canvas.getContext('webgl') ??
      canvas.getContext('experimental-webgl');
    if (context === null) return false;
    const gl = context as WebGLRenderingContext;
    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose !== null) lose.loseContext();
    return true;
  } catch {
    // A driver blocklist throws rather than returning null in some browsers.
    return false;
  }
}

export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function useCapability(): Capability {
  const [capability, setCapability] = useState<Capability>(INITIAL);

  useEffect(() => {
    const navigatorWithHints = navigator as Navigator & { deviceMemory?: number };
    const cores =
      typeof navigator.hardwareConcurrency === 'number' ? navigator.hardwareConcurrency : null;
    const memoryGb =
      typeof navigatorWithHints.deviceMemory === 'number' ? navigatorWithHints.deviceMemory : null;

    const read = (): void => {
      const webgl = detectWebgl();
      const reducedMotion = prefersReducedMotion();
      // Two cores or 2 GB is a phone or a throttled VM. The scene would run,
      // badly, and take the rest of the page down with it.
      const tooSmall = (cores !== null && cores <= 2) || (memoryGb !== null && memoryGb <= 2);
      setCapability({
        ready: true,
        webgl,
        reducedMotion,
        cores,
        memoryGb,
        allow3d: webgl && !reducedMotion && !tooSmall,
      });
    };

    read();

    // The setting can change while the page is open -- someone toggles it in
    // the OS, or a screen-reader profile switches it. Re-read rather than
    // trusting the value captured at mount.
    if (typeof window.matchMedia !== 'function') return;
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    query.addEventListener('change', read);
    return () => query.removeEventListener('change', read);
  }, []);

  return capability;
}

/** Just the accessibility half, for components with no 3D in them. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const read = (): void => setReduced(query.matches);
    read();
    query.addEventListener('change', read);
    return () => query.removeEventListener('change', read);
  }, []);

  return reduced;
}
