'use client';

import type { PredictResult } from '@/lib/api';

/**
 * The most recent live prediction in this browser tab (T117.2).
 *
 * The explainability page shows "the per-sample explanation for the last
 * prediction", and the prediction was made on another page. An upload is never
 * kept on the server, so the only honest hand-off is the response itself: the
 * prediction page stores what the API returned, and the explainability page
 * renders its `explanation` block exactly as the API formatted it.
 *
 * `sessionStorage`, not `localStorage`: a recording someone scored should not
 * outlive the tab they scored it in. Every access is guarded, because storage
 * can be unavailable (private windows, blocked site data) and a missing
 * hand-off must render as "nothing scored yet", never as a crash.
 */

const KEY = 'pv-mepcg:last-prediction';

/** Fired on `window` when a new prediction is stored, so an open page updates. */
export const LAST_PREDICTION_EVENT = KEY;

export interface StoredPrediction {
  /** When it was stored, as an ISO instant. */
  saved: string;
  result: PredictResult;
}

export function rememberPrediction(result: PredictResult): void {
  try {
    const stored: StoredPrediction = { saved: new Date().toISOString(), result };
    window.sessionStorage.setItem(KEY, JSON.stringify(stored));
    window.dispatchEvent(new Event(LAST_PREDICTION_EVENT));
  } catch {
    // Storage refused. The prediction page still shows its result; only the
    // hand-off to /explainability is lost, and that page says so.
  }
}

export function recallPrediction(): StoredPrediction | null {
  try {
    const raw = window.sessionStorage.getItem(KEY);
    if (raw === null) return null;
    const parsed = JSON.parse(raw) as Partial<StoredPrediction>;
    if (
      typeof parsed.saved !== 'string' ||
      typeof parsed.result !== 'object' ||
      parsed.result === null ||
      typeof parsed.result.predicted_class !== 'string'
    ) {
      return null;
    }
    return parsed as StoredPrediction;
  } catch {
    return null;
  }
}
