/**
 * Does a payload string talk about the repository rather than the result
 * (T127.1)?
 *
 * Several generated captions and notes were written for the thesis and carry
 * provenance -- a module path, a CSV name, a test file. The product UI shows
 * none of that, so a component that renders free text from a payload asks this
 * first and leaves such a string out (or shows its own plain fallback).
 *
 * This is a presentation filter, not the enforcement. The enforcement is the
 * `internal_info` check in `src/reporting/display_audit.py`, which crawls the
 * built HTML with the same patterns and fails the build on a hit; keep the two
 * in step.
 */
const REPO_PATH =
  /\b(?:outputs|scripts|src|configs|models_saved|cache|dataset|Docs|frontend|lib\/generated)\/[\w.\-/]*|\b[A-Za-z]:[\\/]/;
const FILE_NAME = /\b[\w-]+\.(?:py|ipynb|parquet|joblib|ya?ml|csv|json|tsx?)\b/;
const INTERNAL_WORDS = /\brun manifest\b|\brun id\b|\bgit commit\b|\bexporter\b|\bbuild-time export\b|\bgenerated\//i;

export function mentionsInternals(text: string | null | undefined): boolean {
  if (!text) return false;
  return REPO_PATH.test(text) || FILE_NAME.test(text) || INTERNAL_WORDS.test(text);
}

/** The string itself, or `null` when it carries internal information. */
export function userFacing(text: string | null | undefined): string | null {
  return text && !mentionsInternals(text) ? text : null;
}
