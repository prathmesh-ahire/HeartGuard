/**
 * Display switches that are decisions, not configuration.
 *
 * `SHOW_SCREENING_NOTICE` (T127.2) hides the screening-only notice -- the
 * layout banner and the disclaimer line under every result -- for the
 * university presentation, at the user's request (2026-09-13). It is a
 * temporary exception to research rule 7, tracked as Open Item 17 in
 * `Docs/note.md`. T138.6 restores the notice by flipping this one value to
 * `true`; nothing else has to change.
 *
 * The API responses and every generated report still carry the disclaimer.
 * Only what the dashboard renders is switched.
 *
 * `src/reporting/display_audit.py` and `src/reporting/compliance.py` read this
 * literal, so keep it on one line in exactly this form.
 */
export const SHOW_SCREENING_NOTICE = false;
