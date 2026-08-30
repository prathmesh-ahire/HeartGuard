/**
 * The screening-only disclaimer (research rule 7, T110.3).
 *
 * Rendered in the ROOT LAYOUT, not per page. Per-page placement is how a
 * disclaimer goes missing: a page added later simply does not get one, and
 * nothing fails. Here it is structurally impossible for a route to render
 * without it.
 *
 * The wording is deliberately plain and contains no diagnostic language -- no
 * "diagnose", no "detect disease", no "replaces". This project is an academic
 * screening and decision-support prototype.
 */
export const DISCLAIMER_TEXT =
  'PV-MEPCG / PulseVision is an academic research prototype for screening and ' +
  'decision support. It is not a medical device, it does not diagnose, and it ' +
  'must not be used to make or replace any clinical decision. Every result ' +
  'shown here comes from public research datasets and is reported for ' +
  'methodological evaluation only.';

export function DisclaimerBanner() {
  return (
    <div
      role="note"
      aria-label="Scope and safety notice"
      className="border-b border-amber-300 bg-amber-50 px-4 py-2 text-center text-xs leading-relaxed text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-200"
    >
      <span className="font-semibold">Screening and research use only.</span>{' '}
      {DISCLAIMER_TEXT}
    </div>
  );
}
