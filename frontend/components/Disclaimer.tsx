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
      className="flex items-start gap-2 border-b border-line bg-sunken px-4 py-1.5 text-body-sm leading-relaxed text-ink-2 lg:px-6"
    >
      <span className="mt-px shrink-0 rounded border border-accent-line bg-accent-soft px-1.5 py-0.5 font-mono text-label-sm uppercase text-accent-deep">
        Screening only
      </span>
      <span className="min-w-0">{DISCLAIMER_TEXT}</span>
    </div>
  );
}
