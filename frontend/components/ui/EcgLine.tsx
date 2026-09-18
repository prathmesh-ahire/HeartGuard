/**
 * A decorative heartbeat trace (T139.6): a flat line with one PQRST-shaped
 * spike, used as ambient texture behind a hero or along a panel edge.
 *
 * Pure SVG, no data behind it -- `aria-hidden`, and never placed anywhere it
 * could be mistaken for an actual recording's waveform (that is what
 * `WaveformPlayer` renders, from real samples).
 */
export function EcgLine({ className }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 400 60"
      preserveAspectRatio="none"
      className={className}
      fill="none"
    >
      <path
        d="M0 30 H140 L155 30 L165 10 L178 50 L188 5 L198 30 L215 30 H400"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
