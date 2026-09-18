import { cn } from '@/lib/cn';
import { LAYOUT } from '@/lib/tokens';

/**
 * The footer (T127.1).
 *
 * It used to carry the run manifest -- commit, branch, run id, export time and
 * the list of skipped source folders. That provenance is still checked on
 * every build, by `scripts/45_audit_displayed_values.py`, which confirms the
 * numbers on screen came from a recorded export run. It is simply no longer
 * printed for the person using the app, who has no use for a commit hash.
 *
 * Kept deliberately light (T140.6): it is rendered from the root layout, so
 * every one of the five routes pays for whatever it imports. The richer
 * "results summary, version, run id" footer T140.6 asks for is
 * `LandingFooterSummary`, scoped to `/` alone, for the same reason `landing`
 * itself is its own module and not in the barrel -- see that component.
 */
export function Footer() {
  return (
    <footer className="mt-10 border-t border-line text-body-sm text-ink-3">
      <div className={cn(LAYOUT.page, 'flex flex-wrap items-center justify-between gap-2 py-6')}>
        <p className="text-label-md uppercase text-ink-2">PV-MEPCG / PulseVision</p>
        <p>Phonocardiogram heart-sound analysis · academic research prototype</p>
      </div>
    </footer>
  );
}
