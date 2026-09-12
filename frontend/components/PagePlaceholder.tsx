import { routeFor } from '@/lib/routes';

/**
 * The scaffold body for a route whose content has not been built yet
 * (Phase 110, T110.4).
 *
 * It says plainly that the page is not finished, and names the phase that will
 * finish it. A route that renders an empty shell reads as a broken page; one
 * that renders a plausible-looking placeholder is worse, because it invites
 * somebody to screenshot it. Neither is acceptable, so the state is stated.
 *
 * This component contains no numbers at all, which is also why every scaffold
 * route passes the metric guard rail unchanged.
 */
export function PagePlaceholder({
  title,
  href,
  phase,
}: {
  title: string;
  href: string;
  phase: string;
}) {
  const route = routeFor(href);

  return (
    <section className="max-w-3xl">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {route ? (
        <p className="mt-3 text-ink-2">{route.summary}</p>
      ) : null}
      <div className="mt-6 rounded border border-dashed border-line bg-sunken p-4 text-sm text-ink-2">
        <p className="font-medium text-ink">
          Route scaffolded; content not built yet.
        </p>
        <p className="mt-1">
          This page is part of the route tree defined in Phase 110. Its content is
          built in {phase}. Nothing on it is a result, and no value here has been
          measured.
        </p>
      </div>
    </section>
  );
}
