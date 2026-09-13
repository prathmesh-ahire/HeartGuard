import { ButtonLink } from '@/components/ui/Button';
import { routeFor } from '@/lib/routes';

/**
 * The body for a page whose content has not been built yet.
 *
 * It says plainly that the page is not finished. A route that renders an empty
 * shell reads as a broken page; one that renders a plausible-looking
 * placeholder is worse, because it invites somebody to screenshot it. Neither
 * is acceptable, so the state is stated.
 *
 * This component contains no numbers at all, which is also why every such
 * route passes the metric guard rail unchanged.
 */
export function PagePlaceholder({ title, href }: { title: string; href: string }) {
  const route = routeFor(href);

  return (
    <section className="max-w-3xl">
      <h1 className="text-headline-lg text-ink">{title}</h1>
      {route ? <p className="mt-3 text-body-lg text-ink-2">{route.summary}</p> : null}
      <div className="mt-6 rounded-lg border border-dashed border-line bg-sunken p-4 text-body-md text-ink-2">
        <p className="font-medium text-ink">This page is not built yet.</p>
        <p className="mt-1">
          Nothing on it is a result, and no value here has been measured.
        </p>
      </div>
      <ButtonLink href="/" tone="primary" icon="pulse" className="mt-6">
        Analyse a recording
      </ButtonLink>
    </section>
  );
}
