import Link from 'next/link';

/**
 * The body of a retired route (T127.4).
 *
 * `redirect()` from `next/navigation` cannot be used here: under
 * `output: 'export'` it only writes a `NEXT_REDIRECT` marker into the RSC
 * payload, so the page moves only once JavaScript has run -- and a server
 * serving `out/` answers the old URL with an empty page. A meta refresh is
 * honoured by the browser on its own, before any script, and the link below it
 * covers anything that ignores both.
 */
export function Redirect({ to, label }: { to: string; label: string }) {
  return (
    <section className="max-w-2xl">
      <meta httpEquiv="refresh" content={'0; url=' + to} />
      <h1 className="text-headline-lg text-ink">This page has moved</h1>
      <p className="mt-3 text-body-lg text-ink-2">
        It is now part of{' '}
        <Link href={to} className="text-accent-strong underline underline-offset-2">
          {label}
        </Link>
        .
      </p>
    </section>
  );
}
