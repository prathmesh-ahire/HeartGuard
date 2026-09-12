import Link from 'next/link';

export default function NotFound() {
  return (
    <section className="max-w-2xl">
      <h1 className="text-2xl font-semibold tracking-tight">Page not found</h1>
      <p className="mt-3 text-ink-2">
        That route is not part of this site.{' '}
        <Link href="/" className="text-accent-strong hover:underline">
          Return to the overview
        </Link>
        .
      </p>
    </section>
  );
}
