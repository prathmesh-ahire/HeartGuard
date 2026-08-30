import Link from 'next/link';

export default function NotFound() {
  return (
    <section className="max-w-2xl">
      <h1 className="text-2xl font-semibold tracking-tight">Page not found</h1>
      <p className="mt-3 text-slate-600 dark:text-slate-400">
        That route is not part of this site.{' '}
        <Link href="/" className="text-sky-700 hover:underline dark:text-sky-400">
          Return to the overview
        </Link>
        .
      </p>
    </section>
  );
}
