import { Redirect } from '@/components/Redirect';
import { LEGACY_REDIRECTS, routeFor } from '@/lib/routes';

const to = LEGACY_REDIRECTS['/models/'] ?? '/';

/** A retired route (T127.4). Its content moved; this sends the reader to it. */
export default function Page() {
  return <Redirect to={to} label={routeFor(to)?.label ?? 'the new page'} />;
}
