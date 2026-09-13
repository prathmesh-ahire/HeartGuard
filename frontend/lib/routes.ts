/**
 * The route tree (T127.3, T127.4).
 *
 * Declared once, here, and consumed by the navigation, the top bar, the About
 * tab bar and the Playwright smoke test, so a page that exists but is
 * unreachable -- or a nav link to a route that was never built -- is a type
 * error rather than a 404 somebody finds later.
 *
 * Five product pages. Everything about the project itself lives under About
 * the Model, as tabs. The old document routes still resolve: each one is a
 * redirect page to its new home, so a bookmark or a thesis reference never
 * lands on a 404.
 */
export interface RouteDefinition {
  /** The URL path. */
  href: string;
  /** Short label for the navigation. */
  label: string;
  /** What the page is for, used as the page description and the nav tooltip. */
  summary: string;
}

export const ROUTES: readonly RouteDefinition[] = [
  {
    href: '/',
    label: 'Analyse',
    summary: 'Upload or record a heart sound and get a screening result with its confidence.',
  },
  {
    href: '/history/',
    label: 'History',
    summary: 'Every recording you have analysed, with its result, notes and tags.',
  },
  {
    href: '/insights/',
    label: 'Insights',
    summary: 'Trends across your analyses: how many, what they found, and how confident.',
  },
  {
    href: '/reports/',
    label: 'Reports',
    summary: 'Download a report for a recording, or a summary of the model.',
  },
  {
    href: '/about/',
    label: 'About the Model',
    summary: 'How the model works, the data it learned from, how well it performs, and its limits.',
  },
];

/** The tabs of About the Model. The first is the About landing page itself. */
export const ABOUT_TABS: readonly RouteDefinition[] = [
  {
    href: '/about/',
    label: 'How it works',
    summary: 'The research objectives, the steps from a recording to a result, and how each recording is cleaned.',
  },
  {
    href: '/about/datasets/',
    label: 'Datasets',
    summary: 'The four public heart-sound collections the model learned from.',
  },
  {
    href: '/about/features/',
    label: 'Features',
    summary: 'The measurements taken from every recording, and which ones the model kept.',
  },
  {
    href: '/about/models/',
    label: 'Models',
    summary: 'The models compared, and how their settings were searched.',
  },
  {
    href: '/about/performance/',
    label: 'Performance',
    summary: 'How results hold up under harder conditions, and what the model relies on.',
  },
  {
    href: '/about/limitations/',
    label: 'Limitations',
    summary: 'What these results can and cannot support.',
  },
];

/**
 * Every retired route and its new home (T127.4). Each key has a page under
 * `app/` that does nothing but redirect here.
 *
 * `/reports/` is not in the list: the new Reports page lives at the same URL.
 */
export const LEGACY_REDIRECTS: Readonly<Record<string, string>> = {
  '/dataset/': '/about/datasets/',
  '/preprocessing/': '/about/',
  '/features/': '/about/features/',
  '/models/': '/about/models/',
  '/optimization/': '/about/models/',
  '/robustness/': '/about/performance/',
  '/explainability/': '/about/performance/',
  '/limitations/': '/about/limitations/',
  '/predict/binary/': '/',
  '/predict/multiclass/': '/',
  '/predict/murmur/': '/',
};

/**
 * The design reference (T127.5). Served by `next dev` only: its page file is
 * `page.design.tsx`, which `next.config.mjs` treats as a page outside a
 * production build. Never in the navigation.
 */
export const DESIGN_ROUTE = '/design/';

export function routeFor(href: string): RouteDefinition | undefined {
  return ROUTES.find((route) => route.href === href) ?? ABOUT_TABS.find((tab) => tab.href === href);
}

/** True when `pathname` is `href`, with or without its trailing slash. */
export function matchesRoute(pathname: string, href: string): boolean {
  return pathname === href || pathname === href.slice(0, -1) || `${pathname}/` === href;
}

/** The top-level page a pathname belongs to. */
export function sectionFor(pathname: string): RouteDefinition {
  const about = ROUTES[ROUTES.length - 1] as RouteDefinition;
  if (pathname.startsWith('/about')) return about;
  return ROUTES.find((route) => matchesRoute(pathname, route.href)) ?? (ROUTES[0] as RouteDefinition);
}
