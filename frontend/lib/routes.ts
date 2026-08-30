/**
 * The route tree (T110.4): home plus the eleven document pages.
 *
 * Declared once, here, and consumed by the navbar, the footer and the
 * Playwright smoke test (T118.2), so a page that exists but is unreachable --
 * or a nav link to a route that was never built -- is a type error rather than
 * a 404 somebody finds later.
 *
 * `/design` (T111.6) and `/limitations` (T117.5) are added by the phases that
 * build them; they are not document pages.
 */
export interface RouteDefinition {
  /** The URL path. */
  href: string;
  /** Short label for the navbar. */
  label: string;
  /** What the page is for, used as the page description and the nav tooltip. */
  summary: string;
  /** Navbar grouping. */
  group: 'overview' | 'method' | 'results' | 'predict';
}

export const ROUTES: readonly RouteDefinition[] = [
  {
    href: '/',
    label: 'Home',
    summary:
      'The framework, the six research objectives, and what this prototype is and is not.',
    group: 'overview',
  },
  {
    href: '/dataset/',
    label: 'Dataset',
    summary:
      'The four public PCG corpora as audited against the files on disk: inventory, class balance, durations and the fold map.',
    group: 'overview',
  },
  {
    href: '/preprocessing/',
    label: 'Preprocessing',
    summary:
      'Resampling, band-pass filtering, normalization and the signal-quality assessment applied before any feature is computed.',
    group: 'method',
  },
  {
    href: '/features/',
    label: 'Features',
    summary:
      'The locked 138-feature registry across six families, and the subset selected inside the training folds.',
    group: 'method',
  },
  {
    href: '/models/',
    label: 'Models',
    summary:
      'The declared models, their configuration, and the fold-wise comparison between them.',
    group: 'results',
  },
  {
    href: '/optimization/',
    label: 'Optimization',
    summary:
      'The search space, the methods compared, the parameters selected and what the search actually bought.',
    group: 'results',
  },
  {
    href: '/robustness/',
    label: 'Robustness',
    summary:
      'How results move under added noise, shortened recordings, a different corpus and a different auscultation location.',
    group: 'results',
  },
  {
    href: '/explainability/',
    label: 'Explainability',
    summary:
      'Global feature importance, family-level contribution, and a per-sample explanation of the most recent prediction.',
    group: 'results',
  },
  {
    href: '/reports/',
    label: 'Reports',
    summary:
      'Generated reports, and the evidence browser linking every displayed value to the CSV it came from.',
    group: 'results',
  },
  {
    href: '/predict/binary/',
    label: 'Binary',
    summary:
      'Upload a recording for a normal / abnormal screening indication with its confidence.',
    group: 'predict',
  },
  {
    href: '/predict/multiclass/',
    label: 'Multiclass',
    summary:
      'PASCAL A four-class and PASCAL B three-class acoustic-event output with the full probability distribution.',
    group: 'predict',
  },
  {
    href: '/predict/murmur/',
    label: 'Murmur / Outcome',
    summary:
      'CirCor murmur and clinical-outcome output at recording and patient level, with the per-location breakdown.',
    group: 'predict',
  },
];

export const GROUP_LABELS: Record<RouteDefinition['group'], string> = {
  overview: 'Overview',
  method: 'Method',
  results: 'Results',
  predict: 'Prediction',
};

export function routeFor(href: string): RouteDefinition | undefined {
  return ROUTES.find((route) => route.href === href);
}
