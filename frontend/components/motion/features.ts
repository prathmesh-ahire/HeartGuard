/**
 * Framer Motion's DOM feature set, in its own module so it can be code-split.
 *
 * Importing `motion` pulls the whole animation feature set into whatever chunk
 * imports it -- 34 kB gzipped, paid by every route, because the page transition
 * lives in `app/template.tsx` and every route has a template. `LazyMotion` with
 * an async `features` loader defers all of that: the initial bundle carries only
 * the ~5 kB `m` component, and the features arrive with the first animation.
 *
 * It has to be a separate file. `() => import('framer-motion')` inside the
 * component would re-import the module the component is already in, and the
 * bundler would keep it eager.
 */
export { domAnimation as default } from 'framer-motion';
