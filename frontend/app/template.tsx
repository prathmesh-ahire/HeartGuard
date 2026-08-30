import { PageTransition } from '@/components/motion/Reveal';

/**
 * Route transitions (T112.5).
 *
 * A template remounts on every navigation; a layout does not. The disclaimer,
 * navbar and footer stay in `layout.tsx` precisely because they must NOT
 * remount -- a banner that fades back in on every route change is a banner
 * people learn to ignore.
 */
export default function Template({ children }: { children: React.ReactNode }) {
  return <PageTransition>{children}</PageTransition>;
}
