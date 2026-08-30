import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { SURFACE } from '@/lib/tokens';

/**
 * A surface. `glass` is translucent and belongs over the 3D hero; `solid` is
 * the default everywhere else, because a chart read through a blur is a chart
 * misread.
 */
export function GlassCard({
  children,
  className,
  variant = 'solid',
  as: Component = 'div',
}: {
  children: ReactNode;
  className?: string;
  variant?: 'solid' | 'glass' | 'sunken';
  as?: 'div' | 'section' | 'article' | 'aside';
}) {
  const surface =
    variant === 'glass' ? SURFACE.glass : variant === 'sunken' ? SURFACE.sunken : SURFACE.card;
  return <Component className={cn(surface, 'p-5', className)}>{children}</Component>;
}
