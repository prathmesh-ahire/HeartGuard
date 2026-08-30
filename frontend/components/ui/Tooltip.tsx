'use client';

import * as RadixTooltip from '@radix-ui/react-tooltip';
import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';

/**
 * Mount once, near the root of whatever uses tooltips.
 */
export function TooltipProvider({ children }: { children: ReactNode }) {
  return (
    <RadixTooltip.Provider delayDuration={200} skipDelayDuration={300}>
      {children}
    </RadixTooltip.Provider>
  );
}

/**
 * A tooltip whose content is text, never a computed number. Anywhere a value
 * appears in a tooltip it is the pre-formatted `display` string from
 * `generated/`, passed in by the caller.
 */
export function Tooltip({
  children,
  content,
  side = 'top',
  className,
}: {
  children: ReactNode;
  content: ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
  className?: string;
}) {
  return (
    <RadixTooltip.Root>
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      <RadixTooltip.Portal>
        <RadixTooltip.Content
          side={side}
          sideOffset={6}
          collisionPadding={8}
          className={cn(
            'z-50 max-w-xs rounded border border-slate-700 bg-slate-900 px-2.5 py-1.5',
            'text-xs leading-relaxed text-slate-100 shadow-lg',
            'dark:border-slate-600 dark:bg-slate-800',
            className,
          )}
        >
          {content}
          <RadixTooltip.Arrow className="fill-slate-900 dark:fill-slate-800" />
        </RadixTooltip.Content>
      </RadixTooltip.Portal>
    </RadixTooltip.Root>
  );
}
