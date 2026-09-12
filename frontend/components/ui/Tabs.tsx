'use client';

import * as RadixTabs from '@radix-ui/react-tabs';
import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';

/**
 * Radix underneath, so keyboard navigation, roving focus and the ARIA wiring
 * are correct rather than approximated.
 */
export interface TabItem {
  value: string;
  label: string;
  content: ReactNode;
}

export function Tabs({
  items,
  defaultValue,
  className,
  ariaLabel,
}: {
  items: TabItem[];
  defaultValue?: string;
  className?: string;
  ariaLabel: string;
}) {
  const first = items[0];
  if (first === undefined) return null;

  return (
    <RadixTabs.Root defaultValue={defaultValue ?? first.value} className={cn('w-full', className)}>
      <RadixTabs.List
        aria-label={ariaLabel}
        className="flex flex-wrap gap-1 border-b border-line"
      >
        {items.map((item) => (
          <RadixTabs.Trigger
            key={item.value}
            value={item.value}
            className={cn(
              'rounded-t px-3 py-2 text-sm text-ink-2 transition-colors',
              'hover:text-ink',
              'data-[state=active]:border-b-2 data-[state=active]:border-accent',
              'data-[state=active]:font-medium data-[state=active]:text-ink',
              '',
            )}
          >
            {item.label}
          </RadixTabs.Trigger>
        ))}
      </RadixTabs.List>
      {items.map((item) => (
        <RadixTabs.Content key={item.value} value={item.value} className="pt-4">
          {item.content}
        </RadixTabs.Content>
      ))}
    </RadixTabs.Root>
  );
}
