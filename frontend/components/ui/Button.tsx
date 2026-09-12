import Link from 'next/link';
import type { ComponentProps, ReactNode } from 'react';

import { cn } from '@/lib/cn';
import { Icon, type IconName } from '@/components/ui/Icon';

/**
 * The control surface: one place that decides what a primary action looks
 * like, so a button on the dataset page and a button on a predict page cannot
 * disagree about it.
 *
 * Three tones, and they mean different things rather than looking different:
 * `primary` is the one action a screen is for, `secondary` is everything else,
 * `ghost` is a control that must not compete with the data beside it.
 */
const TONES = {
  primary:
    'border-accent-strong bg-accent text-on-accent shadow-accent hover:bg-accent-strong active:scale-[0.98]',
  secondary:
    'border-line bg-panel text-ink-2 hover:border-accent-line hover:text-accent-strong active:scale-[0.98]',
  ghost:
    'border-transparent bg-transparent text-ink-2 hover:bg-accent-soft hover:text-accent-strong',
} as const;

const SIZES = {
  sm: 'px-2.5 py-1 text-label-sm',
  md: 'px-3 py-1.5 text-label-md',
  lg: 'px-4 py-2 text-label-lg',
} as const;

export type ButtonTone = keyof typeof TONES;

function classes(tone: ButtonTone, size: keyof typeof SIZES, className?: string): string {
  return cn(
    'inline-flex items-center justify-center gap-2 rounded-lg border',
    'font-mono uppercase transition-all',
    TONES[tone],
    SIZES[size],
    className,
  );
}

export function Button({
  children,
  tone = 'secondary',
  size = 'md',
  icon,
  className,
  ...rest
}: {
  children: ReactNode;
  tone?: ButtonTone;
  size?: keyof typeof SIZES;
  icon?: IconName;
  className?: string;
} & Omit<ComponentProps<'button'>, 'className' | 'children'>) {
  return (
    <button type="button" className={classes(tone, size, className)} {...rest}>
      {icon ? <Icon name={icon} className="h-3.5 w-3.5 shrink-0" /> : null}
      {children}
    </button>
  );
}

export function ButtonLink({
  children,
  href,
  tone = 'secondary',
  size = 'md',
  icon,
  className,
}: {
  children: ReactNode;
  href: string;
  tone?: ButtonTone;
  size?: keyof typeof SIZES;
  icon?: IconName;
  className?: string;
}) {
  return (
    <Link href={href} className={classes(tone, size, className)}>
      {icon ? <Icon name={icon} className="h-3.5 w-3.5 shrink-0" /> : null}
      {children}
    </Link>
  );
}
