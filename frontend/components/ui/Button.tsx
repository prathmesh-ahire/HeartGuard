'use client';

import Link from 'next/link';
import type { ComponentProps, ReactNode } from 'react';

import { LazyMotion, m } from 'framer-motion';

import { useReducedMotion } from '@/lib/capability';
import { cn } from '@/lib/cn';
import { PRESS_SPRING } from '@/lib/motion';
import { Icon, type IconName } from '@/components/ui/Icon';

const loadFeatures = () => import('@/components/motion/features').then((mod) => mod.default);

/**
 * The control surface: one place that decides what a primary action looks
 * like, so a button on the dataset page and a button on a predict page cannot
 * disagree about it.
 *
 * Three tones, and they mean different things rather than looking different:
 * `primary` is the one action a screen is for, `secondary` is everything else,
 * `ghost` is a control that must not compete with the data beside it.
 *
 * The press feedback (T136.4) is a spring on `scale`, not the earlier
 * `active:scale-[0.98]` CSS rule -- the two would otherwise fight, since an
 * inline style from Framer always wins over a Tailwind utility class of equal
 * specificity, so the CSS rule now only exists implicitly (removed below).
 */
const TONES = {
  primary: 'border-accent-strong bg-accent text-on-accent shadow-accent hover:bg-accent-strong',
  secondary: 'border-line bg-panel text-ink-2 hover:border-accent-line hover:text-accent-strong',
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

const MotionLink = m(Link);

export function Button({
  children,
  tone = 'secondary',
  size = 'md',
  icon,
  className,
  disabled,
  ...rest
}: {
  children: ReactNode;
  tone?: ButtonTone;
  size?: keyof typeof SIZES;
  icon?: IconName;
  className?: string;
} & Omit<
  ComponentProps<'button'>,
  // React's native DOM event types and Framer's MotionProps declare these
  // five with incompatible signatures (a DOM AnimationEvent vs Framer's own
  // AnimationDefinition); `m.button` needs its own (which nothing here sets).
  'className' | 'children' | 'onAnimationStart' | 'onAnimationEnd' | 'onDrag' | 'onDragStart' | 'onDragEnd'
>) {
  const reduced = useReducedMotion();
  const press = !reduced && disabled !== true;
  return (
    <LazyMotion features={loadFeatures} strict>
      <m.button
        type="button"
        disabled={disabled}
        className={classes(tone, size, className)}
        whileHover={press ? { scale: 1.03 } : undefined}
        whileTap={press ? { scale: 0.96 } : undefined}
        whileFocus={press ? { scale: 1.02 } : undefined}
        transition={PRESS_SPRING}
        {...rest}
      >
        {icon ? <Icon name={icon} className="h-3.5 w-3.5 shrink-0" /> : null}
        {children}
      </m.button>
    </LazyMotion>
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
  const reduced = useReducedMotion();
  return (
    <LazyMotion features={loadFeatures} strict>
      <MotionLink
        href={href}
        className={classes(tone, size, className)}
        whileHover={reduced ? undefined : { scale: 1.03 }}
        whileTap={reduced ? undefined : { scale: 0.96 }}
        whileFocus={reduced ? undefined : { scale: 1.02 }}
        transition={PRESS_SPRING}
      >
        {icon ? <Icon name={icon} className="h-3.5 w-3.5 shrink-0" /> : null}
        {children}
      </MotionLink>
    </LazyMotion>
  );
}
