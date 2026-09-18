import { type ClassValue, clsx } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

/**
 * The custom font-size keys declared in `tailwind.config.ts`.
 *
 * tailwind-merge only knows Tailwind's built-in scale. Any other `text-*` class
 * it assumes is a text COLOUR, so `cn('text-on-accent', 'text-label-lg')` used
 * to decide the two conflicted and silently drop `text-on-accent` -- every
 * primary button rendered its label in body ink on the accent fill. Declaring
 * the keys as font sizes makes a size and a colour coexist, as they should.
 *
 * Keep this list in step with `theme.extend.fontSize`.
 */
const FONT_SIZES = [
  'label-sm',
  'label-md',
  'label-lg',
  'telemetry-sm',
  'telemetry',
  'body-sm',
  'body-md',
  'body-lg',
  'headline-sm',
  'headline-md',
  'headline-lg',
  'headline-xl',
  'hero',
];

const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      'font-size': [{ text: FONT_SIZES }],
    },
  },
});

/**
 * Compose Tailwind classes, letting a later class win over an earlier one that
 * targets the same property. Without the merge, `cn('p-2', 'p-4')` emits both
 * and the winner depends on stylesheet order rather than on the caller.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
