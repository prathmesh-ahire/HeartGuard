import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Compose Tailwind classes, letting a later class win over an earlier one that
 * targets the same property. Without the merge, `cn('p-2', 'p-4')` emits both
 * and the winner depends on stylesheet order rather than on the caller.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
