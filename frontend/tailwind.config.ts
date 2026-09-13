import type { Config } from 'tailwindcss';

/**
 * Design tokens for the rose clinical workstation theme.
 *
 * Colour is declared as CSS custom properties in `globals.css` and referenced
 * here through `rgb(var(--x) / <alpha-value>)`, for one reason: the same
 * utility class has to resolve to a light-mode value and a dark-mode value
 * without the component knowing which mode it is in. `bg-panel` is written
 * once; `.dark` redefines what "panel" means.
 *
 * Chart series colours are NOT here. They come from `generated/theme.json` via
 * `lib/tokens.ts`, shared with the matplotlib figures -- see the note there.
 *
 * `darkMode: 'class'` rather than 'media', because next-themes toggles a class
 * on <html> and needs Tailwind to follow it rather than the OS setting.
 */
const withOpacity = (variable: string) => `rgb(var(${variable}) / <alpha-value>)`;

const config: Config = {
  darkMode: 'class',
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        /** The page ground, one step below every panel. */
        surface: withOpacity('--surface'),
        /** Card, panel and table ground. */
        panel: withOpacity('--panel'),
        /** Recessed ground: waveform beds, table headers, code blocks. */
        sunken: withOpacity('--sunken'),
        /** Floating ground: popovers, dropdowns, tooltips. */
        raised: withOpacity('--raised'),

        /** Hairline structural rules. `line-strong` is for focused controls. */
        line: withOpacity('--line'),
        'line-strong': withOpacity('--line-strong'),

        /** Text, in descending emphasis. */
        ink: withOpacity('--ink'),
        'ink-2': withOpacity('--ink-2'),
        'ink-3': withOpacity('--ink-3'),

        /** Wine accent (#9E182B). Primary actions, active nav, live indicators. */
        accent: withOpacity('--accent'),
        'accent-strong': withOpacity('--accent-strong'),
        'accent-deep': withOpacity('--accent-deep'),
        /** Blush wash behind active chips and selected rows. */
        'accent-soft': withOpacity('--accent-soft'),
        'accent-line': withOpacity('--accent-line'),
        'on-accent': withOpacity('--on-accent'),

        /** Failure. Deliberately not the accent -- see globals.css. */
        danger: withOpacity('--danger'),
        'danger-soft': withOpacity('--danger-soft'),
        'danger-line': withOpacity('--danger-line'),
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      fontSize: {
        /* Monospace instrument labels: uppercase, positive tracking. */
        'label-sm': ['10px', { lineHeight: '12px', letterSpacing: '0.06em', fontWeight: '500' }],
        'label-md': ['11px', { lineHeight: '14px', letterSpacing: '0.04em', fontWeight: '600' }],
        'label-lg': ['13px', { lineHeight: '18px', letterSpacing: '0.03em', fontWeight: '600' }],
        /* Numeric readouts. Tabular figures are applied by the .stat class. */
        'telemetry-sm': ['14px', { lineHeight: '18px', fontWeight: '500' }],
        telemetry: ['28px', { lineHeight: '32px', letterSpacing: '-0.03em', fontWeight: '700' }],
        /* Prose and headings. */
        'body-sm': ['12px', { lineHeight: '16px' }],
        'body-md': ['13px', { lineHeight: '18px' }],
        'body-lg': ['15px', { lineHeight: '22px' }],
        'headline-sm': ['16px', { lineHeight: '22px', letterSpacing: '-0.01em', fontWeight: '600' }],
        'headline-md': ['18px', { lineHeight: '24px', letterSpacing: '-0.01em', fontWeight: '600' }],
        'headline-lg': ['24px', { lineHeight: '32px', letterSpacing: '-0.015em', fontWeight: '600' }],
        'headline-xl': ['32px', { lineHeight: '40px', letterSpacing: '-0.02em', fontWeight: '700' }],
      },
      borderRadius: {
        /* Instrument geometry: tighter than the Tailwind defaults at every
         * step. Chips and inputs stay stamp-sharp, panels stay contained. */
        DEFAULT: '0.25rem',
        md: '0.3125rem',
        lg: '0.375rem',
        xl: '0.5rem',
        '2xl': '0.75rem',
      },
      spacing: {
        rail: '15rem',
        topbar: '3.5rem',
      },
      boxShadow: {
        panel: '0 1px 2px rgb(15 23 42 / 0.04)',
        raised: '0 4px 12px -2px rgb(15 23 42 / 0.08)',
        accent: '0 2px 10px rgb(var(--accent) / 0.28)',
      },
      keyframes: {
        'pulse-subtle': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.45', transform: 'scale(0.96)' },
        },
      },
      animation: {
        'pulse-subtle': 'pulse-subtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
};

export default config;
