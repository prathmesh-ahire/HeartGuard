import type { Config } from 'tailwindcss';

/**
 * Design tokens for the PulseVision scheme (T128.1, T128.2).
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
 *
 * The layout scale follows the spacing of the Stitch reference project
 * ("PulseVision Clinical Dashboard"): a 4px base, 16 / 24 / 40px page margins
 * at phone / tablet / desktop, one content width. Shapes only -- nothing in
 * that project is a number this dashboard may show.
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
        /** The page ground (#F2E0D2), one step below every panel. */
        surface: withOpacity('--surface'),
        /** Card, panel and table ground. */
        panel: withOpacity('--panel'),
        /** Recessed ground: waveform beds, table headers, code blocks. */
        sunken: withOpacity('--sunken'),
        /** Floating ground: popovers, dropdowns, tooltips. */
        raised: withOpacity('--raised'),

        /** Rules and borders (#F2AFBC). `line-strong` is for focused controls. */
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
        /** Blush fill (#F9CBD6) behind active chips and selected rows. */
        'accent-soft': withOpacity('--accent-soft'),
        'accent-line': withOpacity('--accent-line'),
        'on-accent': withOpacity('--on-accent'),

        /** The veil behind an overlay. */
        scrim: withOpacity('--scrim'),

        /** Status, never brand: failure, a good outcome, a caution. */
        danger: withOpacity('--danger'),
        'danger-soft': withOpacity('--danger-soft'),
        'danger-line': withOpacity('--danger-line'),
        good: withOpacity('--good'),
        'good-soft': withOpacity('--good-soft'),
        'good-line': withOpacity('--good-line'),
        warn: withOpacity('--warn'),
        'warn-soft': withOpacity('--warn-soft'),
        'warn-line': withOpacity('--warn-line'),
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
        /* T128.2: the page title area. Larger and looser than a section head,
         * so each page opens with one clear statement of what it is for. */
        lede: ['16px', { lineHeight: '26px' }],
        title: ['30px', { lineHeight: '38px', letterSpacing: '-0.02em', fontWeight: '650' }],
        display: ['40px', { lineHeight: '48px', letterSpacing: '-0.02em', fontWeight: '700' }],
        /* T139.4: one fluid step for a hero-level heading, so the landing
         * page's headline scales with viewport width instead of jumping
         * between fixed breakpoints. `display` above stays fixed for section
         * heads that sit inside a normal content column. */
        hero: ['clamp(2.5rem, 1.9rem + 3vw, 4.5rem)', { lineHeight: '1.04', letterSpacing: '-0.025em', fontWeight: '700' }],
      },
      maxWidth: {
        /** One content width for every page (T128.2). */
        content: '76rem',
        /** A comfortable line length for prose. */
        reading: '42rem',
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
        panel: '0 1px 2px rgb(var(--scrim) / 0.04)',
        raised: '0 12px 32px -8px rgb(var(--scrim) / 0.22)',
        accent: '0 2px 10px rgb(var(--accent) / 0.28)',
        /* T139.3: the global card-hover glow, softer and wider than `accent`
         * above (which marks a pressed/active state, not a hover). */
        glow: '0 16px 40px -12px rgb(var(--accent) / 0.32)',
      },
      keyframes: {
        'pulse-subtle': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.45', transform: 'scale(0.96)' },
        },
        /* T128.3 / T128.5 -- entrances. Each is used only behind `motion-safe:`. */
        'overlay-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'modal-in': {
          from: { opacity: '0', transform: 'translateY(8px) scale(0.98)' },
          to: { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        'drawer-in-right': { from: { transform: 'translateX(100%)' }, to: { transform: 'translateX(0)' } },
        'drawer-in-left': { from: { transform: 'translateX(-100%)' }, to: { transform: 'translateX(0)' } },
        'sheet-in': { from: { transform: 'translateY(100%)' }, to: { transform: 'translateY(0)' } },
        skeleton: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
        /* T140.2: the hero's segmentation chips, each drifting up and down a
         * few pixels on its own phase offset (staggered via `animation-delay`
         * at the call site) so the set reads as afloat rather than synced. */
        'float-subtle': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-6px)' },
        },
      },
      animation: {
        'pulse-subtle': 'pulse-subtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'overlay-in': 'overlay-in 180ms ease-out',
        'modal-in': 'modal-in 220ms cubic-bezier(0.16, 1, 0.3, 1)',
        'drawer-in-right': 'drawer-in-right 280ms cubic-bezier(0.16, 1, 0.3, 1)',
        'drawer-in-left': 'drawer-in-left 280ms cubic-bezier(0.16, 1, 0.3, 1)',
        'sheet-in': 'sheet-in 280ms cubic-bezier(0.16, 1, 0.3, 1)',
        skeleton: 'skeleton 1.6s ease-in-out infinite',
        'float-subtle': 'float-subtle 4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};

export default config;
