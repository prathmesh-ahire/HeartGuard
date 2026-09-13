/**
 * The icon set.
 *
 * Drawn inline rather than pulled from an icon font, because the export is a
 * static offline bundle: a Material Symbols <link> to fonts.googleapis.com
 * renders as tofu boxes the moment the page is opened without a network, and
 * the screenshot gate would capture that. Inline paths cost bytes once and
 * never fail to load.
 *
 * All paths are 24x24, stroke-based, and inherit `currentColor`, so an icon
 * takes the colour of the control it sits in.
 */
const PATHS = {
  /* Navigation */
  home: 'M3 10.5 12 3l9 7.5M5.25 9.5V20h13.5V9.5M9.75 20v-6h4.5v6',
  dataset: 'M4 6c0-1.66 3.58-3 8-3s8 1.34 8 3-3.58 3-8 3-8-1.34-8-3Zm0 0v12c0 1.66 3.58 3 8 3s8-1.34 8-3V6M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3',
  waveform: 'M2 12h3l2.5-7 3 15 3-11 2.5 6h6',
  features: 'M4 6h16M4 12h10M4 18h6M18 9.5v5M15.5 12h5',
  models: 'M9 3h6v3h3v12h-3v3H9v-3H6V6h3V3Zm0 6h6v6H9V9Z',
  optimization: 'M12 3v3m0 12v3M3 12h3m12 0h3m-2.5 0a6.5 6.5 0 1 1-13 0 6.5 6.5 0 0 1 13 0Zm-3.5 0a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z',
  robustness: 'M12 3 4.5 6v6c0 4.5 3.2 7.8 7.5 9 4.3-1.2 7.5-4.5 7.5-9V6L12 3Zm-3 9 2.2 2.2L15.5 10',
  explainability: 'M12 3a4 4 0 0 0-4 4 3.5 3.5 0 0 0-1 6.8V17a3 3 0 0 0 5 2.2A3 3 0 0 0 17 17v-3.2A3.5 3.5 0 0 0 16 7a4 4 0 0 0-4-4Zm0 0v18',
  reports: 'M6 3h8l4 4v14H6V3Zm8 0v4h4M9 12h6M9 16h6',
  binary: 'M3 12h4l2-5 3 10 2-7 1.5 2H21',
  multiclass: 'M12 3 3 7.5 12 12l9-4.5L12 3ZM3 12.5 12 17l9-4.5M3 17 12 21.5 21 17',
  murmur: 'M6 3v4a4 4 0 0 0 8 0V3M6 3H4.5M6 3h1.5M14 3h-1.5M14 3h1.5M10 11v3a5 5 0 0 0 5 5 4 4 0 0 0 4-4v-2m0 0a1.75 1.75 0 1 0 0-3.5 1.75 1.75 0 0 0 0 3.5Z',
  design: 'M12 3a9 9 0 0 0 0 18c1.1 0 2-.9 2-2 0-.5-.2-1-.5-1.3-.3-.4-.5-.8-.5-1.2 0-1 .8-1.8 1.8-1.8H17a4 4 0 0 0 4-4c0-4.4-4-8-9-8Zm-4.5 9a1.25 1.25 0 1 0 0-2.5 1.25 1.25 0 0 0 0 2.5Zm3-4a1.25 1.25 0 1 0 0-2.5 1.25 1.25 0 0 0 0 2.5Zm5 0a1.25 1.25 0 1 0 0-2.5 1.25 1.25 0 0 0 0 2.5Z',
  limitations: 'M12 4 2.5 20h19L12 4Zm0 6v5m0 3h.01',
  history: 'M3 12a9 9 0 1 0 2.64-6.36M3 4v4h4m5-1v5l3 2',
  insights: 'M4 20V11m5.33 9V5m5.34 15v-6M20 20V8M3 20h18',
  about: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0-10v5.5M12 7.75h.01',

  /* Controls and chrome */
  sun: 'M12 4V2m0 20v-2m8-8h2M2 12h2m13.66-5.66 1.41-1.41M4.93 19.07l1.41-1.41m11.32 0 1.41 1.41M4.93 4.93l1.41 1.41M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z',
  moon: 'M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z',
  menu: 'M4 7h16M4 12h16M4 17h16',
  close: 'M6 6l12 12M18 6 6 18',
  external: 'M14 4h6v6M20 4l-8 8M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5',
  upload: 'M12 16V4m0 0L8 8m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3',
  check: 'm5 12.5 4.5 4.5L19 7',
  pulse: 'M3 12h4l2-5 3 10 2-7 1.5 2H21',
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({
  name,
  className = 'h-4 w-4',
  strokeWidth = 1.6,
}: {
  name: IconName;
  className?: string;
  strokeWidth?: number;
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className={className}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
