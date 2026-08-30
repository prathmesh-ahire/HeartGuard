/**
 * Next.js configuration for the PV-MEPCG / PulseVision dashboard.
 *
 * `output: 'export'` is the whole architecture in one line. The site is a
 * static export: no server, no runtime data fetching, no API routes. Every
 * precomputed number is baked in at build time by
 * `scripts/17_export_frontend_data.py`, and the only thing that crosses the
 * wire at runtime is `POST /predict` to the FastAPI service.
 *
 * `images.unoptimized` is required by `output: 'export'`: the default image
 * optimizer needs a Node server, which a static export does not have. The
 * figures are pre-rendered 300 dpi PNGs from matplotlib, so there is nothing
 * to optimize at request time anyway.
 *
 * `trailingSlash` makes every route emit `<route>/index.html`, which is what
 * lets the exported site be served by any static file server -- including
 * opening `out/index.html` from disk -- rather than needing rewrite rules.
 */
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  reactStrictMode: true,
  images: { unoptimized: true },
  eslint: { dirs: ['app', 'components', 'lib'] },
};

export default nextConfig;
