import { fileURLToPath } from 'node:url';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

/**
 * Component unit tests (T118.1). jsdom, React Testing Library, and the real
 * `lib/generated/` payloads -- a component test fed an invented table proves
 * nothing about the tables this dashboard actually renders.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./', import.meta.url)) },
  },
  test: {
    environment: 'jsdom',
    include: ['tests/unit/**/*.test.{ts,tsx}'],
    setupFiles: ['tests/unit/setup.ts'],
    restoreMocks: true,
  },
});
