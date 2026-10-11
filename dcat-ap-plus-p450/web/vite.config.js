import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));

// The site renders ../docs/specifications/components.json in place; it never
// keeps a copy. The dev server must therefore be allowed to serve files from
// the whole dcat-ap-plus-p450 package, not just web/.
const packageRoot = path.resolve(here, '..');

export default defineConfig({
  plugins: [react()],
  server: {
    fs: { allow: [packageRoot] },
  },
});
