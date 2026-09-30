import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
// `vitest/config` re-exports Vite's `defineConfig` with the `test` key typed,
// so this stays ONE config file for the build and the suite.
import { defineConfig } from "vitest/config";

// Built assets land INSIDE the python package so the pip wheel ships them.
// The user's machine never runs Node; `npm run build` happens at dev/CI time
// and the dist is committed (same model as the legacy data/ui files).
export default defineConfig({
  base: "/v2/",
  plugins: [react(), tailwindcss()],
  // The jsdom environment needs Node 25's broken global `localStorage` repaired
  // before any module reads it — see `src/test-setup.ts`. No new dependency.
  test: {
    setupFiles: ["./src/test-setup.ts"],
  },
  build: {
    outDir: "../src/rigma/data/ui_v2",
    emptyOutDir: true,
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:11500" },
  },
});
