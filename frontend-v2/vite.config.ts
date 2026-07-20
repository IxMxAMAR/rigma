import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// Built assets land INSIDE the python package so the pip wheel ships them.
// The user's machine never runs Node; `npm run build` happens at dev/CI time
// and the dist is committed (same model as the legacy data/ui files).
export default defineConfig({
  base: "/v2/",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../src/rigma/data/ui_v2",
    emptyOutDir: true,
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:11500" },
  },
});
