import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

/**
 * Vite + Preact build.
 * NOTE on security headers:
 *  - In production, serve behind an HTTP server that sets a strict Content-Security-Policy (CSP),
 *    HSTS, X-Content-Type-Options, Referrer-Policy, etc.
 *  - During local dev, CSP is intentionally not enforced in index.html because Vite injects dev scripts.
 */
export default defineConfig({
  plugins: [preact()],
  server: {
    port: 5173,
    strictPort: true
  },
  build: {
    target: "es2022"
  }
});
