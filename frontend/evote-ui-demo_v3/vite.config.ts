/**
 * Vite build configuration for the eVote UI demo.
 *
 * The configuration keeps the development server deterministic by fixing the port,
 * enables the Preact integration, and targets a modern JavaScript runtime for the
 * production bundle. Operational security headers are intentionally left to the
 * serving layer rather than encoded in the development build configuration.
 */
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
