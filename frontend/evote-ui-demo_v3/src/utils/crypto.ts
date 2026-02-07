/**
 * Small crypto utilities for receipt short codes, etc.
 * Uses browser SubtleCrypto (available in Chromium-based browsers).
 */

export async function sha256Hex(input: string): Promise<string> {
  const b = new TextEncoder().encode(input);
  const h = await crypto.subtle.digest("SHA-256", b);
  return [...new Uint8Array(h)].map(x => x.toString(16).padStart(2, "0")).join("");
}
