/**
 * Small browser-side cryptographic helper functions.
 *
 * The utilities in this file are intentionally limited to non-sensitive client-side
 * tasks, such as deterministic digest generation for receipt-facing values in the
 * demo environment. They rely on the browser Web Crypto API rather than external
 * libraries.
 */

/**
 * Compute a SHA-256 digest over the supplied UTF-8 text and return the result as a
 * lowercase hexadecimal string.
 */
export async function sha256Hex(input: string): Promise<string> {
  const b = new TextEncoder().encode(input);
  const h = await crypto.subtle.digest("SHA-256", b);
  return [...new Uint8Array(h)].map(x => x.toString(16).padStart(2, "0")).join("");
}
