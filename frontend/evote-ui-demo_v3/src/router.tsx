/**
 * Minimal hash-based router for the demo application.
 *
 * The router avoids external routing dependencies and works without server-side
 * rewrite rules, which makes the build easier to host in constrained or kiosk-like
 * environments. The use of explicit hash routes also keeps operator-visible paths
 * simple during demonstrations and controlled booth deployment.
 */

import { ComponentChildren } from "preact";
import { useEffect, useMemo, useState } from "preact/hooks";

/**
 * Ultra-light hash router.
 *
 * Rationale (kiosk + remote):
 * - Keeps deployment simple (single entry URL; no server rewrite rules)
 * - Avoids adding routing dependencies
 * - Uses explicit #/ paths which are easy for operators to type/open
 */
export type RouteDef = { path: string; element: () => ComponentChildren };

/**
 * Canonicalise a route string to the internal path form used by the router.
 * Query strings and nested hash fragments are stripped so that route matching is
 * performed only on the stable application path.
 */
function normPath(p: string): string {
  if (!p) return "/";
  const q = p.split("?")[0].split("#")[0];
  return q.startsWith("/") ? q : `/${q}`;
}

/**
 * Read the current application path from the browser hash.
 */
export function currentPath(): string {
  const h = window.location.hash || "#/";
  return normPath(h.replace(/^#/, ""));
}

/**
 * Move the application to a new hash route using the same path normalisation rule
 * that is applied during route matching.
 */
export function navigate(path: string) {
  window.location.hash = `#${normPath(path)}`;
}

/**
 * Route dispatcher component.
 *
 * The component listens for hash changes, resolves the current path against the
 * supplied route table, and renders the matched element or the provided not-found
 * view when no match exists.
 */
export function Router(props: { routes: RouteDef[]; notFound: () => ComponentChildren }) {
  const [path, setPath] = useState<string>(() => currentPath());

  useEffect(() => {
    const onHash = () => setPath(currentPath());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const match = useMemo(() => {
    const p = normPath(path);
    return props.routes.find((r) => r.path === p);
  }, [path, props.routes]);

  return <>{match ? match.element() : props.notFound()}</>;
}
