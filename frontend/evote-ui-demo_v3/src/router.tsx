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

function normPath(p: string): string {
  if (!p) return "/";
  const q = p.split("?")[0].split("#")[0];
  return q.startsWith("/") ? q : `/${q}`;
}

export function currentPath(): string {
  const h = window.location.hash || "#/";
  return normPath(h.replace(/^#/, ""));
}

export function navigate(path: string) {
  window.location.hash = `#${normPath(path)}`;
}

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
