import { Header } from "./components/header";
import { Home } from "./pages/home";
import { NotFound } from "./pages/notfound";
import { A_Start } from "./pages/clientA/a_start";
import { A_Liveness } from "./pages/clientA/a_liveness";
import { A_Ballot } from "./pages/clientA/a_ballot";
import { A_Receipt } from "./pages/clientA/a_receipt";
import { A_End } from "./pages/clientA/a_end";
import { B_Enroll } from "./pages/clientB/b_enroll";
import { B_Verify } from "./pages/clientB/b_verify";
import { Router } from "./router";

/**
 * App shell that hosts both Client A and Client B routes.
 *
 * Production note:
 * - Client B should typically be served from a booth-only origin or at least behind verifier-only API gates.
 * - This unified build is intentional for demonstration: it shows how the UI behaves, while the *real* barrier
 *   is the verifier device credential + internal API restrictions, not obscurity of the route.
 */
export function App() {
  return (
    <div class="app">
      <Header />
      <main id="main">
        <Router
          routes={[
            { path: "/", element: () => <Home /> },

            // Client A
            { path: "/a/start", element: () => <A_Start /> },
            { path: "/a/liveness", element: () => <A_Liveness /> },
            { path: "/a/ballot", element: () => <A_Ballot /> },
            { path: "/a/receipt", element: () => <A_Receipt /> },
            { path: "/a/end", element: () => <A_End /> },

            // Client B
            { path: "/b/enroll", element: () => <B_Enroll /> },
            { path: "/b/verify", element: () => <B_Verify /> },
          ]}
          notFound={() => <NotFound />}
        />
      </main>
    </div>
  );
}
