/**
 * Browser entry point for the demo application.
 *
 * The application installs an in-browser mock backend before rendering so that
 * the full UI flow can be exercised without a running gateway, ledger, or
 * biometric service. This is a demonstration convenience only. In a production
 * build, the mock layer should be removed and the UI should communicate with the
 * real backend interfaces.
 */

import { render } from "preact";
import { App } from "./app";
import { installMockBackend } from "./services/mockBackend";

/**
 * This demo ships with an in-browser mock backend to let you test the UI without Fabric/FAISS.
 * In production, remove this line and point the API base URL to your gateway.
 */
installMockBackend();

render(<App />, document.getElementById("app")!);
