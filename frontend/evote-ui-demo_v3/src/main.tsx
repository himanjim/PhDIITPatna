import { render } from "preact";
import { App } from "./app";
import { installMockBackend } from "./services/mockBackend";

/**
 * This demo ships with an in-browser mock backend to let you test the UI without Fabric/FAISS.
 * In production, remove this line and point the API base URL to your gateway.
 */
installMockBackend();

render(<App />, document.getElementById("app")!);
