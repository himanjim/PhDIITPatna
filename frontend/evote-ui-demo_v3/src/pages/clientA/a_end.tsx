import { useEffect } from "preact/hooks";
import { t, useLang } from "../../i18n";
import { clearSessionState } from "../../state";
import { navigate } from "../../router";

/**
 * End-of-session page:
 * - Confirms that client-side state has been cleared
 * - In a supervised booth, the operator can start the next voter session
 */
export function A_End() {
  const { lang } = useLang();

  useEffect(() => {
    // Defensive clearing: ensure no previous voter data remains in memory.
    clearSessionState();

    // Kiosk hygiene: if you run this as a PWA/hardened browser in kiosk mode,
    // you can also call:
    //   - caches.keys().then(k => k.forEach(caches.delete))
    //   - sessionStorage.clear()
    // But do NOT clear localStorage preferences like language unless required by policy.
    try { sessionStorage.clear(); } catch {}
  }, []);

  return (
    <div class="card">
      <h1>{t(lang, "end_title")}</h1>
      <p>{t(lang, "end_note")}</p>
      <div class="hr"></div>
      <button class="primary" onClick={() => navigate("/a/start")}>Start next voter</button>
      <a class="badge" href="#/">Back to home</a>
    </div>
  );
}
