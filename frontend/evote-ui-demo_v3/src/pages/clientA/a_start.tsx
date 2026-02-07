import { useState } from "preact/hooks";
import { t, useLang } from "../../i18n";
import { api, Mode } from "../../services/api";
import { state, clearSessionState } from "../../state";
import { navigate } from "../../router";

/**
 * Client A start page:
 * - Select mode (remote vs kiosk)
 * - Collect minimal session start inputs (voterId, constituencyId)
 * - Kiosk mode includes officer PIN (placeholder for stronger auth)
 */
export function A_Start() {
  const { lang } = useLang();
  const [mode, setMode] = useState<Mode>("remote");
  const [voterId, setVoterId] = useState("");
  const [constituencyId, setConstituencyId] = useState("PATNA-01");
  const [officerPin, setOfficerPin] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  async function start() {
    setErr("");
    setBusy(true);
    clearSessionState(); // ensure no prior voter state remains in-memory

    try {
      const resp = await api.startSession({
        mode,
        voterId: voterId.trim(),
        constituencyId: constituencyId.trim(),
        officerPin: mode === "kiosk" ? officerPin.trim() : undefined
      });
      state.session = resp;

      // Move to liveness capture next.
      navigate("/a/liveness");
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1>{t(lang, "start_session")}</h1>

      <div class="card">
        <label for="modeSel">{t(lang, "mode")}</label>
        <select id="modeSel" value={mode} onChange={(e) => setMode((e.target as HTMLSelectElement).value as any)}>
          <option value="remote">{t(lang, "mode_remote")}</option>
          <option value="kiosk">{t(lang, "mode_kiosk")}</option>
        </select>

        <label for="voter">{t(lang, "voter_id")}</label>
        <input id="voter" value={voterId} onInput={(e) => setVoterId((e.target as HTMLInputElement).value)} placeholder="e.g., EPIC/UUID" />

        <label for="consti">{t(lang, "constituency")}</label>
        <input id="consti" value={constituencyId} onInput={(e) => setConstituencyId((e.target as HTMLInputElement).value)} placeholder="e.g., PATNA-01" />

        {mode === "kiosk" && (
          <>
            <label for="pin">{t(lang, "officer_pin")}</label>
            <input id="pin" type="password" value={officerPin} onInput={(e) => setOfficerPin((e.target as HTMLInputElement).value)} placeholder="demo: any 4+ digits" />
            <div class="small">In production, replace PIN with phishing-resistant officer authentication (e.g., passkey/FIDO2) + separation-of-duty policy.</div>
          </>
        )}

        <div class="hr"></div>
        <button class="primary" onClick={start} disabled={busy}>{busy ? "…" : t(lang, "next")}</button>
        {err && <div class="notice" style="margin-top:12px"><strong>Error:</strong> {err}</div>}
      </div>

      <div class="notice">
        <strong>Session privacy:</strong> This page clears all prior voter state before starting a new session.
      </div>
    </div>
  );
}
