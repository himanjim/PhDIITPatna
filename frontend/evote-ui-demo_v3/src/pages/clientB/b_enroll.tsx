import { useState } from "preact/hooks";
import { t, useLang } from "../../i18n";
import { api } from "../../services/api";
import { navigate } from "../../router";

/**
 * Device-enrolment page for the verifier terminal.
 *
 * This page models the booth-side provisioning step through which a verifier
 * device receives a device identifier and a device-bound credential. The demo
 * reduces this to a booth identifier and short enrolment code, but the intended
 * operational role is to represent controlled terminal provisioning rather than
 * voter interaction.
 */
export function B_Enroll() {
  const { lang } = useLang();
  const [boothId, setBoothId] = useState("BOOTH-17");
  const [enrollCode, setEnrollCode] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  /**
   * Request verifier-device enrolment and store the issued device credential in
   * browser storage for later receipt-verification calls.
   */
  async function enroll() {
    setErr("");
    setBusy(true);
    try {
      const resp = await api.enrollVerifier({ boothId: boothId.trim(), enrollCode: enrollCode.trim() });
      localStorage.setItem("evote.b.deviceId", resp.deviceId);
      localStorage.setItem("evote.b.deviceToken", resp.deviceToken);
      localStorage.setItem("evote.b.expiresAt", resp.expiresAt);
      navigate("/b/verify");
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1>{t(lang, "enroll_title")}</h1>
      <div class="card">
        <p class="small">
          Enrollment should be performed only under booth supervision. The enrollment code must never be distributed to remote users.
        </p>

        <label for="booth">Booth ID</label>
        <input id="booth" value={boothId} onInput={(e) => setBoothId((e.target as HTMLInputElement).value)} />

        <label for="code">{t(lang, "enroll_code")}</label>
        <input id="code" value={enrollCode} onInput={(e) => setEnrollCode((e.target as HTMLInputElement).value)} placeholder="demo: 123456" />

        <div class="hr"></div>
        <button class="primary" onClick={enroll} disabled={busy}>{busy ? "…" : t(lang, "enroll")}</button>
        {err && <div class="notice" style="margin-top:12px"><strong>Error:</strong> {err}</div>}
      </div>
    </div>
  );
}
