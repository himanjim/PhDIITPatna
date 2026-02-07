import { useEffect, useMemo, useState } from "preact/hooks";
import { QrScan } from "../../components/qrScan";
import { t, useLang } from "../../i18n";
import { api, VerifyResp } from "../../services/api";
import { clearSessionState } from "../../state";
import { navigate } from "../../router";

/**
 * Client B verification page:
 * - Requires booth device enrollment (device-bound credential) to access verify service
 * - Accepts QR scan or manual paste/short-code entry
 * - Displays vote choice only inside supervised booth context
 *
 * SECURITY INTENT:
 *  - Even if a remote voter obtains the receipt, they cannot decrypt/reveal the choice because:
 *      (i) verification service is reachable only from enrolled booth devices (controlled provisioning + device credential),
 *     (ii) network allowlisting is used only as defence-in-depth; primary barrier is credentialed access.
 */
export function B_Verify() {
  const { lang } = useLang();
  const [input, setInput] = useState("");
  const [resp, setResp] = useState<VerifyResp | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const deviceId = localStorage.getItem("evote.b.deviceId") || "";
  const deviceToken = localStorage.getItem("evote.b.deviceToken") || "";
  const expiresAt = localStorage.getItem("evote.b.expiresAt") || "";

  const expired = expiresAt ? (Date.parse(expiresAt) < Date.now()) : false;
  const provisioned = Boolean(deviceId && deviceToken && !expired);

  // Hygiene: if verifier credentials are expired, wipe them.
  if (expired) {
    localStorage.removeItem("evote.b.deviceId");
    localStorage.removeItem("evote.b.deviceToken");
    localStorage.removeItem("evote.b.expiresAt");
  }

  useEffect(() => {
    // Client B is a verifier terminal: it must never reuse any Client A session state.
    clearSessionState();
    try { sessionStorage.clear(); } catch {}
  }, []);

  const statusBadgeClass = useMemo(() => {
    if (!resp) return "badge";
    if (resp.status === "CONFIRMED") return "badge ok";
    if (resp.status === "PENDING") return "badge warn";
    if (resp.status === "SUPERSEDED") return "badge warn";
    return "badge danger";
  }, [resp]);

  async function verify(q: string) {
    setErr("");
    setBusy(true);
    setResp(null);
    try {
      const r = await api.verifyReceipt({
        deviceId,
        deviceToken,
        qrOrShortCode: q.trim()
      });
      setResp(r);
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!provisioned) {
    return (
      <div class="card">
        <h1>{t(lang, "verify_title")}</h1>
        <div class="notice">
          <strong>{t(lang, "not_provisioned")}</strong>
          <div class="small" style="margin-top:8px">
            This is expected on a remote device. In a real deployment, this UI is installed only on managed booth devices.
          </div>
        </div>
        <div class="hr"></div>
        <button class="primary" onClick={() => navigate("/b/enroll")}>{t(lang, "go_enroll")}</button>
        <a class="badge" href="#/">Back to home</a>
      </div>
    );
  }

  return (
    <div>
      <h1>{t(lang, "verify_title")}</h1>

      <QrScan onResult={(txt) => { setInput(txt); verify(txt); }} />

      <div class="card">
        <label for="paste">{t(lang, "paste_qr")}</label>
        <textarea id="paste" value={input} onInput={(e) => setInput((e.target as HTMLTextAreaElement).value)} placeholder="Paste JSON payload or enter short code (e.g., ABCDE-12345)"></textarea>

        <div class="hr"></div>
        <button class="primary" onClick={() => verify(input)} disabled={busy || !input.trim()}>{busy ? "…" : t(lang, "verify")}</button>
      </div>

      {err && <div class="notice"><strong>Error:</strong> {err}</div>}

      {resp && (
        <div class="card" aria-live="polite">
          <div class={statusBadgeClass}>{t(lang, "status")}: {t(lang, resp.status.toLowerCase())}</div>
          <div class="small" style="margin-top:10px">
            Epoch: <span class="kbd">{resp.epoch}</span> • Serial: <span class="kbd">{resp.serial}</span>
          </div>
          <div class="small">
            Commitment: <span class="kbd">{resp.hC.slice(0, 14)}…</span> • CastTime: <span class="kbd">{resp.castTime}</span>
          </div>

          {resp.candidate && (
            <div class="notice" style="margin-top:12px">
              <strong>Recorded choice (booth-only view):</strong>
              <div style="font-size:1.2rem;font-weight:700;margin-top:6px">{resp.candidate.name}</div>
              <div class="small">{resp.candidate.party}</div>
              <div class="small" style="margin-top:8px;color:var(--muted)">
                This choice is displayed only on the supervised verifier terminal and is not exported or logged.
              </div>
            </div>
          )}

          {resp.reason && <div class="small" style="margin-top:10px;color:var(--muted)">{resp.reason}</div>}

          <div class="hr"></div>
          <div class="row">
            <button class="primary" onClick={() => navigate("/a/start")}>{t(lang, "request_revote")}</button>
            <button class="ghost" onClick={() => { setResp(null); setInput(""); }}>{t(lang, "cancel")}</button>
          </div>
        </div>
      )}
    </div>
  );
}
