import { useState } from "preact/hooks";
import { CameraCapture } from "../../components/cameraCapture";
import { t, useLang } from "../../i18n";
import { api } from "../../services/api";
import { state } from "../../state";
import { navigate } from "../../router";

/**
 * Client A liveness page:
 * - Captures a few downscaled still frames from webcam
 * - Sends to /api/liveness (mock)
 * - If biometric de-dup indicates REVOTE, kiosk flow requires booth-officer acknowledgement
 *   before the ballot is displayed (dispute handled by officer per SOP).
 */
export function A_Liveness() {
  const { lang } = useLang();
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [revoteAck, setRevoteAck] = useState(false);

  const session = state.session;
  if (!session) {
    return (
      <div class="card">
        <h2>Session missing</h2>
        <a href="#/a/start">Go to start</a>
      </div>
    );
  }

  async function onCaptured(framesB64: string[]) {
    setErr("");
    setBusy(true);
    setRevoteAck(false);
    try {
      const resp = await api.liveness({ sessionId: session.sessionId, stillsJpegB64: framesB64 }, session.sessionToken);
      state.liveness = resp;

      if (!resp.passed) return;

      // REVOTE handling is a governance decision: the booth officer must inform the voter and
      // handle any dispute before proceeding (kiosk).
      if (resp.dedup.status === "REVOTE" && session.mode === "kiosk") return;

      // Otherwise proceed to ballot.
      navigate("/a/ballot");
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  const l = state.liveness;

  function officerConfirmProceed() {
    setRevoteAck(true);
    navigate("/a/ballot");
  }

  function disputeFlow() {
    // In a real SOP: officer launches a dispute workflow (ID re-checks, supervisor approval, etc.)
    // For the demo, we end the session and return to start.
    navigate("/a/start");
  }

  return (
    <div>
      <h1>{t(lang, "liveness_title")}</h1>
      <p>{t(lang, "liveness_help")}</p>

      <CameraCapture frames={3} onCaptured={onCaptured} disabled={busy} />

      {l && (
        <div class="card">
          <div class={l.passed ? "badge ok" : "badge danger"}>
            {l.passed ? t(lang, "liveness_ok") : t(lang, "liveness_fail")} (score {l.score.toFixed(2)})
          </div>

          {l.dedup.status === "REVOTE" && l.dedup.prev && (
            <div class="notice" style="margin-top:12px">
              <strong>{t(lang, "revote_flag")}:</strong> Prior cast exists at {l.dedup.prev.castTime} (receipt {l.dedup.prev.receiptShortCode}).
              <div class="small" style="margin-top:8px">
                In supervised kiosks, the booth officer must inform the voter that this is a re-voting case.
                If the voter disputes, the officer decides per SOP using additional evidence (identity checks, polling register, supervisor escalation).
              </div>

              {session.mode === "kiosk" && (
                <div class="row" style="margin-top:12px">
                  <button class="primary" onClick={officerConfirmProceed}>{t(lang, "officer_confirm")}</button>
                  <button class="danger" onClick={disputeFlow}>{t(lang, "dispute")}</button>
                </div>
              )}
            </div>
          )}

          {revoteAck && <div class="small" style="margin-top:10px;color:var(--muted)">
            Officer acknowledgement recorded (demo-only); proceeding to ballot.
          </div>}
        </div>
      )}

      {err && <div class="notice"><strong>Error:</strong> {err}</div>}

      <div class="row">
        <button class="ghost" onClick={() => navigate("/a/start")}>{t(lang, "back")}</button>
      </div>
    </div>
  );
}
