import { useState } from "preact/hooks";
import { QrReceipt } from "../../components/qrReceipt";
import { t, useLang } from "../../i18n";
import { api } from "../../services/api";
import { state, clearSessionState } from "../../state";
import { navigate } from "../../router";

/**
 * Client A receipt page:
 * - Presents QR + short code (non-transferable receipt)
 * - SMS is assumed by default (backend responsibility); UI shows confirmation
 * - Optional printout in supervised kiosks (no file download)
 * - Verification on Client B is OPTIONAL (especially for kiosk voters); remote voters may visit supervised kiosk later.
 */
export function A_Receipt() {
  const { lang } = useLang();
  const [err, setErr] = useState("");
  const cast = state.cast;
  const session = state.session;

  if (!cast || !session) {
    return (
      <div class="card">
        <h2>Receipt missing</h2>
        <a href="#/a/start">Start again</a>
      </div>
    );
  }

  function printReceipt() {
    // Dedicated print view: only QR + short code + minimal metadata.
    const html = `
      <!doctype html><html><head><meta charset="utf-8"><title>Receipt</title></head>
      <body style="font-family:system-ui;padding:20px">
        <h2>Voting receipt (non-transferable)</h2>
        <p>Scan at verifier terminal to confirm recorded choice.</p>
        <pre style="white-space:pre-wrap;border:1px solid #ccc;padding:10px;border-radius:10px">${escapeHtml(cast.qrPayload)}</pre>
        <h3>Short code: ${escapeHtml(cast.shortCode)}</h3>
        <p>Epoch: ${cast.epoch} • Serial: ${escapeHtml(cast.serial)} • CastTime: ${escapeHtml(cast.castTime)}</p>
      </body></html>
    `;
    const w = window.open("", "_blank", "noopener,noreferrer");
    if (!w) return;
    w.document.write(html);
    w.document.close();
    w.focus();
    w.print();
    w.close();
  }

  async function endSessionAndNavigate(nextRoute: string) {
    setErr("");
    try {
      await api.endSession(session.sessionId, session.sessionToken);
    } catch (e: any) {
      // Even if backend end fails, we must clear UI state to prevent leakage across voters.
      setErr(`Warning: session end call failed (${e?.message || String(e)}). UI state will still be cleared.`);
    } finally {
      clearSessionState();
      navigate(nextRoute);
    }
  }

  return (
    <div>
      <QrReceipt
        qrPayload={cast.qrPayload}
        shortCode={cast.shortCode}
        printAllowed={cast.printAllowed}
        onPrint={printReceipt}
      />

      <div class="card">
        <div class="badge ok">SMS delivery</div>
        <p class="small">
          {cast.smsSent
            ? "Receipt SMS has been sent by default (no vote choice is included)."
            : "SMS delivery unavailable in this demo."}
        </p>

        <div class="hr"></div>

        <div class="row">
          <button class="primary" onClick={() => endSessionAndNavigate("/b/verify")}>
            {t(lang, "proceed_to_verify")}
          </button>
          <button class="ghost" onClick={() => endSessionAndNavigate("/a/end")}>
            {t(lang, "skip_verify")}
          </button>
        </div>

        {err && <div class="notice" style="margin-top:12px"><strong>Note:</strong> {err}</div>}

        <div class="small" style="margin-top:12px;color:var(--muted)">
          If the vote was cast remotely, the voter may later visit any supervised kiosk to verify using this receipt.
        </div>
      </div>
    </div>
  );
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;" } as any)[c]);
}
