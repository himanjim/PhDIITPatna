import { useEffect, useRef, useState } from "preact/hooks";
import QRCode from "qrcode";
import { t, useLang } from "../i18n";
import { speak } from "../utils/tts";

/**
 * Receipt presentation component for Client A.
 *
 * The component renders the receipt payload as a QR code, displays a human-enterable
 * short code as a fallback path, and optionally exposes a print action when the
 * session context permits supervised kiosk printing. The displayed content is
 * intentionally limited to receipt-facing identifiers and does not reveal the
 * recorded choice on the voter-facing client.
 */
export function QrReceipt(props: {
  qrPayload: string;
  shortCode: string;
  onPrint?: () => void;
  printAllowed?: boolean;
}) {
  const { lang } = useLang();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        setErr("");
        const c = canvasRef.current;
        if (!c) return;
        await QRCode.toCanvas(c, props.qrPayload, {
          errorCorrectionLevel: "M",
          margin: 2,
          width: 320
        });
      } catch (e: any) {
        setErr(e?.message || String(e));
      }
    })();
  }, [props.qrPayload]);

  /**
 * Read the short receipt code aloud to support users who benefit from an audio
 * confirmation path or who cannot easily copy the printed characters visually.
 */
  function speakShortCode() {
    speak(`Receipt code ${props.shortCode}. Proceed to verification if you wish.`, "en-IN");
  }

  return (
    <div class="card">
      <h2>{t(lang, "receipt_title")}</h2>
      <p class="small">
        QR payload contains identifiers only. It does not prove vote choice outside supervised verification.
      </p>
      <div class="row">
        <div class="col">
          <canvas ref={canvasRef} style="background:#fff;border-radius:12px;padding:8px" aria-label="Receipt QR code"></canvas>
          {err && <div class="notice"><strong>QR error:</strong> {err}</div>}
        </div>
        <div class="col">
          <div class="card">
            <div class="badge">Short code</div>
            <div style="font-size:1.6rem;font-weight:700;margin-top:10px">{props.shortCode}</div>
            <p class="small">If scanning is unavailable, enter this code at the verifier terminal.</p>
            <div class="row">
              <button class="ghost" onClick={speakShortCode}>🔊 Read aloud</button>
              {props.printAllowed && props.onPrint && (
                <button class="primary" onClick={props.onPrint}>{t(lang, "print_receipt")}</button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
