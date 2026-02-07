import { useEffect, useRef, useState } from "preact/hooks";
import QRCode from "qrcode";
import { t, useLang } from "../i18n";
import { speak } from "../utils/tts";

/**
 * QR rendering component.
 * - Renders a QR code into a canvas (self-hosted dependency via npm)
 * - Shows the short code in large text for manual entry fallback
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
