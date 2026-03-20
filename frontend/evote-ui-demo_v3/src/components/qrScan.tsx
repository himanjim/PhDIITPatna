import { useEffect, useRef, useState } from "preact/hooks";
import { t, useLang } from "../i18n";

/**
 * QR intake component for the supervised verifier terminal.
 *
 * The component uses the browser BarcodeDetector API when available and falls back
 * to manual receipt entry when that API is unavailable. This keeps the verifier
 * client lightweight by avoiding large third-party scanning libraries while still
 * preserving a manual recovery path.
 */
export function QrScan(props: { onResult: (text: string) => void }) {
  const { lang } = useLang();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [status, setStatus] = useState<string>("");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [supported, setSupported] = useState<boolean>(false);
  const [running, setRunning] = useState<boolean>(false);

  useEffect(() => {
    setSupported("BarcodeDetector" in window);
    return () => {
      if (stream) stream.getTracks().forEach(t => t.stop());
    };
  }, [stream]);

  /**
   * Start the rear-camera scanning session when the browser supports QR detection.
   */
  async function start() {
    setStatus("");
    if (!("BarcodeDetector" in window)) {
      setStatus("BarcodeDetector not supported. Use manual paste.");
      return;
    }
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      setStream(s);
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        await videoRef.current.play();
      }
      setRunning(true);
      loopDetect();
    } catch (e: any) {
      setStatus(`Camera error: ${e?.message || String(e)}`);
    }
  }

  /**
   * Poll the active video stream for QR codes until a result is found or scanning
   * is stopped. The loop is intentionally simple because this component serves only
   * as a demo verifier input path.
   */
  async function loopDetect() {
    const v = videoRef.current;
    if (!v) return;
    const detector = new (window as any).BarcodeDetector({ formats: ["qr_code"] });

    // Simple polling loop (fast enough for demo).
    while (running) {
      try {
        const barcodes = await detector.detect(v);
        if (barcodes && barcodes.length > 0) {
          const value = barcodes[0].rawValue;
          if (value) {
            setStatus("QR detected.");
            props.onResult(value);
            stop();
            break;
          }
        }
      } catch {
        // ignore transient errors
      }
      await new Promise(r => setTimeout(r, 200));
    }
  }

  /**
   * Stop scanning and release the active camera stream.
   */
  function stop() {
    setRunning(false);
    if (stream) stream.getTracks().forEach(t => t.stop());
    setStream(null);
  }

  return (
    <div class="card">
      <div class="row" style="justify-content:space-between;align-items:center">
        <div>
          <h3 style="margin:0">{t(lang, "scan_qr")}</h3>
          <div class="small">{supported ? "Uses the device camera to scan the QR code." : "QR scan not supported; use manual paste."}</div>
        </div>
        <div class="row" style="gap:8px">
          <button class="primary" onClick={start} disabled={!supported || running}>Start scan</button>
          <button class="ghost" onClick={stop} disabled={!running}>Stop</button>
        </div>
      </div>
      <div class="hr"></div>
      <video ref={videoRef} autoplay playsinline muted style="width:100%;max-height:320px;border-radius:12px;border:1px solid var(--border)"></video>
      <div class="notice" aria-live="polite" style="margin-top:12px">
        <strong>Status:</strong> <span class="live">{status || "Idle"}</span>
      </div>
    </div>
  );
}
