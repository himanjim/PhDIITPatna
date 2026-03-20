import { useEffect, useRef, useState } from "preact/hooks";
import { captureDownscaledJpegB64 } from "../utils/image";
import { t, useLang } from "../i18n";

/**
 * Webcam component:
 * - Requests camera permission
 * - Shows live preview
 * - Captures N downscaled still frames as JPEG base64 strings
 *
 * Accessibility:
 * - Provides a clear status message area (aria-live)
 * - Uses buttons with explicit labels
 */

/**
 * Camera capture component used for the liveness step.
 *
 * The component requests webcam access, presents a live preview, captures a small
 * set of downscaled still frames, and returns those frames to the caller as compact
 * base64-encoded JPEG strings. The design prioritises predictable browser behaviour,
 * modest payload size, and explicit user feedback over feature richness.
 */
export function CameraCapture(props: {
  frames: number;
  onCaptured: (framesB64: string[]) => void;
  disabled?: boolean;
  /** If true (default), the camera is stopped immediately after frames are captured. */
  stopAfterCapture?: boolean;
}) {
  const { lang } = useLang();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [status, setStatus] = useState<string>("");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [busy, setBusy] = useState(false);

  /**
 * Release the active media stream and detach it from the video element.
 * This is called both when capture finishes and when the component is closed or
 * restarted, so that camera access is not left open longer than necessary.
 */
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
  }
  setStream(null);
  if (videoRef.current) {
    try {
      (videoRef.current as any).srcObject = null;
    } catch {}
  }
  setStatus("");
}

  useEffect(() => {
    return () => {
      // Ensure camera is released on unmount.
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [stream]);

  /**
 * Request camera access and attach the resulting stream to the preview element.
 * The function prefers a front-facing camera and moderate resolution, which is
 * sufficient for the demo liveness payload while keeping device demands modest.
 */
  async function startCamera() {
    setStatus("");
    // If the camera was already running, stop it before requesting again.
    if (stream) stopCamera();
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      setStream(s);
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        await videoRef.current.play();
      }
    } catch (e: any) {
      setStatus(`Camera error: ${e?.message || String(e)}`);
    }
  }

  /**
 * Capture the requested number of still frames from the active preview stream.
 * Frames are taken with small delays between them so that the mock liveness step
 * receives a short sequence rather than several copies of the same instant.
 */
  async function captureFrames() {
    if (props.disabled) return;
    const v = videoRef.current;
    if (!v) return;

    setBusy(true);
    setStatus(t(lang, "capturing"));
    try {
      const frames: string[] = [];
      // Capture a few frames with small delays.
      for (let i = 0; i < props.frames; i++) {
        frames.push(await captureDownscaledJpegB64(v, 320, 0.6));
        await new Promise(r => setTimeout(r, 700));
      }
      setStatus(`${frames.length} frame(s) captured.`);
      props.onCaptured(frames);
      if (props.stopAfterCapture !== false) stopCamera();
    } catch (e: any) {
      setStatus(`Capture error: ${e?.message || String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div class="card">
      <div class="row" style="align-items:center;justify-content:space-between">
        <div>
          <strong>{t(lang, "consent_camera")}</strong>
          <div class="small">Uses the device webcam to capture a few downscaled still frames.</div>
        </div>
        <div class="row" style="gap:8px">
          <button class="primary" onClick={startCamera} disabled={busy}>{t(lang, "consent_camera")}</button>
          <button class="primary" onClick={captureFrames} disabled={busy || !stream || props.disabled}>{t(lang, "capture")}</button>
          <button class="ghost" onClick={stopCamera} disabled={!stream || busy}>Stop</button>
        </div>
      </div>

      <div class="hr"></div>

      <video ref={videoRef} autoplay playsinline muted style="width:100%;max-height:360px;border-radius:12px;border:1px solid var(--border)"></video>

      <div class="notice" aria-live="polite" style="margin-top:12px">
        <strong>Status:</strong> <span class="live">{status || "Idle"}</span>
      </div>
    </div>
  );
}
