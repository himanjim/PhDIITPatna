/**
 * Camera capture helpers:
 * - Capture from <video> into a square-cropped canvas
 * - Downscale to a fixed size
 * - Export as JPEG base64 (for demo API payloads)
 *
 * NOTE: This does not perform face detection; it crops the center square.
 * In your production build, you can add face detection/cropping on-device (e.g., MediaPipe)
 * but weigh the extra complexity against your low-bandwidth + low-dependency constraints.
 */

export async function captureDownscaledJpegB64(video: HTMLVideoElement, size = 320, quality = 0.6): Promise<string> {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) throw new Error("video not ready");

  // Center square crop
  const side = Math.min(w, h);
  const sx = (w - side) / 2;
  const sy = (h - side) / 2;

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no canvas context");

  ctx.drawImage(video, sx, sy, side, side, 0, 0, size, size);

  const blob: Blob = await new Promise((resolve, reject) => {
    canvas.toBlob((b) => b ? resolve(b) : reject(new Error("toBlob failed")), "image/jpeg", quality);
  });

  const b64 = await blobToBase64(blob);
  // Strip the "data:image/jpeg;base64," prefix for compactness.
  return b64.replace(/^data:image\/jpeg;base64,/, "");
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onerror = () => reject(new Error("read failed"));
    r.onload = () => resolve(String(r.result));
    r.readAsDataURL(blob);
  });
}
