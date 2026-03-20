/**
 * Image-capture helpers used by the liveness workflow.
 *
 * The functions in this file crop the centre square from a live video frame,
 * resize it to a fixed dimension, and export it as a JPEG base64 string suitable
 * for compact API transport in the demo. The implementation intentionally avoids
 * face detection or device-specific optimisation so that behaviour remains simple
 * and predictable.
 */

/**
 * Capture one frame from the supplied video element, crop the centre square,
 * resize it to the requested output dimension, and return the JPEG content as a
 * base64 string without the data-URL prefix.
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

/**
 * Convert one Blob into a data URL so that the caller can extract the encoded
 * image payload.
 */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onerror = () => reject(new Error("read failed"));
    r.onload = () => resolve(String(r.result));
    r.readAsDataURL(blob);
  });
}
