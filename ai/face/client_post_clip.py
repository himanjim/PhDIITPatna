# Client utility (operator-facing)
# -------------------------------
# Sends a single liveness request to the local service using a recorded MP4 clip.
# The script is deliberately minimal: it samples up to 60 frames, encodes each as
# JPEG, base64-encodes the bytes, and POSTs the `frames_b64` array to /v1/liveness.
#
# Environment variables:
#   CLIP_PATH          : full path to the MP4 clip to replay
#   GATEWAY_AUTH_TOKEN : token used for the X-Gateway-Auth header

import os, base64, cv2, requests

URL = "http://127.0.0.1:8000/v1/liveness"
TOKEN = os.environ.get("GATEWAY_AUTH_TOKEN", "test-token")
CLIP  = os.environ["CLIP_PATH"]

cap = cv2.VideoCapture(CLIP)
frames_b64 = []
while len(frames_b64) < 60:
    ok, fr = cap.read()
    if not ok:
        break
    ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        continue
    frames_b64.append(base64.b64encode(buf.tobytes()).decode("ascii"))
cap.release()

payload = {
    "session_id": "S-1",
    "subject_ref": "U-1",
    "frames_b64": frames_b64,
    "prompt": "none",
}
r = requests.post(URL, json=payload, headers={"X-Gateway-Auth": TOKEN}, timeout=120)
print("status", r.status_code)
print(r.text)