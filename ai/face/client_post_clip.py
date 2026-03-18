# This script is a minimal client utility for replaying a recorded MP4
# clip against the local liveness gateway. It extracts up to a fixed
# number of frames from the clip, JPEG-encodes them, converts the bytes to
# base64 strings, and submits the resulting frame list in a single HTTP
# request. The purpose is functional testing of the gateway interface
# rather than benchmarking or large-scale workload generation.

import os, base64, cv2, requests

URL = "http://127.0.0.1:8000/v1/liveness"
TOKEN = os.environ.get("GATEWAY_AUTH_TOKEN", "test-token")
CLIP  = os.environ["CLIP_PATH"]

# Read frames sequentially from the input clip and convert each retained
# frame into the base64-encoded JPEG representation expected by the HTTP
# gateway.
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

# Construct the request payload in the format expected by the liveness
# endpoint, including the session identifier, subject reference, frame
# sequence, and prompt.
payload = {
    "session_id": "S-1",
    "subject_ref": "U-1",
    "frames_b64": frames_b64,
    "prompt": "none",
}
r = requests.post(URL, json=payload, headers={"X-Gateway-Auth": TOKEN}, timeout=120)
print("status", r.status_code)
print(r.text)
