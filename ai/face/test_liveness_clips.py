
"""
test_liveness_clips_v3_FIXED4.py

Fixes baseline call to match FINAL liveness_check signature:
  liveness_check(frames_bgr, det=TwoStageDetector, face_mesh=FaceMesh, ...)

Set env:
  set GATEWAY_AUTH_TOKEN=test-token
  set LVS_CLIPS_DIR=C:\path\to\clips
  set MP_FACE_LANDMARKER_TASK=C:\path\to\face_landmarker.task   (only if mp.solutions missing)
"""

# Notes for maintainers
# ---------------------
# This test suite checks behavioural equivalence between:
#   (a) the reference implementation in `liveness_check.py`, and
#   (b) the HTTP surface exposed by `lvs_service.py`.
# The intent is regression detection at the integration boundary (decode, defaults,
# and context construction), not algorithmic re-validation.
#
# Test data: MP4 clips produced by `record_liveness_clips.py` (scenarios like IDLE,
# HEAD_SHAKE, POSE, etc.). Configure the folder via LVS_CLIPS_DIR.

import base64
import os
from pathlib import Path
from typing import List

import cv2
import pytest
from fastapi.testclient import TestClient

import lvs_service as svc
import liveness_core as core
import liveness_check as lc

SCENARIOS: List[str] = ["IDLE", "HEAD_SHAKE", "POSE", "BLINK_TRY", "NO_MOTION", "MULTIFACE", "BLUR"]

def _clips_dir() -> Path:
    d = os.environ.get("LVS_CLIPS_DIR", "").strip()
    return Path(d) if d else (Path.cwd() / "clips")

def _scenario_path(s: str) -> Path:
    return _clips_dir() / f"{s}.mp4"

def _available_scenarios() -> List[str]:
    return [s for s in SCENARIOS if _scenario_path(s).is_file()]

def _b64_jpeg(frame_bgr):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")

@pytest.mark.parametrize("scenario", _available_scenarios())
def test_service_matches_baseline_on_clip(scenario: str):
    ctx = svc._get_ctx_for_tests()

    clip = _scenario_path(scenario)
    frames = core.load_clip_frames_bgr(str(clip), max_frames=60)
    assert frames, f"no frames: {clip}"

    res = lc.liveness_check(frames_bgr=frames, det=ctx.det, face_mesh=ctx.face_mesh, prompt="none")
    baseline_ok = bool(res.ok)
    baseline_why = str(res.why)

    client = TestClient(svc.app)
    headers = {"X-Gateway-Auth": os.environ.get("GATEWAY_AUTH_TOKEN", "test-token")}
    payload = {
        "session_id": "S-1",
        "subject_ref": "U-1",
        "frames_b64": [_b64_jpeg(f) for f in frames],
        "prompt": "none",
    }
    r = client.post("/v1/liveness", headers=headers, json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    assert bool(data["ok"]) == baseline_ok, (
        f"{scenario}: ok mismatch. baseline={baseline_ok} ({baseline_why}) "
        f"service={data['ok']} ({data.get('why')})"
    )
