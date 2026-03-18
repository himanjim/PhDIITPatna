
"""
This module defines integration-style regression tests for the liveness
pipeline. Each test compares the behavioural outcome of the reference
implementation in `liveness_check.py` with the HTTP response returned by
`lvs_service.py` for the same underlying clip. The purpose is to detect
integration drift at the service boundary, including frame decoding,
default handling, and context construction, rather than to re-validate
the underlying liveness algorithm itself.
"""

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

# Resolve the directory containing scenario clips used by the regression
# tests, preferring the configured environment variable and otherwise
# falling back to a local `clips` folder.
def _clips_dir() -> Path:
    d = os.environ.get("LVS_CLIPS_DIR", "").strip()
    return Path(d) if d else (Path.cwd() / "clips")

# Build the expected file path for one named test scenario clip.
def _scenario_path(s: str) -> Path:
    return _clips_dir() / f"{s}.mp4"

# Return only those scenarios for which a corresponding clip file is
# currently available, so that parametrised tests remain robust to partial
# clip sets.
def _available_scenarios() -> List[str]:
    return [s for s in SCENARIOS if _scenario_path(s).is_file()]

# Encode one BGR frame as a base64 JPEG string in the same transport
# format expected by the HTTP liveness endpoint.
def _b64_jpeg(frame_bgr):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")

# For each available scenario clip, compare the boolean liveness outcome
# returned by the HTTP service with the outcome of the local reference
# implementation executed on the same decoded frames.
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
