
"""
lvs_service_UPDATED_v3_FIXED5.py

FastAPI LVS that calls your FINAL `liveness_check.py` via `liveness_core_UPDATED_v2_FIXED4.py`.

Why this FIXED5 exists
----------------------
Your curl calls used:
  GET  /v1/policy
  POST /v1/liveness/check_clip

But FIXED4 only exposed:
  GET  /v1/health
  POST /v1/liveness

So FIXED5 adds the missing endpoints **without changing** the tested /v1/liveness behaviour.

Endpoints
---------
- GET  /v1/health
- GET  /v1/policy                   (auth required)
- POST /v1/liveness                 (auth required)  [frames_b64]
- POST /v1/liveness/check_clip      (auth required)  [clip_path -> frames]

Auth
----
All protected endpoints require:
  Header: X-Gateway-Auth: <token>
and env var:
  GATEWAY_AUTH_TOKEN=<same token>

MediaPipe Tasks shim
--------------------
If your MediaPipe wheel does not ship `mp.solutions`, keep the shim in place
(sitecustomize.py + mp_face_mesh_tasks_shim_v2.py) and set:
  MP_FACE_LANDMARKER_TASK=C:\path\to\face_landmarker.task
"""

# Service overview
# ---------------
# FastAPI wrapper around the final liveness pipeline.
# Responsibilities are intentionally narrow:
#   1) authenticate requests (X-Gateway-Auth)
#   2) validate and decode inputs (frames_b64 or local clip_path)
#   3) call the liveness core and return a stable, audit-friendly response
#
# The policy hash is a compact fingerprint of threshold configuration. It enables
# downstream dashboards to display which policy was enforced for a given request.


from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import liveness_core as core


# -----------------------------
# Config (env-driven)
# -----------------------------
ELECTION_ID = os.environ.get("ELECTION_ID", "EID-TEST").strip() or "EID-TEST"
GATEWAY_AUTH_TOKEN = os.environ.get("GATEWAY_AUTH_TOKEN", "").strip()
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "90"))

DET_STRIDE = int(os.environ.get("DET_STRIDE", "6"))
MESH_STRIDE = int(os.environ.get("MESH_STRIDE", "3"))
MIN_ONEFACE_FRAC = float(os.environ.get("MIN_ONEFACE_FRAC", "0.60"))
MAX_TWOPLUS_FRAC = float(os.environ.get("MAX_TWOPLUS_FRAC", "0.05"))
MOTION_THR = float(os.environ.get("MOTION_THR", "0.012"))
BLUR_THR = float(os.environ.get("BLUR_THR", "80.0"))
REPLAY_THR = float(os.environ.get("REPLAY_THR", "0.0045"))
MASK_THR = float(os.environ.get("MASK_THR", "0.35"))
PRINT_THR = float(os.environ.get("PRINT_THR", "12.0"))
YAW_THR = float(os.environ.get("YAW_THR", "0.045"))

DET_SIZE = os.environ.get("DET_SIZE", "640x640")
DET_FAST_SIZE = os.environ.get("DET_FAST_SIZE", "512x512")
DET_SCORE_THR = float(os.environ.get("DET_SCORE_THR", "0.55"))
CTX_ID = os.environ.get("CTX_ID", "")
PROVIDERS = os.environ.get("PROVIDERS", "")

# Policy snapshot & hash (useful for audit / transparency dashboards)
_POLICY_DOMAIN = "AI_POLICY_HASH_V1"
_POLICY_OBJ = {
    "election_id": ELECTION_ID,
    "det_fast_size": DET_FAST_SIZE,
    "det_size": DET_SIZE,
    "det_score_thr": DET_SCORE_THR,
    "det_stride": DET_STRIDE,
    "mesh_stride": MESH_STRIDE,
    "min_oneface_frac": MIN_ONEFACE_FRAC,
    "max_twoplus_frac": MAX_TWOPLUS_FRAC,
    "motion_thr": MOTION_THR,
    "blur_thr": BLUR_THR,
    "replay_thr": REPLAY_THR,
    "mask_thr": MASK_THR,
    "print_thr": PRINT_THR,
    "yaw_thr": YAW_THR,
}
_POLICY_HASH = hashlib.sha256(
    (_POLICY_DOMAIN + "\n" + json.dumps(_POLICY_OBJ, sort_keys=True, separators=(",", ":"))).encode("utf-8")
).hexdigest()


# -----------------------------
# Lazy shared context
# -----------------------------
_CTX: Optional[core.LivenessContext] = None


def _ensure_ctx() -> core.LivenessContext:
    """
    Build TwoStageDetector + FaceMesh once and reuse.
    """
    global _CTX
    if _CTX is not None:
        return _CTX

    ctx_id_val: Optional[int] = None
    if str(CTX_ID).strip():
        try:
            ctx_id_val = int(CTX_ID)
        except Exception:
            ctx_id_val = None

    _CTX = core.build_context(
        providers_csv=PROVIDERS,
        ctx_id=ctx_id_val,
        det_fast_size=DET_FAST_SIZE,
        det_size=DET_SIZE,
        det_score_thr=DET_SCORE_THR,
        mp_refine_landmarks=True,
        mp_min_det=0.6,
        mp_min_track=0.6,
    )
    return _CTX


def _auth_or_403(req: Request) -> None:
    """
    Fail-closed auth: if token not configured, return 500 (misconfiguration).
    """
    if not GATEWAY_AUTH_TOKEN:
        raise HTTPException(status_code=500, detail="GATEWAY_AUTH_TOKEN not configured")
    tok = req.headers.get("X-Gateway-Auth", "")
    if tok != GATEWAY_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")


def _decode_frame_b64(s: str) -> np.ndarray:
    """
    Base64 -> JPEG bytes -> BGR frame.
    """
    b = base64.b64decode(s, validate=True)
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("frame decode failed")
    return img


def _reason_code(why: str) -> str:
    """
    Lightweight mapping from human-readable `why` to a stable reason_code.
    Keep it conservative: do not overfit strings.
    """
    w = (why or "").lower()
    if "oneface_frac" in w or "twoplus_frac" in w or "face count" in w:
        return "FACE_COUNT_FAIL"
    if "blur" in w:
        return "BLUR_TOO_HIGH"
    if "motion" in w:
        return "MOTION_TOO_LOW"
    if "replay" in w:
        return "REPLAY_SUSPECT"
    if "mask" in w:
        return "MASK_SUSPECT"
    if "print" in w:
        return "PRINT_SUSPECT"
    if "yaw" in w or "pose" in w:
        return "PROMPT_POSE_FAIL"
    if "blink" in w:
        return "PROMPT_BLINK_FAIL"
    return "FAIL_OTHER"


def _retryable(code: str) -> bool:
    """
    Retryable means "user can fix by re-capturing a better clip".
    """
    return code in {
        "BLUR_TOO_HIGH",
        "MOTION_TOO_LOW",
        "PROMPT_BLINK_FAIL",
        "PROMPT_POSE_FAIL",
        "FACE_COUNT_FAIL",
    }


def _make_response(ok: bool, why: str, metrics: Dict[str, float]) -> Dict:
    code = _reason_code(why)
    return {
        "ok": bool(ok),
        "status": "PASS" if ok else "FAIL",
        "reason_code": code,
        "retryable": (False if ok else bool(_retryable(code))),
        "why": str(why),
        "election_id": ELECTION_ID,
        "policy_hash_domain": _POLICY_DOMAIN,
        "policy_hash_sha256": _POLICY_HASH,
        "metrics": {k: float(v) for k, v in (metrics or {}).items() if isinstance(v, (int, float))},
    }


# -----------------------------
# API models
# -----------------------------
Prompt = Literal["none", "blink", "pose"]


class LivenessRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    subject_ref: str = Field(..., min_length=1, max_length=128)
    frames_b64: List[str] = Field(..., min_items=3)
    prompt: Optional[Prompt] = "none"


class ClipRequest(BaseModel):
    clip_path: str = Field(..., min_length=1)
    max_frames: int = Field(60, ge=3, le=600)
    prompt: Optional[Prompt] = "none"


class LivenessResponse(BaseModel):
    ok: bool
    status: Literal["PASS", "FAIL"]
    reason_code: str
    retryable: bool
    why: str
    election_id: str
    policy_hash_domain: str
    policy_hash_sha256: str
    metrics: Dict[str, float]


# -----------------------------
# App + routes
# -----------------------------
app = FastAPI(title="LVS", version="3.1-fixed5")


@app.get("/v1/health")
def health():
    return {"status": "ok", "election_id": ELECTION_ID, "policy_hash_sha256": _POLICY_HASH}


@app.get("/v1/policy")
def policy(req: Request):
    _auth_or_403(req)
    return {
        "policy": _POLICY_OBJ,
        "policy_hash_domain": _POLICY_DOMAIN,
        "policy_hash_sha256": _POLICY_HASH,
    }


@app.post("/v1/liveness", response_model=LivenessResponse)
def liveness(req: Request, body: LivenessRequest):
    _auth_or_403(req)

    if len(body.frames_b64) > MAX_FRAMES:
        raise HTTPException(status_code=413, detail=f"too many frames (max {MAX_FRAMES})")

    try:
        frames = [_decode_frame_b64(s) for s in body.frames_b64]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad frames: {e}")

    ctx = _ensure_ctx()
    t0 = time.perf_counter()

    out = core.run_liveness(
        frames_bgr=frames,
        ctx=ctx,
        prompt=str(body.prompt or "none"),
        det_stride=DET_STRIDE,
        mesh_stride=MESH_STRIDE,
        min_oneface_frac=MIN_ONEFACE_FRAC,
        max_twoplus_frac=MAX_TWOPLUS_FRAC,
        motion_thr=MOTION_THR,
        blur_thr=BLUR_THR,
        replay_thr=REPLAY_THR,
        mask_thr=MASK_THR,
        print_thr=PRINT_THR,
        yaw_thr=YAW_THR,
    )

    wall_ms = (time.perf_counter() - t0) * 1000.0
    metrics = dict(out.get("metrics") or {})
    metrics["service_wall_ms"] = float(wall_ms)

    return _make_response(bool(out["ok"]), str(out["why"]), metrics)


@app.post("/v1/liveness/check_clip", response_model=LivenessResponse)
def liveness_check_clip(req: Request, body: ClipRequest):
    """
    Convenience endpoint for local testing with MP4 clips (e.g., record_liveness_clips.py output).
    Not intended for production clients.
    """
    _auth_or_403(req)

    if not os.path.isfile(body.clip_path):
        raise HTTPException(status_code=400, detail=f"clip not found: {body.clip_path}")

    ctx = _ensure_ctx()
    frames = core.load_clip_frames_bgr(body.clip_path, max_frames=int(body.max_frames))
    if not frames:
        raise HTTPException(status_code=400, detail="no frames decoded from clip")

    out = core.run_liveness(
        frames_bgr=frames,
        ctx=ctx,
        prompt=str(body.prompt or "none"),
        det_stride=DET_STRIDE,
        mesh_stride=MESH_STRIDE,
        min_oneface_frac=MIN_ONEFACE_FRAC,
        max_twoplus_frac=MAX_TWOPLUS_FRAC,
        motion_thr=MOTION_THR,
        blur_thr=BLUR_THR,
        replay_thr=REPLAY_THR,
        mask_thr=MASK_THR,
        print_thr=PRINT_THR,
        yaw_thr=YAW_THR,
    )
    return _make_response(bool(out["ok"]), str(out["why"]), dict(out.get("metrics") or {}))


# pytest helper
def _get_ctx_for_tests() -> core.LivenessContext:
    return _ensure_ctx()
