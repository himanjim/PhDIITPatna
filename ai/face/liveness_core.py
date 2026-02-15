
"""
liveness_core_UPDATED_v2_FIXED4.py

Fixes your current test failure:
  AttributeError: 'FaceAnalysis' object has no attribute 'count'

Root cause
----------
Your FINAL `liveness_check.py` expects `det` to be an instance of
`TwoStageDetector` (it calls det.count(...) and det.get(...)).

This wrapper therefore builds and passes:
  det = liveness_check.TwoStageDetector(...)

and builds:
  face_mesh = mp.solutions.face_mesh.FaceMesh(...)

without modifying liveness_check.py.
"""

# Design note
# -----------
# This module acts as a thin integration layer around the *final* `liveness_check.py`.
# It standardises how heavyweight objects are built (TwoStageDetector + FaceMesh) and
# exposes small helper functions used by both the web service and the clip-based tests.
#
# Importantly, `liveness_check.py` expects `det` to implement the TwoStageDetector API
# (e.g., count/get). Passing an InsightFace FaceAnalysis instance directly would be a
# type mismatch. The `build_context()` function ensures the correct detector is used.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from mp_face_mesh_tasks_shim import patch_mediapipe_solutions


@dataclass
class LivenessContext:
    det: Any
    face_mesh: Any
    providers: List[str]
    ctx_id: int
    det_fast_size: Tuple[int, int]
    det_strong_size: Tuple[int, int]
    det_score_thr: float


def _parse_wh(s: str, default: Tuple[int, int]) -> Tuple[int, int]:
    try:
        s = (s or "").lower().replace(" ", "")
        if "x" in s:
            a, b = s.split("x", 1)
            w = int(a); h = int(b)
            if w > 0 and h > 0:
                return (w, h)
    except Exception:
        pass
    return default


def _default_providers_and_ctxid() -> Tuple[List[str], int]:
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        use_cuda = "CUDAExecutionProvider" in avail
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        ctx_id = 0 if use_cuda else -1
        return providers, ctx_id
    except Exception:
        return ["CPUExecutionProvider"], -1


def build_context(
    *,
    providers_csv: str = "",
    ctx_id: Optional[int] = None,
    det_fast_size: str = "512x512",
    det_size: str = "640x640",
    det_score_thr: float = 0.55,
    mp_refine_landmarks: bool = True,
    mp_min_det: float = 0.6,
    mp_min_track: float = 0.6,
) -> LivenessContext:
    if providers_csv.strip():
        providers = [p.strip() for p in providers_csv.split(",") if p.strip()]
        if ctx_id is None:
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    else:
        providers, auto_ctx = _default_providers_and_ctxid()
        if ctx_id is None:
            ctx_id = auto_ctx

    fast_sz = _parse_wh(det_fast_size, (512, 512))
    strong_sz = _parse_wh(det_size, (640, 640))

    import liveness_check as lc
    det = lc.TwoStageDetector(
        providers=providers,
        ctx_id=int(ctx_id),
        fast_size=fast_sz,
        strong_size=strong_sz,
        det_score_thr=float(det_score_thr),
    )

    patch_mediapipe_solutions()
    import mediapipe as mp  # type: ignore
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=bool(mp_refine_landmarks),
        min_detection_confidence=float(mp_min_det),
        min_tracking_confidence=float(mp_min_track),
    )

    return LivenessContext(
        det=det,
        face_mesh=face_mesh,
        providers=providers,
        ctx_id=int(ctx_id),
        det_fast_size=fast_sz,
        det_strong_size=strong_sz,
        det_score_thr=float(det_score_thr),
    )


def load_clip_frames_bgr(clip_path: str, *, max_frames: int = 60) -> List[np.ndarray]:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")
    out: List[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
        if max_frames and len(out) >= int(max_frames):
            break
    cap.release()
    return out


def run_liveness(
    frames_bgr: List[np.ndarray],
    ctx: LivenessContext,
    *,
    roi_bbox_override: Optional[Tuple[int, int, int, int]] = None,
    min_oneface_frac: float = 0.60,
    max_twoplus_frac: float = 0.05,
    det_stride: int = 6,
    motion_thr: float = 0.012,
    blur_thr: float = 80.0,
    bright_lo: float = 35.0,
    bright_hi: float = 220.0,
    brightness_gate: bool = False,
    roi_pad: float = 0.35,
    roi_resize_w: int = 160,
    roi_stride: int = 2,
    motion_norm_size: Tuple[int, int] = (160, 160),
    replay_thr: float = 0.0045,
    mask_thr: float = 0.35,
    print_thr: float = 12.0,
    prompt: str = "none",
    mesh_stride: int = 3,
    yaw_thr: float = 0.045,
) -> Dict[str, Any]:
    import liveness_check as lc
    res = lc.liveness_check(
        frames_bgr=frames_bgr,
        det=ctx.det,
        face_mesh=ctx.face_mesh,
        roi_bbox_override=roi_bbox_override,
        min_oneface_frac=float(min_oneface_frac),
        max_twoplus_frac=float(max_twoplus_frac),
        det_stride=int(det_stride),
        motion_thr=float(motion_thr),
        blur_thr=float(blur_thr),
        bright_lo=float(bright_lo),
        bright_hi=float(bright_hi),
        brightness_gate=bool(brightness_gate),
        roi_pad=float(roi_pad),
        roi_resize_w=int(roi_resize_w),
        roi_stride=int(roi_stride),
        motion_norm_size=tuple(motion_norm_size),
        replay_thr=float(replay_thr),
        mask_thr=float(mask_thr),
        print_thr=float(print_thr),
        prompt=str(prompt),
        mesh_stride=int(mesh_stride),
        yaw_thr=float(yaw_thr),
    )

    scal = res.scalars.__dict__.copy()
    tim = res.timing.__dict__.copy()

    metrics: Dict[str, float] = {}
    for k, v in scal.items():
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    for k, v in tim.items():
        try:
            metrics[f"t_{k}"] = float(v)
        except Exception:
            pass

    return {"ok": bool(res.ok), "why": str(res.why), "metrics": metrics}
