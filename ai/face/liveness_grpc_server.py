#!/usr/bin/env python3
"""
liveness_grpc_server.py

One “replica” gRPC server for liveness benchmarking (clip-cache mode).

Key measurement
---------------
- Preload MP4 clips into RAM at startup (decoded BGR frames).
- RPC carries only (clip_id, prompt, max_frames).
- server_compute_ms is measured around core.run_liveness(...) only.

Concurrency control
-------------------
Even if gRPC accepts many concurrent requests, do not run unlimited in-flight
liveness calls on one GPU/one process. We cap compute concurrency via a semaphore
(MAX_INFER_CONCURRENCY). Start with 1, increase only if validated.

Optional auth
-------------
If INTERNAL_AUTH_TOKEN is set, the request must include metadata:
  x-internal-auth: <token>

Build stubs (once):
  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. liveness.proto
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import grpc

import liveness_core as core
import liveness_pb2
import liveness_pb2_grpc


# -----------------------------
# Env-driven configuration
# -----------------------------
ELECTION_ID = os.environ.get("ELECTION_ID", "EID-TEST").strip() or "EID-TEST"
INTERNAL_AUTH_TOKEN = os.environ.get("INTERNAL_AUTH_TOKEN", "").strip()

MAX_FRAMES_DEFAULT = int(os.environ.get("MAX_FRAMES_DEFAULT", "60"))
MAX_INFER_CONCURRENCY = int(os.environ.get("MAX_INFER_CONCURRENCY", "1"))
GRPC_MAX_WORKERS = int(os.environ.get("GRPC_MAX_WORKERS", "16"))

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

# Optional: resize clips at cache-load time (OFF by default; keep 0 for accuracy parity)
CLIP_RESIZE_W = int(os.environ.get("CLIP_RESIZE_W", "0"))  # 0 => no resize


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
_POLICY_JSON = json.dumps(_POLICY_OBJ, sort_keys=True, separators=(",", ":"))
_POLICY_HASH = hashlib.sha256((_POLICY_DOMAIN + "\n" + _POLICY_JSON).encode("utf-8")).hexdigest()


def _reason_code(why: str) -> str:
    w = (why or "").lower()
    if "multiple/no faces" in w or "no face" in w or "onefacefrac" in w or "twoplusfrac" in w:
        return "FACE_COUNT_FAIL"
    if "blur" in w:
        return "BLUR_TOO_HIGH"
    if "micro-motion" in w or "motion" in w:
        return "MOTION_TOO_LOW"
    if "replay" in w:
        return "REPLAY_SUSPECT"
    if "mask" in w:
        return "MASK_SUSPECT"
    if "print" in w or "colorfulness" in w:
        return "PRINT_SUSPECT"
    if "blink" in w:
        return "PROMPT_BLINK_FAIL"
    if "pose" in w or "yaw" in w:
        return "PROMPT_POSE_FAIL"
    return "FAIL_OTHER"


def _auth_or_abort(context: grpc.ServicerContext) -> None:
    if not INTERNAL_AUTH_TOKEN:
        return
    md = dict(context.invocation_metadata())
    if md.get("x-internal-auth", "") != INTERNAL_AUTH_TOKEN:
        context.abort(grpc.StatusCode.PERMISSION_DENIED, "forbidden")


def _parse_bind(bind: str) -> str:
    bind = bind.strip()
    return ("0.0.0.0" + bind) if bind.startswith(":") else bind


def _build_ctx() -> core.LivenessContext:
    ctx_id_val: Optional[int] = None
    if str(CTX_ID).strip():
        try:
            ctx_id_val = int(CTX_ID)
        except Exception:
            ctx_id_val = None

    return core.build_context(
        providers_csv=PROVIDERS,
        ctx_id=ctx_id_val,
        det_fast_size=DET_FAST_SIZE,
        det_size=DET_SIZE,
        det_score_thr=DET_SCORE_THR,
        mp_refine_landmarks=True,
        mp_min_det=0.6,
        mp_min_track=0.6,
    )


def _load_clips(clips_dir: Path, preload_max_frames: int, resize_w: int) -> Dict[str, List]:
    if not clips_dir.is_dir():
        raise SystemExit(f"clips-dir not found: {clips_dir}")

    cache: Dict[str, List] = {}
    exts = {".mp4", ".avi", ".mov", ".mkv"}

    for p in sorted(clips_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue

        frames = core.load_clip_frames_bgr(str(p), max_frames=preload_max_frames)
        if not frames:
            raise RuntimeError(f"decoded zero frames: {p}")

        # Optional resize at load time (keeps per-RPC cost constant)
        if resize_w and frames[0].shape[1] > resize_w:
            resized = []
            for fr in frames:
                h, w = fr.shape[:2]
                nh = int(h * (resize_w / float(w)))
                resized.append(cv2.resize(fr, (resize_w, nh), interpolation=cv2.INTER_AREA))
            frames = resized

        cache[p.stem] = frames

    if not cache:
        raise SystemExit(f"No video clips found in {clips_dir}")

    return cache


class LivenessBenchServicer(liveness_pb2_grpc.LivenessBenchServicer):
    def __init__(self, *, replica_id: str, clips: Dict[str, List], ctx: core.LivenessContext):
        self._replica_id = replica_id
        self._clips = clips
        self._ctx = ctx
        self._clip_id_by_lower = {k.lower(): k for k in clips.keys()}
        self._sem = threading.Semaphore(max(1, int(MAX_INFER_CONCURRENCY)))

    def Health(self, request, context):
        return liveness_pb2.HealthResponse(status="ok", replica_id=self._replica_id)

    def GetPolicy(self, request, context):
        _auth_or_abort(context)
        return liveness_pb2.PolicyResponse(
            policy_hash_domain=_POLICY_DOMAIN,
            policy_hash_sha256=_POLICY_HASH,
            policy_json=_POLICY_JSON,
        )

    def ListClips(self, request, context):
        _auth_or_abort(context)
        return liveness_pb2.ListClipsResponse(clip_ids=sorted(self._clips.keys()))

    def CheckByClipId(self, request, context):
        _auth_or_abort(context)

        clip_id = (request.clip_id or "").strip()
        if not clip_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "clip_id required")

        key = self._clip_id_by_lower.get(clip_id.lower())
        if key is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown clip_id: {clip_id}")
        frames = self._clips[key]

        prompt = (request.prompt or "none").strip().lower()
        if prompt not in ("none", "blink", "pose"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid prompt: {prompt}")

        max_frames = int(request.max_frames or 0)
        if max_frames <= 0:
            max_frames = MAX_FRAMES_DEFAULT
        max_frames = max(3, min(max_frames, len(frames)))

        t_total0 = time.perf_counter()
        self._sem.acquire()
        try:
            t0 = time.perf_counter()
            out = core.run_liveness(
                frames_bgr=frames[:max_frames],
                ctx=self._ctx,
                prompt=prompt,
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
            compute_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            self._sem.release()

        total_ms = (time.perf_counter() - t_total0) * 1000.0

        ok = bool(out.get("ok"))
        why = str(out.get("why") or "")
        metrics = dict(out.get("metrics") or {})
        metrics["server_total_ms"] = float(total_ms)

        metrics_clean = {}
        for k, v in metrics.items():
            try:
                metrics_clean[str(k)] = float(v)
            except Exception:
                continue

        return liveness_pb2.CheckByClipIdResponse(
            ok=ok,
            why=why,
            reason_code=_reason_code(why),
            replica_id=self._replica_id,
            server_compute_ms=float(compute_ms),
            metrics=metrics_clean,
            policy_hash_domain=_POLICY_DOMAIN,
            policy_hash_sha256=_POLICY_HASH,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0:9101")
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--replica-id", default=os.environ.get("REPLICA_ID", "replica-1"))
    ap.add_argument("--preload-max-frames", type=int, default=60)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--grpc-workers", type=int, default=0, help="override GRPC_MAX_WORKERS (0=env)")
    args = ap.parse_args()

    bind = _parse_bind(args.bind)
    clips_dir = Path(args.clips_dir)
    grpc_workers = args.grpc_workers if args.grpc_workers > 0 else GRPC_MAX_WORKERS

    ctx = _build_ctx()
    clip_cache = _load_clips(clips_dir, args.preload_max_frames, CLIP_RESIZE_W)

    if args.warmup:
        for frames in clip_cache.values():
            _ = core.run_liveness(
                frames_bgr=frames[: min(20, len(frames))],
                ctx=ctx,
                prompt="none",
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

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max(4, int(grpc_workers))),
        options=[
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
        ],
    )
    liveness_pb2_grpc.add_LivenessBenchServicer_to_server(
        LivenessBenchServicer(replica_id=str(args.replica_id), clips=clip_cache, ctx=ctx),
        server,
    )
    server.add_insecure_port(bind)
    server.start()

    print(
        f"[OK] replica started {bind} replica_id={args.replica_id} clips={len(clip_cache)} "
        f"max_infer_conc={MAX_INFER_CONCURRENCY} grpc_workers={grpc_workers} clip_resize_w={CLIP_RESIZE_W or 'off'}"
    )
    server.wait_for_termination()


if __name__ == "__main__":
    main()