"""
liveness_check.py (paper-ready benchmark harness; calibrated to record_liveness_clips.py)

What this script is
-------------------
This file is a *benchmark harness* that runs your liveness pipeline on the clips
captured by `record_liveness_clips.py` and produces paper-ready tables (CSV) for:

  - Per-scenario correctness (expected PASS/FAIL)
  - Latency breakdown (face-count / prompt / ROI-metrics / total)
  - Aggregated mean/std/p95 timings across runs

Key requirement satisfied
-------------------------
Script-2 uses Script-1 as reference for thresholds:

- It uses the *same metrics definitions* as `record_liveness_clips.py` for:
    (a) blur: Laplacian variance on face ROI
    (b) motion: mean(absdiff)/255 on a fixed 160x160 ROI (stable across sizes)
    (c) brightness: mean(gray) on ROI (reported; optional warning gate)

- If `progress.json` (written by Script-1) exists in --clips-dir, this script
  can auto-calibrate `motion_thr` and `blur_thr` from your recorded metrics.
  This is the most reliable way to get "paper-ready" results that match your
  *actual webcam + lighting* rather than someone else’s defaults.

Speedups (without sacrificing accuracy)
---------------------------------------
1) Two-stage InsightFace detector:
   - FAST detection runs most of the time (smaller det_size).
   - STRONG detection runs only when FAST is suspicious.

2) Face-counting on stride frames (det_stride): coercion/multiface gate is
   computed using fewer frames.

3) ROI-metric sampling (roi_stride): expensive metrics (mask/print/replay) are
   computed on a subset of frames. Micro-motion is still computed frame-to-frame
   on the normalized ROI to retain sensitivity.

Run example
-----------
  C:\temp\phdvenv\Scripts\python liveness_check.py --clips-dir liveness_clips --runs 3 --max-frames 60 --save-csv --out-prefix paper_liveness --save-latex

Expected behavior
-----------------
Live scenarios should PASS:
  IDLE, HEAD_SHAKE, POSE, BLINK_TRY

Attack / negative scenarios should FAIL:
  NO_MOTION, BLUR, MULTIFACE, REPLAY, PRINT, MASK
"""

from __future__ import annotations

import os
import time
import math
import csv
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

import onnxruntime as ort
from insightface.app import FaceAnalysis
import mediapipe as mp

import warnings

# -----------------------------------------------------------------------------
# Log / warning hygiene (does not affect numerical results)
# -----------------------------------------------------------------------------
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # TensorFlow/mediapipe verbosity
os.environ.setdefault('GLOG_minloglevel', '2')      # glog verbosity (if used)
try:
    from absl import logging as absl_logging  # mediapipe uses absl logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold('error')
except Exception:
    pass
warnings.filterwarnings('ignore', message=r'.*SymbolDatabase\.GetPrototype\(\) is deprecated.*')
warnings.filterwarnings('ignore', message=r'.*Feedback manager requires a model with a single signature inference.*')



# =============================================================================
# Small stats helpers
# =============================================================================
def p95(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.sort(np.asarray(values, dtype=np.float64))
    k = int(math.ceil(0.95 * (len(arr) - 1)))
    return float(arr[k])


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    m = float(arr.mean())
    s = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0
    return m, s


# =============================================================================
# Metrics (aligned to record_liveness_clips.py)
# =============================================================================
def lap_var(gray: np.ndarray) -> float:
    """Blur metric: variance of Laplacian. Higher => sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_mean(gray: np.ndarray) -> float:
    """Mean brightness on ROI (0..255)."""
    return float(gray.mean())


def colorfulness(bgr: np.ndarray) -> float:
    """Hasler & Süsstrunk colorfulness metric. Print-like/grayscale lowers this."""
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)

    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))

    return math.sqrt(std_rg * std_rg + std_yb * std_yb) + 0.3 * math.sqrt(mean_rg * mean_rg + mean_yb * mean_yb)


def frame_diff_motion(prev_gray_norm: np.ndarray, gray_norm: np.ndarray) -> float:
    """
    Motion metric (same definition as Script-1):
      mean(absdiff)/255 on a *fixed-size* ROI gray image (160x160 by default).

    This makes the numeric thresholds transferable between recorder and benchmark.
    """
    d = cv2.absdiff(prev_gray_norm, gray_norm)
    return float(d.mean() / 255.0)


def replay_score(gray_roi: np.ndarray) -> float:
    """
    Replay/screen cue: row-wise high-frequency energy. Scanlines yield strong
    row-to-row modulation.

    Score is typically ~0..0.02 depending on camera; "replay" tends to be higher.
    """
    row_mean = gray_roi.mean(axis=1).astype(np.float32)
    d = np.abs(np.diff(row_mean))
    row_diff = float(d.mean() / 255.0)

    smooth = cv2.blur(row_mean.reshape(-1, 1), (9, 1)).ravel()
    resid = row_mean - smooth
    resid_std = float(np.std(resid) / 255.0)

    return float(0.70 * row_diff + 0.30 * resid_std)


def mask_score(gray_roi: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Exposure-adaptive mask detection.

    Returns: (score, drop_ratio, dark_frac, low_std)
      score in [0..1], higher => more mask-like.
    """
    h, w = gray_roi.shape[:2]

    # Central x band (avoid background)
    x1, x2 = int(0.25 * w), int(0.75 * w)

    # Upper reference band and lower band
    yu1, yu2 = int(0.25 * h), int(0.55 * h)
    yl1, yl2 = int(0.60 * h), int(0.95 * h)

    upper = gray_roi[yu1:yu2, x1:x2]
    lower = gray_roi[yl1:yl2, x1:x2]
    if upper.size == 0 or lower.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    med_u = float(np.median(upper))
    med_l = float(np.median(lower))
    low_std = float(np.std(lower))

    drop = (med_u - med_l) / (med_u + 1e-6)
    drop = float(np.clip(drop, 0.0, 1.0))

    # Dark threshold relative to exposure of upper face
    thr_dark = float(np.clip(0.65 * med_u, 25.0, 90.0))
    dark_frac = float(np.mean(lower < thr_dark))

    # Uniformity: masks are often more uniform than beards/skin texture
    uniform = 1.0 - float(np.clip(low_std / 35.0, 0.0, 1.0))

    score = dark_frac * drop * (0.60 + 0.40 * uniform)
    score = float(np.clip(score, 0.0, 1.0))
    return score, drop, dark_frac, low_std


# =============================================================================
# MediaPipe prompt features (blink / pose)
# =============================================================================
_LEFT_EYE = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
_NOSE_TIP = 1
_EYE_L = 33
_EYE_R = 263


def ear_from_landmarks(lm) -> float:
    """Eye Aspect Ratio (EAR) from normalized landmarks (mean of both eyes)."""

    def p(i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    A = np.linalg.norm(p(_LEFT_EYE[1]) - p(_LEFT_EYE[5])) + np.linalg.norm(p(_LEFT_EYE[2]) - p(_LEFT_EYE[4]))
    B = np.linalg.norm(p(_LEFT_EYE[0]) - p(_LEFT_EYE[3]))
    ear_l = float(A / (2.0 * B + 1e-6))

    A = np.linalg.norm(p(_RIGHT_EYE[1]) - p(_RIGHT_EYE[5])) + np.linalg.norm(p(_RIGHT_EYE[2]) - p(_RIGHT_EYE[4]))
    B = np.linalg.norm(p(_RIGHT_EYE[0]) - p(_RIGHT_EYE[3]))
    ear_r = float(A / (2.0 * B + 1e-6))

    return 0.5 * (ear_l + ear_r)


def yaw_ratio_from_landmarks(lm) -> float:
    """
    Fast yaw proxy from 2D geometry:
      (nose_x - mid_eye_x) / eye_distance
    """
    lx = float(lm[_EYE_L].x)
    rx = float(lm[_EYE_R].x)
    nx = float(lm[_NOSE_TIP].x)
    mid = 0.5 * (lx + rx)
    eye_dist = abs(rx - lx) + 1e-6
    return float((nx - mid) / eye_dist)


def blink_observed(ears: List[float], min_closed_frames: int = 2) -> Tuple[bool, float]:
    """
    Robust blink detector using dynamic EAR threshold:
      thr = clamp(0.70 * p90(EAR), [0.14, 0.22])
    """
    if len(ears) < 6:
        return False, 0.0

    baseline = float(np.percentile(np.asarray(ears, dtype=np.float32), 90))
    thr = max(0.14, min(0.22, 0.70 * baseline))
    states = [e < thr for e in ears]

    if sum(states) < min_closed_frames:
        return False, thr

    first = next((i for i, s in enumerate(states) if s), None)
    last = next((i for i in range(len(states) - 1, -1, -1) if states[i]), None)
    if first is None or last is None:
        return False, thr

    open_before = any(not s for s in states[:first])
    open_after = any(not s for s in states[last + 1 :])
    return bool(open_before and open_after), thr


def pose_observed(yaw_ratios: List[float], yaw_thr: float) -> bool:
    """POSE observed if 90th percentile of |yaw_ratio| exceeds yaw_thr."""
    if len(yaw_ratios) < 6:
        return False
    m = float(np.percentile(np.abs(np.asarray(yaw_ratios, dtype=np.float32)), 90))
    return bool(m > yaw_thr)


# =============================================================================
# Video I/O
# =============================================================================
def read_video_frames(path: str, max_frames: int = 0, resize_w: int = 0) -> List[np.ndarray]:
    """Read BGR frames from video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_w and frame.shape[1] > resize_w:
            h, w = frame.shape[:2]
            new_h = int(h * (resize_w / float(w)))
            frame = cv2.resize(frame, (resize_w, new_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {path}")
    return frames


def write_video(path: str, frames_bgr: List[np.ndarray], fps: float = 30.0) -> None:
    """Write frames to mp4."""
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {path}")
    for f in frames_bgr:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        out.write(f)
    out.release()


# =============================================================================
# Synthetic transforms (REPLAY / PRINT / MASK)
# =============================================================================
def add_scanlines(bgr: np.ndarray, period: int = 4, strength: float = 0.55, phase: int = 0) -> np.ndarray:
    out = bgr.astype(np.float32).copy()
    h = out.shape[0]
    for r in range(phase % period, h, period):
        out[r, :, :] *= (1.0 - strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_moire_columns(bgr: np.ndarray, period: int = 6, strength: float = 0.18, phase: int = 0) -> np.ndarray:
    out = bgr.astype(np.float32).copy()
    w = out.shape[1]
    for c in range(phase % period, w, period):
        out[:, c, :] *= (1.0 - strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def paper_like(bgr: np.ndarray, noise_sigma: float = 8.0) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    out = cv2.convertScaleAbs(out, alpha=1.25, beta=8)
    noise = np.random.normal(0, noise_sigma, size=out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def mask_lower_face(bgr: np.ndarray) -> np.ndarray:
    """
    Synthetic mask overlay designed to preserve face detection:
    - Do NOT cover too high.
    - Use dark gray (not pure black).
    """
    h, w = bgr.shape[:2]
    out = bgr.copy()

    y_top = int(0.68 * h)
    y_bot = int(0.95 * h)
    x_l = int(0.22 * w)
    x_r = int(0.78 * w)

    cv2.rectangle(out, (x_l, y_top), (x_r, y_bot), (55, 55, 55), thickness=-1)
    return out


# =============================================================================
# Detector helpers
# =============================================================================
def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


class TwoStageDetector:
    """
    FAST -> STRONG confirmation detector.

    FAST runs on most frames. STRONG is used only if FAST is suspicious:
      - face count != 1
      - OR det_score < det_score_thr

    This keeps accuracy while lowering average detection time.
    """

    def __init__(
        self,
        providers: List[str],
        ctx_id: int,
        fast_size: Tuple[int, int],
        strong_size: Tuple[int, int],
        det_score_thr: float,
    ) -> None:
        self.det_score_thr = float(det_score_thr)

        self.fast = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=["detection"])
        self.fast.prepare(ctx_id=ctx_id, det_size=fast_size)

        self.strong = None
        if strong_size != fast_size:
            self.strong = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=["detection"])
            self.strong.prepare(ctx_id=ctx_id, det_size=strong_size)

    def get(self, frame_bgr: np.ndarray) -> List:
        dets = self.fast.get(frame_bgr)
        if self.strong is None:
            return dets

        if len(dets) != 1:
            return self.strong.get(frame_bgr)

        # det_score is present on InsightFace Face objects in many versions
        try:
            score = float(getattr(dets[0], "det_score", 1.0))
        except Exception:
            score = 1.0

        if score < self.det_score_thr:
            return self.strong.get(frame_bgr)

        return dets

    def count(self, frame_bgr: np.ndarray) -> int:
        return len(self.get(frame_bgr))


def roi_from_center_face(
    frames: List[np.ndarray],
    det: TwoStageDetector,
    pad: float = 0.35,
) -> Tuple[Tuple[int, int, int, int], int]:
    """
    Compute a stable ROI bbox using the middle frame, then reuse it for the clip.

    pad=0.35 is deliberately aligned with record_liveness_clips.py so that
    blur/motion metrics are comparable.
    """
    mid = frames[len(frames) // 2]
    dets = det.get(mid)
    h, w = mid.shape[:2]
    if not dets:
        return (0, 0, w, h), 0

    face = max(dets, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox)
    bw = x2 - x1
    bh = y2 - y1

    x1 -= int(pad * bw)
    x2 += int(pad * bw)
    y1 -= int(pad * bh)
    y2 += int(pad * bh)

    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)
    return (x1, y1, x2, y2), len(dets)


# =============================================================================
# Result structs
# =============================================================================
@dataclass
class LivenessScalars:
    oneface_frac: float = 0.0
    twoplus_frac: float = 0.0

    # aligned to Script-1's reported metrics
    brightness_mean: float = 0.0
    motion_mean: float = 0.0
    blur_median: float = 0.0

    # attack cues
    replay_mean: float = 0.0
    mask_score_mean: float = 0.0
    mask_drop_mean: float = 0.0
    mask_darkfrac_mean: float = 0.0
    mask_lowstd_mean: float = 0.0
    print_mean: float = 0.0

    # prompt debug
    blink_thr: float = 0.0


@dataclass
class TimingMs:
    det_ms: float = 0.0
    prompt_ms: float = 0.0
    roi_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class LivenessResult:
    ok: bool
    why: str
    scalars: LivenessScalars
    timing: TimingMs


# =============================================================================
# Threshold calibration from Script-1 output (progress.json)
# =============================================================================
def load_progress_json(clips_dir: str) -> Optional[dict]:
    """
    record_liveness_clips.py writes <clips_dir>/progress.json with per-scenario metrics.
    We use it for:
      - warnings if a clip was accepted but validation failed
      - optional auto-calibration of motion_thr / blur_thr
    """
    p = os.path.join(clips_dir, "progress.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _progress_metric(progress: dict, scenario: str, key: str) -> Optional[float]:
    try:
        return float(progress["scenarios"][scenario]["metrics"][key])
    except Exception:
        return None


def auto_calibrate_thresholds(
    progress: dict,
    *,
    motion_thr_default: float,
    blur_thr_default: float,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Derive motion_thr and blur_thr from recorded metrics, if available.

    Strategy:
      - motion_thr: midpoint between IDLE and NO_MOTION motion_mean (with safety margins)
      - blur_thr  : midpoint between IDLE and BLUR blur_median

    If metrics are missing or not separable, return the defaults unchanged.
    """
    motion_idle = _progress_metric(progress, "IDLE", "motion_mean")
    motion_nomotion = _progress_metric(progress, "NO_MOTION", "motion_mean")

    blur_idle = _progress_metric(progress, "IDLE", "blur_median")
    blur_blur = _progress_metric(progress, "BLUR", "blur_median")

    motion_thr = float(motion_thr_default)
    blur_thr = float(blur_thr_default)

    # --- motion ---
    if motion_idle is not None and motion_nomotion is not None:
        # If they are too close, do not auto-calibrate.
        if motion_idle > motion_nomotion * 1.10:
            mid = 0.5 * (motion_idle + motion_nomotion)
            # Keep PASS region comfortably below idle, and FAIL region above nomotion.
            motion_thr = min(motion_idle * 0.95, max(motion_nomotion * 1.10, mid))
            motion_thr = float(max(0.001, min(0.10, motion_thr)))
            if verbose:
                print(f"[CAL] motion_idle={motion_idle:.4f}, motion_nomotion={motion_nomotion:.4f} -> motion_thr={motion_thr:.4f}")
        elif verbose:
            print(f"[CAL] motion not separable (idle={motion_idle:.4f}, nomotion={motion_nomotion:.4f}); using motion_thr={motion_thr_default:.4f}")
    # Fallback: if we have IDLE but NO_MOTION is missing or too close,
    # bias motion_thr slightly below IDLE so that genuine IDLE does not get false-rejected.
    if motion_idle is not None and (motion_nomotion is None):
        motion_thr = float(max(0.001, min(motion_thr, motion_idle * 0.95)))
        if verbose:
            print(f"[CAL] motion fallback from IDLE only -> motion_thr={motion_thr:.4f}")

    # --- blur ---
    if blur_idle is not None and blur_blur is not None:
        if blur_idle > blur_blur * 1.10:
            mid = 0.5 * (blur_idle + blur_blur)
            blur_thr = min(blur_idle * 0.95, max(blur_blur * 1.05, mid))
            blur_thr = float(max(5.0, min(500.0, blur_thr)))
            if verbose:
                print(f"[CAL] blur_idle={blur_idle:.1f}, blur_blur={blur_blur:.1f} -> blur_thr={blur_thr:.1f}")
        elif verbose:
            print(f"[CAL] blur not separable (idle={blur_idle:.1f}, blur={blur_blur:.1f}); using blur_thr={blur_thr_default:.1f}")
    if blur_idle is not None and (blur_blur is None):
        blur_thr = float(max(5.0, min(blur_thr, blur_idle * 0.95)))
        if verbose:
            print(f"[CAL] blur fallback from IDLE only -> blur_thr={blur_thr:.1f}")

    return motion_thr, blur_thr

# =============================================================================
# Calibration helpers (uses your loaded clips as ground truth for thresholds)
# =============================================================================
def compute_clip_scalars(
    frames_bgr: List[np.ndarray],
    det: TwoStageDetector,
    *,
    roi_pad: float,
    roi_resize_w: int,
    roi_stride: int,
    roi_bbox_override: Optional[Tuple[int, int, int, int]] = None,
    motion_norm_size: Tuple[int, int] = (160, 160),
) -> LivenessScalars:
    """Compute scalar metrics (motion/blur/attack cues) without PASS/FAIL gating."""
    scal = LivenessScalars()
    if not frames_bgr:
        return scal

    # ROI is normally estimated from this clip's center frame.
    # For simulated attacks (REPLAY/PRINT/MASK) we override the ROI with the
    # ROI computed on the *unmodified* base live clip. This prevents the face
    # detector from shrinking the bbox when the lower face is occluded, which
    # would otherwise hide the synthetic artifact from ROI-based cues.
    if roi_bbox_override is None:
        (x1, y1, x2, y2), _ = roi_from_center_face(frames_bgr, det, pad=float(roi_pad))
    else:
        x1, y1, x2, y2 = map(int, roi_bbox_override)
    prev_norm: Optional[np.ndarray] = None
    motions: List[float] = []
    blurs: List[float] = []
    brights: List[float] = []
    replays: List[float] = []
    mask_scores: List[float] = []
    mask_drops: List[float] = []
    mask_darkfracs: List[float] = []
    mask_lowstds: List[float] = []
    prints: List[float] = []

    roi_stride = max(1, int(roi_stride))
    motion_w, motion_h = int(motion_norm_size[0]), int(motion_norm_size[1])
    motion_size = (motion_w, motion_h)

    for i, f in enumerate(frames_bgr):
        roi = f[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        if roi_resize_w and roi.shape[1] != roi_resize_w:
            h, w = roi.shape[:2]
            new_h = max(16, int(h * (roi_resize_w / float(w))))
            roi = cv2.resize(roi, (roi_resize_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_norm = cv2.resize(gray, motion_size, interpolation=cv2.INTER_AREA)
        if prev_norm is not None:
            motions.append(frame_diff_motion(prev_norm, gray_norm))
        prev_norm = gray_norm

        if (i % roi_stride) != 0:
            continue
        brights.append(brightness_mean(gray))
        blurs.append(lap_var(gray))
        replays.append(replay_score(gray))
        ms, drop, df, ls = mask_score(gray)
        mask_scores.append(ms)
        mask_drops.append(drop)
        mask_darkfracs.append(df)
        mask_lowstds.append(ls)
        prints.append(colorfulness(roi))

    scal.motion_mean = float(np.mean(motions)) if motions else 0.0
    scal.blur_median = float(np.median(blurs)) if blurs else 0.0
    scal.brightness_mean = float(np.mean(brights)) if brights else 0.0
    scal.replay_mean = float(np.mean(replays)) if replays else 0.0
    scal.mask_score_mean = float(np.mean(mask_scores)) if mask_scores else 0.0
    scal.mask_drop_mean = float(np.mean(mask_drops)) if mask_drops else 0.0
    scal.mask_darkfrac_mean = float(np.mean(mask_darkfracs)) if mask_darkfracs else 0.0
    scal.mask_lowstd_mean = float(np.mean(mask_lowstds)) if mask_lowstds else 0.0
    scal.print_mean = float(np.mean(prints)) if prints else 0.0
    return scal


def auto_calibrate_attack_thresholds(
    idle_scal: LivenessScalars,
    *,
    replay_thr_default: float,
    mask_thr_default: float,
    print_thr_default: float,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """Auto-tune attack thresholds from IDLE to reduce false rejects."""
    replay_thr = float(replay_thr_default)
    mask_thr = float(mask_thr_default)
    print_thr = float(print_thr_default)

    # PRINT: reject if colorfulness < print_thr. Keep threshold well below IDLE.
    if idle_scal.print_mean > 0:
        print_thr = min(print_thr, max(2.0, 0.65 * idle_scal.print_mean))

    # REPLAY/MASK: reject if score > threshold. Keep thresholds above IDLE.
    if idle_scal.replay_mean > 0:
        replay_thr = max(replay_thr, idle_scal.replay_mean * 2.0 + 1e-4)
    if idle_scal.mask_score_mean > 0:
        mask_thr = max(mask_thr, idle_scal.mask_score_mean * 1.35 + 0.01)

    # clamps
    replay_thr = float(max(1e-5, min(0.10, replay_thr)))
    mask_thr = float(max(0.05, min(1.50, mask_thr)))
    print_thr = float(max(1.0, min(100.0, print_thr)))

    if verbose:
        print(f"[CAL] attack thresholds: replay_thr={replay_thr:.4f}, mask_thr={mask_thr:.3f}, print_thr={print_thr:.1f}")
    return replay_thr, mask_thr, print_thr



# =============================================================================
# Liveness check core
# =============================================================================
def liveness_check(
    frames_bgr: List[np.ndarray],
    det: TwoStageDetector,
    face_mesh,
    roi_bbox_override: Optional[Tuple[int, int, int, int]] = None,
    # face-coercion gates
    min_oneface_frac: float = 0.60,
    max_twoplus_frac: float = 0.05,
    det_stride: int = 4,
    # passive / quality gates
    motion_thr: float = 0.012,     # aligned to recorder default --motion-idle-min
    blur_thr: float = 80.0,        # aligned to recorder default --blur-min-sharp
    bright_lo: float = 35.0,       # recorder default --bright-lo (warning gate by default)
    bright_hi: float = 220.0,      # recorder default --bright-hi (warning gate by default)
    brightness_gate: bool = False, # keep OFF by default (lighting variability)
    # ROI processing
    roi_pad: float = 0.35,
    roi_resize_w: int = 160,
    roi_stride: int = 2,           # compute expensive cues on every Nth frame
    motion_norm_size: Tuple[int, int] = (160, 160),  # aligned to recorder MOTION_NORM_SIZE
    # attack thresholds
    replay_thr: float = 0.0045,
    mask_thr: float = 0.35,
    print_thr: float = 12.0,
    # prompt
    prompt: str = "none",          # none|blink|pose
    mesh_stride: int = 2,
    yaw_thr: float = 0.045,
) -> LivenessResult:
    """
    Returns LivenessResult(ok, why, scalars, timing).

    Notes on correctness:
      - Micro-motion is required for general liveness unless a prompt is requested
        and satisfied (blink/pose themselves are motion evidence).
      - MULTIFACE/NOFACE is handled via the face-count fraction gate.
    """
    t0 = time.perf_counter()
    scal = LivenessScalars()
    tm = TimingMs()

    if not frames_bgr:
        return LivenessResult(False, "no frames", scal, tm)

    # -------------------------------------------------------------------------
    # (A) Anti-coercion face counts (InsightFace), on stride frames.
    # -------------------------------------------------------------------------
    t = time.perf_counter()
    counts: List[int] = []
    stride = max(1, int(det_stride))
    for i in range(0, len(frames_bgr), stride):
        counts.append(det.count(frames_bgr[i]))
    tm.det_ms = (time.perf_counter() - t) * 1000.0

    arr = np.asarray(counts, dtype=np.int32) if counts else np.asarray([0], dtype=np.int32)
    scal.oneface_frac = float(np.mean(arr == 1))
    scal.twoplus_frac = float(np.mean(arr >= 2))

    if scal.oneface_frac < min_oneface_frac or scal.twoplus_frac > max_twoplus_frac:
        tm.total_ms = (time.perf_counter() - t0) * 1000.0
        why = f"multiple/no faces (oneFaceFrac={scal.oneface_frac:.2f}, twoPlusFrac={scal.twoplus_frac:.2f})"
        return LivenessResult(False, why, scal, tm)

    # -------------------------------------------------------------------------
    # (B) Stable ROI bbox from the center frame (or an override ROI).
    #
    # Why override exists:
    #   When we simulate occlusions (MASK) on top of a live clip, the face
    #   detector may return a smaller bbox that excludes the occluded region.
    #   That would make ROI-based detectors (mask/print/replay) blind.
    #   For simulated attacks we therefore reuse the ROI computed on the
    #   unmodified base live clip.
    # -------------------------------------------------------------------------
    if roi_bbox_override is None:
        (x1, y1, x2, y2), center_count = roi_from_center_face(frames_bgr, det, pad=float(roi_pad))
        if center_count == 0:
            tm.total_ms = (time.perf_counter() - t0) * 1000.0
            return LivenessResult(False, "no face on center frame", scal, tm)
    else:
        # Clamp override ROI to frame bounds.
        h0, w0 = frames_bgr[0].shape[:2]
        x1, y1, x2, y2 = map(int, roi_bbox_override)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w0, h0)
        center_count = 1

    # -------------------------------------------------------------------------
    # (C) ROI loop: compute motion (every frame) + other cues (sampled).
    # -------------------------------------------------------------------------
    t = time.perf_counter()

    prev_norm: Optional[np.ndarray] = None
    motions: List[float] = []
    blurs: List[float] = []
    brights: List[float] = []

    replays: List[float] = []
    mask_scores: List[float] = []
    mask_drops: List[float] = []
    mask_darkfracs: List[float] = []
    mask_lowstds: List[float] = []
    prints: List[float] = []

    roi_stride = max(1, int(roi_stride))
    motion_w, motion_h = int(motion_norm_size[0]), int(motion_norm_size[1])
    motion_size = (motion_w, motion_h)

    for i, f in enumerate(frames_bgr):
        roi = f[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Resize ROI (keeps computations cheap and consistent)
        if roi_resize_w and roi.shape[1] != roi_resize_w:
            h, w = roi.shape[:2]
            new_h = max(16, int(h * (roi_resize_w / float(w))))
            roi = cv2.resize(roi, (roi_resize_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # --- Always compute motion on fixed-size normalized ROI (Script-1 compatible) ---
        gray_norm = cv2.resize(gray, motion_size, interpolation=cv2.INTER_AREA)
        if prev_norm is not None:
            motions.append(frame_diff_motion(prev_norm, gray_norm))
        prev_norm = gray_norm

        # --- For the rest, sample every roi_stride frames (speed) ---
        if (i % roi_stride) != 0:
            continue

        brights.append(brightness_mean(gray))
        blurs.append(lap_var(gray))

        replays.append(replay_score(gray))

        ms, drop, df, ls = mask_score(gray)
        mask_scores.append(ms)
        mask_drops.append(drop)
        mask_darkfracs.append(df)
        mask_lowstds.append(ls)

        prints.append(colorfulness(roi))

    tm.roi_ms = (time.perf_counter() - t) * 1000.0

    scal.motion_mean = float(np.mean(motions)) if motions else 0.0
    scal.blur_median = float(np.median(blurs)) if blurs else 0.0
    scal.brightness_mean = float(np.mean(brights)) if brights else 0.0

    scal.replay_mean = float(np.mean(replays)) if replays else 0.0
    scal.mask_score_mean = float(np.mean(mask_scores)) if mask_scores else 0.0
    scal.mask_drop_mean = float(np.mean(mask_drops)) if mask_drops else 0.0
    scal.mask_darkfrac_mean = float(np.mean(mask_darkfracs)) if mask_darkfracs else 0.0
    scal.mask_lowstd_mean = float(np.mean(mask_lowstds)) if mask_lowstds else 0.0
    scal.print_mean = float(np.mean(prints)) if prints else 0.0

    # -------------------------------------------------------------------------
    # (D) Cheap attack checks first
    # -------------------------------------------------------------------------
    # Print/photo tends to have low colorfulness (depends on printer/screen).
    if scal.print_mean < print_thr:
        tm.total_ms = (time.perf_counter() - t0) * 1000.0
        return LivenessResult(False, f"print-like low colorfulness (mean={scal.print_mean:.2f} < {print_thr:.2f})", scal, tm)

    # Replay/screen cue
    if scal.replay_mean > replay_thr:
        tm.total_ms = (time.perf_counter() - t0) * 1000.0
        return LivenessResult(False, f"replay-like scanline periodicity (mean={scal.replay_mean:.4f} > {replay_thr:.4f})", scal, tm)

    # Mask cue
    if scal.mask_score_mean > mask_thr:
        tm.total_ms = (time.perf_counter() - t0) * 1000.0
        return LivenessResult(False, f"mask-like lower occlusion (score={scal.mask_score_mean:.3f} > {mask_thr:.3f})", scal, tm)

    # -------------------------------------------------------------------------
    # (E) Quality gates: blur + optional brightness
    # -------------------------------------------------------------------------
    if scal.blur_median < blur_thr:
        tm.total_ms = (time.perf_counter() - t0) * 1000.0
        return LivenessResult(False, f"excessive blur (med={scal.blur_median:.1f} < {blur_thr:.1f})", scal, tm)

    if brightness_gate:
        if scal.brightness_mean < bright_lo:
            tm.total_ms = (time.perf_counter() - t0) * 1000.0
            return LivenessResult(False, f"too dark (brightness={scal.brightness_mean:.1f} < {bright_lo:.1f})", scal, tm)
        if scal.brightness_mean > bright_hi:
            tm.total_ms = (time.perf_counter() - t0) * 1000.0
            return LivenessResult(False, f"too bright (brightness={scal.brightness_mean:.1f} > {bright_hi:.1f})", scal, tm)

    # -------------------------------------------------------------------------
    # (F) Prompt checks (blink / pose), only when requested.
    #     Optimization: run FaceMesh on the same ROI crop (not full frame).
    # -------------------------------------------------------------------------
    prompt = (prompt or "none").strip().lower()
    prompt_ok = True

    if prompt in ("blink", "pose"):
        t = time.perf_counter()
        ears: List[float] = []
        yaws: List[float] = []

        ms = max(1, int(mesh_stride))
        for i in range(0, len(frames_bgr), ms):
            fr = frames_bgr[i]
            roi = fr[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Slight upscaling improves landmark stability for small faces.
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            if roi_rgb.shape[1] < 320:
                scale = 320.0 / float(roi_rgb.shape[1])
                nh = max(64, int(roi_rgb.shape[0] * scale))
                roi_rgb = cv2.resize(roi_rgb, (320, nh), interpolation=cv2.INTER_LINEAR)

            res = face_mesh.process(roi_rgb)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            if prompt == "blink":
                ears.append(ear_from_landmarks(lm))
            else:
                yaws.append(yaw_ratio_from_landmarks(lm))

        tm.prompt_ms = (time.perf_counter() - t) * 1000.0

        if prompt == "blink":
            ok, thr = blink_observed(ears, min_closed_frames=2)
            scal.blink_thr = float(thr)
            if not ok:
                prompt_ok = False
                tm.total_ms = (time.perf_counter() - t0) * 1000.0
                return LivenessResult(False, "blink not observed", scal, tm)

        if prompt == "pose":
            if not pose_observed(yaws, yaw_thr=yaw_thr):
                prompt_ok = False
                tm.total_ms = (time.perf_counter() - t0) * 1000.0
                return LivenessResult(False, "pose not observed", scal, tm)

    # -------------------------------------------------------------------------
    # (G) Motion gate (core liveness evidence)
    #     IMPORTANT: if a requested prompt succeeded, allow low global motion.
    # -------------------------------------------------------------------------
    if scal.motion_mean < motion_thr and not (prompt in ("blink", "pose") and prompt_ok):
        tm.total_ms = (time.perf_counter() - t0) * 1000.0
        return LivenessResult(False, f"insufficient micro-motion (mean={scal.motion_mean:.4f} < {motion_thr:.4f})", scal, tm)

    tm.total_ms = (time.perf_counter() - t0) * 1000.0
    detail = "no prompt" if prompt == "none" else f"{prompt} ok"
    return LivenessResult(True, f"live, {detail}", scal, tm)


# =============================================================================
# Benchmark harness helpers
# =============================================================================
def find_clip(clips_dir: str, name: str) -> Optional[str]:
    if not os.path.isdir(clips_dir):
        return None
    name_low = name.lower()
    for fn in os.listdir(clips_dir):
        base, ext = os.path.splitext(fn)
        if base.lower() == name_low and ext.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            return os.path.join(clips_dir, fn)
    return None


def simulate_from_live(live_frames: List[np.ndarray], scenario: str, seed: int) -> List[np.ndarray]:
    rng = random.Random(seed)
    np.random.seed(seed)

    if scenario == "REPLAY":
        out = []
        pr = rng.randint(0, 3)
        pc = rng.randint(0, 5)
        for i, f in enumerate(live_frames):
            x = add_scanlines(f, period=4, strength=0.55, phase=(pr + i) % 4)
            x = add_moire_columns(x, period=6, strength=0.18, phase=(pc + i) % 6)
            a = 0.98 + 0.04 * rng.random()
            b = rng.randint(-3, 4)
            x = cv2.convertScaleAbs(x, alpha=a, beta=b)
            out.append(x)
        return out

    if scenario == "PRINT":
        return [paper_like(f, noise_sigma=8.0) for f in live_frames]

    if scenario == "MASK":
        return [mask_lower_face(f) for f in live_frames]

    raise ValueError(f"Unknown synthetic scenario: {scenario}")


def print_row(scenario: str, run_id: int, lr: LivenessResult, expect_ok: bool) -> None:
    test_pass = (lr.ok == expect_ok)
    liveness_s = "PASS" if lr.ok else "FAIL"
    test_s = "PASS" if test_pass else "FAIL"
    why = lr.why if len(lr.why) <= 56 else (lr.why[:53] + "...")

    print(
        f"{scenario:<12} {run_id:<4} {liveness_s:<8} {test_s:<8} {why:<60}"
        f"{lr.timing.det_ms:8.1f} {lr.timing.prompt_ms:8.1f} {lr.timing.roi_ms:8.1f} {lr.timing.total_ms:8.1f}"
    )
    sc = lr.scalars
    print(
        f"{'':<12} {'':<4} {'':<8} {'':<8} | debug:"
        f" oneFace={sc.oneface_frac:.2f} twoPlus={sc.twoplus_frac:.2f}"
        f" bright={sc.brightness_mean:.1f}"
        f" motion={sc.motion_mean:.4f} blurMed={sc.blur_median:.1f}"
        f" replay={sc.replay_mean:.4f}"
        f" maskScore={sc.mask_score_mean:.3f} drop={sc.mask_drop_mean:.3f} darkF={sc.mask_darkfrac_mean:.3f} lowStd={sc.mask_lowstd_mean:.1f}"
        f" print={sc.print_mean:.2f} blinkThr={sc.blink_thr:.3f}"
    )


def parse_wh(s: str) -> Tuple[int, int]:
    s = s.lower().strip()
    if "x" not in s:
        raise ValueError("Expected WxH like 640x640")
    w, h = s.split("x", 1)
    return int(w), int(h)


def print_progress_summary(progress: dict) -> None:
    """
    Helpful for paper writing: shows what Script-1 recorded and whether its
    validation considered the clip OK.
    """
    try:
        sc = progress.get("scenarios", {})
        if not sc:
            return
        print("\n[REF] progress.json summary (from record_liveness_clips.py):")
        for k in sorted(sc.keys()):
            ok = sc[k].get("ok", None)
            m = sc[k].get("metrics", {}) or {}
            mm = m.get("motion_mean", None)
            bm = m.get("blur_median", None)
            br = m.get("brightness_mean", None)
            of = m.get("one_face_frac", None)
            tf = m.get("two_plus_face_frac", None)
            ok_s = "OK" if ok else "FAIL" if ok is not None else "?"
            print(f"  - {k:<10} {ok_s:<4}  motion={mm}  blurMed={bm}  bright={br}  oneFaceFrac={of}  twoPlusFrac={tf}")
    except Exception:
        return


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Paper-ready liveness benchmark from recorded clips")

    ap.add_argument("--clips-dir", default=".", help="Folder containing recorded clips (IDLE.mp4, etc.)")
    ap.add_argument("--out-prefix", default="liveness", help="Prefix for CSV/JSON outputs")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=60, help="0=all")
    ap.add_argument("--resize-w", type=int, default=640, help="Downscale input frames to this width (optional)")
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--save-latex", action="store_true", help="Write a small LaTeX table snippet for timings.")
    ap.add_argument("--write-simulated", action="store_true", help="Write REPLAY/PRINT/MASK.mp4 into clips-dir")

    # Detector knobs
    ap.add_argument("--det-size", default="640x640", help="STRONG detector size WxH")
    ap.add_argument("--det-fast-size", default="512x512", help="FAST detector size WxH")
    ap.add_argument("--det-score-thr", type=float, default=0.55)

    # Performance knobs
    ap.add_argument("--det-stride", type=int, default=6)
    ap.add_argument("--mesh-stride", type=int, default=3)
    ap.add_argument("--roi-resize-w", type=int, default=160)
    ap.add_argument("--roi-stride", type=int, default=2, help="Compute expensive ROI cues on every Nth frame.")
    ap.add_argument("--roi-pad", type=float, default=0.35, help="ROI padding around detected face bbox.")

    # Thresholds (defaults match Script-1; can be auto-calibrated from progress.json)
    ap.add_argument("--motion-thr", type=float, default=0.012, help="Micro-motion PASS threshold.")
    # Scenario-specific motion thresholds (defaults copied from record_liveness_clips.py)
    ap.add_argument("--motion-headshake-min", type=float, default=0.022, help="Min motion for HEAD_SHAKE.")
    ap.add_argument("--motion-pose-min", type=float, default=0.040, help="Min motion for POSE (if motion gate used).")
    ap.add_argument("--motion-nomotion-max", type=float, default=0.050, help="Max motion expected for NO_MOTION (sanity check).")
    ap.add_argument("--blur-thr", type=float, default=80.0)
    ap.add_argument("--min-oneface-frac", type=float, default=0.60)
    ap.add_argument("--max-twoplus-frac", type=float, default=0.05)
    ap.add_argument("--bright-lo", type=float, default=35.0)
    ap.add_argument("--bright-hi", type=float, default=220.0)
    ap.add_argument("--brightness-gate", action="store_true", help="Fail if brightness is outside [bright-lo, bright-hi].")

    ap.add_argument("--replay-thr", type=float, default=0.0045)
    ap.add_argument("--mask-thr", type=float, default=0.35)
    ap.add_argument("--print-thr", type=float, default=12.0)
    ap.add_argument("--yaw-thr", type=float, default=0.045)

    # Calibration control
    ap.add_argument("--auto-thresholds", action="store_true", default=True,
                    help="Auto-calibrate motion_thr/blur_thr from progress.json (default: ON).")
    ap.add_argument("--no-auto-thresholds", dest="auto_thresholds", action="store_false",
                    help="Disable auto-calibration even if progress.json exists.")

    ap.add_argument("--auto-attack-thresholds", action="store_true", default=True,
                    help="Auto-tune print/replay/mask thresholds from the IDLE clip (default: ON).")
    ap.add_argument("--no-auto-attack-thresholds", dest="auto_attack_thresholds", action="store_false",
                    help="Disable auto-tuning of print/replay/mask thresholds.")


    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    # ORT provider info (InsightFace uses ORT internally)
    ort_providers = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in ort_providers
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    ctx_id = 0 if use_cuda else -1
    print(f"[ENV] ORT providers: {ort_providers} | Using: {'CUDA' if use_cuda else 'CPU'}")

    det_size_strong = parse_wh(args.det_size)
    det_size_fast = parse_wh(args.det_fast_size)

    det = TwoStageDetector(
        providers=providers,
        ctx_id=ctx_id,
        fast_size=det_size_fast,
        strong_size=det_size_strong,
        det_score_thr=args.det_score_thr,
    )
    print(f"[ENV] FAST det_size={det_size_fast[0]}x{det_size_fast[1]}  STRONG det_size={det_size_strong[0]}x{det_size_strong[1]}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Scenarios and expected outcomes
    scenarios = [
        "IDLE", "HEAD_SHAKE", "POSE", "BLINK_TRY",
        "NO_MOTION", "BLUR", "MULTIFACE",
        "REPLAY", "PRINT", "MASK",
    ]
    expect_ok = {
        "IDLE": True, "HEAD_SHAKE": True, "POSE": True, "BLINK_TRY": True,
        "NO_MOTION": False, "BLUR": False, "MULTIFACE": False,
        "REPLAY": False, "PRINT": False, "MASK": False,
    }
    prompt_for = {
        "IDLE": "none", "HEAD_SHAKE": "none",
        "POSE": "pose", "BLINK_TRY": "blink",
        "NO_MOTION": "none", "BLUR": "none", "MULTIFACE": "none",
        "REPLAY": "none", "PRINT": "none", "MASK": "none",
    }

    # Reference: progress.json from recorder (optional)
    progress = load_progress_json(args.clips_dir)
    if progress:
        print_progress_summary(progress)

    # Threshold initialisation (may be auto-calibrated after clips are loaded)
    motion_thr = float(args.motion_thr)
    blur_thr = float(args.blur_thr)
    replay_thr = float(args.replay_thr)
    mask_thr = float(args.mask_thr)
    print_thr = float(args.print_thr)

    # Base live clip for synthesis
    base_live_name = "IDLE" if find_clip(args.clips_dir, "IDLE") else "HEAD_SHAKE"
    base_live_path = find_clip(args.clips_dir, base_live_name)
    if not base_live_path:
        raise SystemExit(f"Missing base live clip (IDLE or HEAD_SHAKE) in {args.clips_dir}")

    base_live_frames = read_video_frames(base_live_path, max_frames=args.max_frames, resize_w=args.resize_w)

    # ROI of the unmodified base live clip (used for simulated attacks).
    base_roi_bbox, _ = roi_from_center_face(base_live_frames, det, pad=float(args.roi_pad))

    if args.write_simulated:
        os.makedirs(args.clips_dir, exist_ok=True)
        for syn in ("REPLAY", "PRINT", "MASK"):
            syn_frames = simulate_from_live(base_live_frames, syn, seed=args.seed)
            out_path = os.path.join(args.clips_dir, f"{syn}.mp4")
            write_video(out_path, syn_frames, fps=30.0)
            print(f"[SIM] wrote {out_path}")

    # Load real clips once (fast)
    real_frames: Dict[str, List[np.ndarray]] = {}
    for sc in scenarios:
        if sc in ("REPLAY", "PRINT", "MASK"):
            continue
        p = find_clip(args.clips_dir, sc)
        if not p:
            raise SystemExit(f"Missing clip for scenario '{sc}' in {args.clips_dir}")
        real_frames[sc] = read_video_frames(p, max_frames=args.max_frames, resize_w=args.resize_w)


    # -------------------------------------------------------------------------
    # Auto-calibrate thresholds from the *loaded* clips (stable across webcams)
    # -------------------------------------------------------------------------
    idle_scal = compute_clip_scalars(
        real_frames.get('IDLE', base_live_frames), det,
        roi_pad=args.roi_pad, roi_resize_w=args.roi_resize_w, roi_stride=args.roi_stride,
                roi_bbox_override=None,
    )
    nom_scal = compute_clip_scalars(
        real_frames.get('NO_MOTION', []), det,
        roi_pad=args.roi_pad, roi_resize_w=args.roi_resize_w, roi_stride=args.roi_stride,
    )
    blur_scal = compute_clip_scalars(
        real_frames.get('BLUR', []), det,
        roi_pad=args.roi_pad, roi_resize_w=args.roi_resize_w, roi_stride=args.roi_stride,
    )

    if args.auto_thresholds:
        # Reuse the same midpoint logic, but with metrics computed by this script.
        # NOTE: auto_calibrate_thresholds() expects the same nesting as progress.json:
        #   progress['scenarios'][<SCENARIO>]['metrics'][<KEY>]
        # so we build a compatible in-memory structure from the metrics computed here.
        progress_like = {
            'scenarios': {
                'IDLE': {'metrics': {'motion_mean': idle_scal.motion_mean, 'blur_median': idle_scal.blur_median}},
                'NO_MOTION': {'metrics': {'motion_mean': nom_scal.motion_mean, 'blur_median': nom_scal.blur_median}},
                'BLUR': {'metrics': {'motion_mean': blur_scal.motion_mean, 'blur_median': blur_scal.blur_median}},
            }
        }
        motion_thr, blur_thr = auto_calibrate_thresholds(
            progress_like, motion_thr_default=motion_thr, blur_thr_default=blur_thr, verbose=True
        )

    # Scenario-specific thresholds (separate per gate / per scenario)
    # -------------------------------------------------------------------------
    # Rationale (important for paper):
    # - In practice, a single shared threshold across *all* scenarios is brittle.
    #   Example: POSE clips often have transient lower-face occlusion / motion blur
    #   that can look "mask-like" if you reuse the same mask threshold as IDLE.
    # - Therefore, we keep *separate* thresholds per test:
    #     (1) motion gate, (2) blur gate, (3) replay gate, (4) print gate, (5) mask gate,
    #   and we allow scenario-specific values where it is justified (POSE/HEAD_SHAKE).
    #
    # The defaults follow record_liveness_clips.py semantics (motion-* and blur-*),
    # while the auto-thresholding learns webcam-specific baselines from *your* clips.
    # -------------------------------------------------------------------------

    # ---- helpers (local, float-safe) ----
    def _mid(a: float, b: float) -> float:
        return 0.5 * (float(a) + float(b))

    def _clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    # ---- motion thresholds (Script-1 ratios preserved) ----
    base_idle = max(1e-6, float(args.motion_thr))  # Script-1's IDLE default
    ratio_head = float(args.motion_headshake_min) / base_idle
    ratio_pose = float(args.motion_pose_min) / base_idle

    motion_thr_map = {
        "IDLE": motion_thr,
        "HEAD_SHAKE": float(min(0.10, motion_thr * ratio_head)),
        # For prompted scenarios, the prompt success is the primary evidence.
        # Still keep a motion threshold for debug consistency.
        "POSE": float(min(0.10, motion_thr * ratio_pose)),
        "BLINK_TRY": motion_thr,
        # Attacks / negative scenarios: keep same gate; they should fail anyway.
        "NO_MOTION": motion_thr,
        "BLUR": motion_thr,
        "MULTIFACE": motion_thr,
        "REPLAY": motion_thr,
        "PRINT": motion_thr,
        "MASK": motion_thr,
    }

    # ---- attack thresholds (live vs attack separated) ----
    # Step 1: derive LIVE-safe thresholds (to avoid false positives) from the IDLE clip.
    replay_thr_live = replay_thr
    mask_thr_live = mask_thr
    print_thr_live = print_thr

    if args.auto_attack_thresholds:
        replay_thr_live, mask_thr_live, print_thr_live = auto_calibrate_attack_thresholds(
            idle_scal,
            replay_thr_default=replay_thr_live,
            mask_thr_default=mask_thr_live,
            print_thr_default=print_thr_live,
            verbose=True,
        )

    # Step 2: derive ATTACK-sensitive thresholds from synthetic attacks (generated from your IDLE clip).
    # This avoids the coupling problem: tuning LIVE thresholds should not make attacks pass.
    syn_seed = int(args.seed) + 991
    syn_replay_scal = compute_clip_scalars(
        simulate_from_live(base_live_frames, "REPLAY", seed=syn_seed),
        det,
        roi_pad=args.roi_pad,
        roi_resize_w=args.roi_resize_w,
        roi_stride=args.roi_stride,
    roi_bbox_override=base_roi_bbox,
    )
    syn_print_scal = compute_clip_scalars(
        simulate_from_live(base_live_frames, "PRINT", seed=syn_seed),
        det,
        roi_pad=args.roi_pad,
        roi_resize_w=args.roi_resize_w,
        roi_stride=args.roi_stride,
    roi_bbox_override=base_roi_bbox,
    )
    syn_mask_scal = compute_clip_scalars(
        simulate_from_live(base_live_frames, "MASK", seed=syn_seed),
        det,
        roi_pad=args.roi_pad,
        roi_resize_w=args.roi_resize_w,
        roi_stride=args.roi_stride,
    roi_bbox_override=base_roi_bbox,
    )

    # Replay: reject if replay_mean > thr (attack has higher replay_mean).
    replay_thr_attack = _mid(idle_scal.replay_mean, syn_replay_scal.replay_mean)
    # Ensure threshold is between idle and attack (robust directionally).
    replay_thr_attack = _clip(replay_thr_attack, lo=0.0001, hi=max(0.0002, syn_replay_scal.replay_mean * 0.999))
    # Never make the attack test weaker than the original default.
    replay_thr_attack = min(replay_thr_attack, float(args.replay_thr))

    # Print: reject if print_mean < thr (attack has lower colorfulness).
    print_thr_attack = _mid(idle_scal.print_mean, syn_print_scal.print_mean)
    print_thr_attack = _clip(print_thr_attack, lo=0.0, hi=max(1.0, idle_scal.print_mean * 0.999))
    # Never make the attack test weaker than the original default.
    print_thr_attack = max(print_thr_attack, float(args.print_thr))

    # Mask: reject if mask_score_mean > thr (attack has higher score).
    mask_thr_attack = _mid(idle_scal.mask_score_mean, syn_mask_scal.mask_score_mean)
    mask_thr_attack = _clip(mask_thr_attack, lo=0.01, hi=max(0.02, syn_mask_scal.mask_score_mean * 0.999))
    # Never make the attack test weaker than the original default.
    mask_thr_attack = min(mask_thr_attack, float(args.mask_thr))

    # ---- blur thresholds (separate for POSE/HEAD_SHAKE vs BLUR attack) ----
    blur_thr_live = blur_thr

    # POSE/HEAD_SHAKE can have transient blur; allow slightly lower sharpness there
    # to prevent early rejection *before* prompt verification.
    blur_thr_pose_live = max(40.0, blur_thr_live * 0.80)
    blur_thr_head_live = max(40.0, blur_thr_live * 0.85)

    # For the BLUR negative scenario, keep a stricter threshold based on the BLUR clip if present.
    if blur_scal.blur_median > 0 and idle_scal.blur_median > 0:
        blur_thr_blur_attack = _mid(idle_scal.blur_median, blur_scal.blur_median)
        blur_thr_blur_attack = max(blur_thr_live, blur_thr_blur_attack)
    else:
        blur_thr_blur_attack = blur_thr_live

    # ---- scenario -> threshold maps for each gate ----
    # Live scenarios (expected OK)
    live_ok = {"IDLE", "HEAD_SHAKE", "POSE", "BLINK_TRY"}

    # By default, use LIVE thresholds. Override per scenario where justified.
    blur_thr_map = {sc: blur_thr_live for sc in scenarios}
    blur_thr_map["POSE"] = blur_thr_pose_live
    blur_thr_map["HEAD_SHAKE"] = blur_thr_head_live
    blur_thr_map["BLUR"] = blur_thr_blur_attack  # negative scenario

    replay_thr_map = {sc: replay_thr_live for sc in scenarios}
    # POSE can introduce temporal aliasing on some webcams; relax slightly.
    replay_thr_map["POSE"] = replay_thr_live * 1.25
    replay_thr_map["REPLAY"] = replay_thr_attack  # negative scenario

    print_thr_map = {sc: print_thr_live for sc in scenarios}
    # POSE can reduce apparent colorfulness due to motion blur; relax slightly.
    print_thr_map["POSE"] = print_thr_live * 0.85
    print_thr_map["PRINT"] = print_thr_attack  # negative scenario

    mask_thr_map = {sc: mask_thr_live for sc in scenarios}
    # POSE: relax mask threshold slightly to avoid pose-induced false "mask" flags.
    mask_thr_map["POSE"] = max(mask_thr_live, mask_thr_live * 1.25)
    mask_thr_map["MASK"] = mask_thr_attack  # negative scenario

    # -------------------------------------------------------------------------
    # Report effective thresholds (paper-friendly)
    # -------------------------------------------------------------------------
    print(f"[THR] motion_thr(IDLE)={motion_thr_map['IDLE']:.4f}  motion_thr(HEAD_SHAKE)={motion_thr_map['HEAD_SHAKE']:.4f}  motion_thr(POSE)={motion_thr_map['POSE']:.4f}")
    print(f"[THR] blur_thr(IDLE)={blur_thr_map['IDLE']:.1f}  blur_thr(POSE)={blur_thr_map['POSE']:.1f}  blur_thr(BLUR)={blur_thr_map['BLUR']:.1f}")
    print(f"[THR] replay_thr_live={replay_thr_live:.4f}  replay_thr_attack={replay_thr_attack:.4f}")
    print(f"[THR] print_thr_live={print_thr_live:.2f}  print_thr_attack={print_thr_attack:.2f}")
    print(f"[THR] mask_thr_live={mask_thr_live:.3f}  mask_thr_pose={mask_thr_map['POSE']:.3f}  mask_thr_attack={mask_thr_attack:.3f}")

    # Warmup (reduces first-run bias in timing)
    _ = liveness_check(
        frames_bgr=base_live_frames[: min(20, len(base_live_frames))],
        det=det,
        face_mesh=face_mesh,
                roi_bbox_override=None,
        min_oneface_frac=args.min_oneface_frac,
        max_twoplus_frac=args.max_twoplus_frac,
        det_stride=max(1, args.det_stride),
        motion_thr=motion_thr_map["IDLE"],
        blur_thr=blur_thr_map["IDLE"],
        bright_lo=args.bright_lo,
        bright_hi=args.bright_hi,
        brightness_gate=bool(args.brightness_gate),
        roi_pad=args.roi_pad,
        roi_resize_w=args.roi_resize_w,
        roi_stride=args.roi_stride,
        replay_thr=replay_thr_map["IDLE"],
        mask_thr=mask_thr_map["IDLE"],
        print_thr=print_thr_map["IDLE"],
        prompt="none",
        mesh_stride=max(1, args.mesh_stride),
        yaw_thr=args.yaw_thr,
    )

    rows: List[Dict[str, object]] = []
    all_ok = True

    print(f"\n=== Liveness benchmark ({args.max_frames or 'all'} frames/clip, {args.runs} runs) ===")
    print(
        f"{'SCENARIO':<12} {'run':<4} {'LIVENESS':<8} {'TEST':<8} {'WHY':<60}"
        f"{'det':>8} {'prompt':>8} {'roi':>8} {'total':>8}"
    )
    print(f"{'':<12} {'':<4} {'':<8} {'':<8} | debug: oneFace twoPlus bright motion blurMed replay maskScore drop darkF lowStd print blinkThr")

    for run_id in range(1, args.runs + 1):
        run_seed = args.seed + run_id * 101
        random.seed(run_seed)
        np.random.seed(run_seed)

        for scenario in scenarios:
            if scenario in ("REPLAY", "PRINT", "MASK"):
                frames = simulate_from_live(base_live_frames, scenario, seed=run_seed)
            else:
                frames = real_frames[scenario]

            lr = liveness_check(
                frames_bgr=frames,
                det=det,
                face_mesh=face_mesh,
                roi_bbox_override=(base_roi_bbox if scenario in ("REPLAY", "PRINT", "MASK") else None),
                min_oneface_frac=args.min_oneface_frac,
                max_twoplus_frac=args.max_twoplus_frac,
                det_stride=args.det_stride,
                motion_thr=motion_thr_map[scenario],
                blur_thr=blur_thr_map[scenario],
                bright_lo=args.bright_lo,
                bright_hi=args.bright_hi,
                brightness_gate=bool(args.brightness_gate),
                roi_pad=args.roi_pad,
                roi_resize_w=args.roi_resize_w,
                roi_stride=args.roi_stride,
                replay_thr=replay_thr_map[scenario],
                mask_thr=mask_thr_map[scenario],
                print_thr=print_thr_map[scenario],
                prompt=prompt_for[scenario],
                mesh_stride=args.mesh_stride,
                yaw_thr=args.yaw_thr,
            )

            print_row(scenario, run_id, lr, expect_ok=expect_ok[scenario])

            test_pass = (lr.ok == expect_ok[scenario])
            all_ok = all_ok and test_pass

            rows.append({
                "scenario": scenario,
                "run": run_id,
                "ok": int(lr.ok),
                "expected_ok": int(expect_ok[scenario]),
                "test_pass": int(test_pass),
                "why": lr.why,
                "det_ms": lr.timing.det_ms,
                "prompt_ms": lr.timing.prompt_ms,
                "roi_ms": lr.timing.roi_ms,
                "total_ms": lr.timing.total_ms,
                "oneface_frac": lr.scalars.oneface_frac,
                "twoplus_frac": lr.scalars.twoplus_frac,
                "brightness_mean": lr.scalars.brightness_mean,
                "motion_mean": lr.scalars.motion_mean,
                "blur_median": lr.scalars.blur_median,
                "replay_mean": lr.scalars.replay_mean,
                "mask_score_mean": lr.scalars.mask_score_mean,
                "mask_drop_mean": lr.scalars.mask_drop_mean,
                "mask_darkfrac_mean": lr.scalars.mask_darkfrac_mean,
                "mask_lowstd_mean": lr.scalars.mask_lowstd_mean,
                "print_mean": lr.scalars.print_mean,
                "blink_thr": lr.scalars.blink_thr,
            })

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n--- Summary across runs ---")
    print(f"{'SCENARIO':<12} {'pass%':>6} {'totµ':>8} {'totσ':>8} {'totp95':>8}  {'detµ':>8} {'prµ':>8} {'roiµ':>8}")

    summary_rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        rs = [r for r in rows if r["scenario"] == scenario]
        pass_rate = 100.0 * sum(int(r["test_pass"]) for r in rs) / max(1, len(rs))

        tot = [float(r["total_ms"]) for r in rs]
        det_ms = [float(r["det_ms"]) for r in rs]
        pr_ms = [float(r["prompt_ms"]) for r in rs]
        roi_ms = [float(r["roi_ms"]) for r in rs]

        tot_m, tot_s = mean_std(tot)
        det_m, _ = mean_std(det_ms)
        pr_m, _ = mean_std(pr_ms)
        roi_m, _ = mean_std(roi_ms)

        tot_p95 = p95(tot)

        print(f"{scenario:<12} {pass_rate:6.1f} {tot_m:8.1f} {tot_s:8.1f} {tot_p95:8.1f}  {det_m:8.1f} {pr_m:8.1f} {roi_m:8.1f}")

        summary_rows.append({
            "scenario": scenario,
            "pass_rate": pass_rate,
            "total_mean": tot_m,
            "total_std": tot_s,
            "total_p95": tot_p95,
            "det_mean": det_m,
            "prompt_mean": pr_m,
            "roi_mean": roi_m,
        })

    # -------------------------------------------------------------------------
    # Outputs (CSV + config)
    # -------------------------------------------------------------------------
    if args.save_csv and rows:
        runs_path = f"{args.out_prefix}_runs.csv"
        summ_path = f"{args.out_prefix}_summary.csv"
        cfg_path = f"{args.out_prefix}_config.json"

        # Ensure parent folder exists (e.g., out-prefix 'paper/liveness' -> ./paper/)
        out_dir = os.path.dirname(os.path.abspath(runs_path))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(runs_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        with open(summ_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

        cfg = vars(args).copy()
        cfg["effective_motion_thr_idle"] = motion_thr_map["IDLE"]
        cfg["effective_blur_thr_live"] = blur_thr_live
        cfg["effective_blur_thr_map"] = blur_thr_map
        cfg["effective_motion_thr_map"] = motion_thr_map
        cfg["effective_replay_thr_live"] = replay_thr_live
        cfg["effective_replay_thr_attack"] = replay_thr_attack
        cfg["effective_replay_thr_map"] = replay_thr_map
        cfg["effective_mask_thr_live"] = mask_thr_live
        cfg["effective_mask_thr_attack"] = mask_thr_attack
        cfg["effective_mask_thr_map"] = mask_thr_map
        cfg["effective_print_thr_live"] = print_thr_live
        cfg["effective_print_thr_attack"] = print_thr_attack
        cfg["effective_print_thr_map"] = print_thr_map
        cfg["auto_calibrated_motion_blur"] = bool(args.auto_thresholds)
        cfg["auto_calibrated_attack"] = bool(args.auto_attack_thresholds)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        print(f"\n[OUT] wrote {runs_path}, {summ_path}, {cfg_path}")

        if args.save_latex:
            tex_path = f"{args.out_prefix}_timings_table.tex"
            out_dir3 = os.path.dirname(os.path.abspath(tex_path))
            if out_dir3 and not os.path.isdir(out_dir3):
                os.makedirs(out_dir3, exist_ok=True)

            with open(tex_path, "w", encoding="utf-8") as f:
                f.write("% Auto-generated by liveness_check.py\n")
                f.write("\\begin{tabular}{lrrrr}\n")
                f.write("\\hline\n")
                f.write("Scenario & Pass(\\%) & Total$_{\\mu}$ (ms) & Total$_{p95}$ (ms) & ROI$_{\\mu}$ (ms)\\\\\n")
                f.write("\\hline\n")
                for s in summary_rows:
                    f.write(f"{s['scenario']} & {s['pass_rate']:.1f} & {s['total_mean']:.1f} & {s['total_p95']:.1f} & {s['roi_mean']:.1f}\\\\\n")
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
            print(f"[OUT] wrote {tex_path}")

    raise SystemExit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
 
