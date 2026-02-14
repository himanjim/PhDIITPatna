# record_liveness_clips.py
"""
Webcam-based liveness clip recorder with per-scenario validation + RESUME.

Key features
------------
1) Records multiple scenarios (IDLE, HEAD_SHAKE, POSE, BLINK_TRY, NO_MOTION, MULTIFACE).
2) Validates each clip with cheap metrics (Haar face count, brightness, blur, frame-diff motion).
3) SAVES EACH SCENARIO IMMEDIATELY (atomic save) so progress is never lost.
4) RESUME BY DEFAULT:
   - If <out_dir>/<SCENARIO>.mp4 exists, it is treated as complete and skipped.
   - A progress.json file records what was saved, with metrics + reasons.
5) Optional BLUR:
   - You can record BLUR manually, or auto-generate BLUR from another clip (default: IDLE).

Notes
-----
- Face detection is Haar-cascade only (fast sanity gate). It is not a production face detector.
- Validation thresholds are only to ensure you recorded “distinct enough” scenarios.
  Tune thresholds to your lighting/camera setup if needed.
"""

import os
import time
import json
import argparse
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np


# =============================
# Scenarios (live capture)
# =============================
SCENARIOS_DEFAULT = [
    ("IDLE",       "Look at camera normally. Natural tiny micro-motions are OK."),
    ("HEAD_SHAKE", "Shake head left-right naturally (small/medium amplitude)."),
    ("POSE",       "Turn head left then right (clear yaw change)."),
    ("BLINK_TRY",  "Blink naturally 2–3 times during the clip."),
    ("NO_MOTION",  "Try to stay as still as possible (minimize head/eye motion)."),
    ("MULTIFACE",  "Bring a second face into frame for part of the clip."),
    # BLUR is handled separately (manual or auto-generated).
]


@dataclass
class ValidateResult:
    ok: bool
    reasons: List[str]
    metrics: Dict[str, float]


# =============================
# Small utilities
# =============================
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def atomic_write_json(path: str, obj: dict):
    """
    Write JSON atomically:
      - write to temp file in same folder
      - os.replace(temp, final) (atomic on Windows NTFS for same drive)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".__tmp__.", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def load_progress(progress_path: str) -> dict:
    if not os.path.exists(progress_path):
        return {"version": 1, "updated_at": None, "scenarios": {}}
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "scenarios" not in data:
            data["scenarios"] = {}
        return data
    except Exception:
        # If progress.json is corrupted, start fresh but do NOT delete videos on disk.
        return {"version": 1, "updated_at": None, "scenarios": {}}


def update_progress(progress_path: str, progress: dict, scenario: str, out_path: str,
                    vr: ValidateResult):
    """
    Record scenario completion to progress.json.
    We store metrics + ok flag + reasons so you can inspect later.
    """
    progress["updated_at"] = now_iso()
    progress["scenarios"][scenario] = {
        "path": os.path.abspath(out_path),
        "saved_at": now_iso(),
        "ok": bool(vr.ok),
        "reasons": list(vr.reasons),
        "metrics": dict(vr.metrics),
    }
    atomic_write_json(progress_path, progress)


# =============================
# UI helpers
# =============================
def put_multiline_text(img, lines, x=10, y=25, dy=24,
                       color=(0, 255, 0), scale=0.6, thickness=2):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)


# =============================
# Metrics
# =============================
def laplacian_var(gray: np.ndarray) -> float:
    """Blur metric: variance of Laplacian. Higher => sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_mean(gray: np.ndarray) -> float:
    return float(gray.mean())


# =============================
# Face detection (Haar) - cheap sanity gate
# =============================
def load_haar() -> cv2.CascadeClassifier:
    xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(xml)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {xml}")
    return cascade


def detect_faces_haar(cascade: cv2.CascadeClassifier, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of face bboxes (x,y,w,h). Haar is approximate; treat as a sanity gate only.
    """
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def largest_face_bbox(faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not faces:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def crop_roi(gray: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], pad: float = 0.35) -> np.ndarray:
    """
    Crop around face bbox with padding. If bbox missing, return full frame.
    Cropping ROI makes motion/blur metrics less sensitive to background noise.
    """
    if bbox is None:
        return gray

    h, w = gray.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    pw, ph = bw * (1.0 + pad), bh * (1.0 + pad)

    x1 = int(max(0, cx - pw / 2.0))
    y1 = int(max(0, cy - ph / 2.0))
    x2 = int(min(w, cx + pw / 2.0))
    y2 = int(min(h, cy + ph / 2.0))

    if x2 <= x1 or y2 <= y1:
        return gray
    return gray[y1:y2, x1:x2]


# =============================
# Video IO (atomic save)
# =============================
def save_mp4_atomic(out_path: str, frames: List[np.ndarray], fps: float):
    """
    Atomic MP4 save:
      - write to a temp path in the same folder
      - os.replace(temp, out_path)
    This prevents losing an entire clip if the process is interrupted mid-write.
    """
    if not frames:
        raise RuntimeError("No frames to save")

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    tmp_path = os.path.join(out_dir, f"{os.path.basename(out_path)}.__tmp__.mp4")
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    vw = cv2.VideoWriter(tmp_path, fourcc, float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {tmp_path}")

    for fr in frames:
        vw.write(fr)
    vw.release()

    # Atomic replace
    os.replace(tmp_path, out_path)


def load_mp4(path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames, float(fps) if fps and fps > 0 else 0.0


# =============================
# BLUR generation (auto)
# =============================
def _motion_blur_kernel(ksize: int, angle_deg: float) -> np.ndarray:
    """Build a normalized motion-blur kernel (line) rotated by angle."""
    k = np.zeros((ksize, ksize), dtype=np.float32)
    k[ksize // 2, :] = 1.0
    k /= max(1e-6, k.sum())

    M = cv2.getRotationMatrix2D((ksize / 2.0, ksize / 2.0), angle_deg, 1.0)
    k_rot = cv2.warpAffine(k, M, (ksize, ksize))
    s = float(k_rot.sum())
    if s > 1e-6:
        k_rot /= s
    return k_rot


def generate_blur_clip(frames: List[np.ndarray],
                       gaussian_ksize: int = 11,
                       motion_ksize: int = 17,
                       downscale: float = 0.0) -> List[np.ndarray]:
    """
    Generate a BLUR clip from a real clip by applying:
      - Gaussian blur
      - Motion blur (random angle)
      - Optional downscale-upscale softness
    """
    if not frames:
        return []

    # Ensure odd kernel sizes
    if gaussian_ksize % 2 == 0:
        gaussian_ksize += 1
    if motion_ksize % 2 == 0:
        motion_ksize += 1

    angle = float(np.random.uniform(0, 180))
    k_motion = _motion_blur_kernel(motion_ksize, angle_deg=angle)

    out = []
    for fr in frames:
        x = fr.copy()

        if downscale and 0.2 <= downscale < 1.0:
            h, w = x.shape[:2]
            dw = max(16, int(w * downscale))
            dh = max(16, int(h * downscale))
            x = cv2.resize(x, (dw, dh), interpolation=cv2.INTER_AREA)
            x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)

        x = cv2.GaussianBlur(x, (gaussian_ksize, gaussian_ksize), 0)
        x = cv2.filter2D(x, -1, k_motion)
        out.append(x)
    return out


# =============================
# Validation
# =============================
def validate_clip(
    frames_bgr: List[np.ndarray],
    fps_target: float,
    seconds_target: float,
    scenario: str,
    cascade: cv2.CascadeClassifier,
    # thresholds
    min_one_face_frac: float,
    min_two_face_frac: float,
    bright_lo: float,
    bright_hi: float,
    blur_min_sharp: float,
    blur_max_blur: float,
    motion_idle_min: float,
    motion_nomotion_max: float,
    motion_pose_min: float,
    motion_headshake_min: float,
) -> ValidateResult:
    """
    Validation is a simple gating layer to ensure clips are "distinct enough".
    It does NOT guarantee correctness of any real liveness model.
    """
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    n = len(frames_bgr)
    expected = int(round(fps_target * seconds_target))
    tol = int(max(3, 0.10 * expected))

    metrics["frames"] = float(n)
    metrics["expected_frames"] = float(expected)

    if not (expected - tol <= n <= expected + tol):
        reasons.append(f"duration mismatch: frames={n}, expected≈{expected}±{tol}")

    br = []
    bl = []
    motion = []
    face_counts = []

    # Motion is computed on a fixed-size ROI to avoid cv2.absdiff size mismatch
    prev_norm = None
    MOTION_NORM_SIZE = (160, 160)  # (W,H)

    for fr in frames_bgr:
        gray_full = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

        faces = detect_faces_haar(cascade, gray_full)
        face_counts.append(len(faces))
        bbox = largest_face_bbox(faces)

        roi = crop_roi(gray_full, bbox, pad=0.35)
        br.append(brightness_mean(roi))
        bl.append(laplacian_var(roi))

        # Normalize ROI to fixed size for stable frame-diff motion scoring
        norm = cv2.resize(roi, MOTION_NORM_SIZE, interpolation=cv2.INTER_AREA)

        if prev_norm is not None:
            d = cv2.absdiff(prev_norm, norm)
            motion.append(float(d.mean()) / 255.0)

        prev_norm = norm

    br_mean = float(np.mean(br)) if br else 0.0
    blur_med = float(np.median(bl)) if bl else 0.0
    motion_mean = float(np.mean(motion)) if motion else 0.0
    one_face_frac = float(np.mean(np.array(face_counts) == 1)) if face_counts else 0.0
    two_plus_frac = float(np.mean(np.array(face_counts) >= 2)) if face_counts else 0.0

    metrics.update({
        "brightness_mean": br_mean,
        "blur_median": blur_med,
        "motion_mean": motion_mean,
        "one_face_frac": one_face_frac,
        "two_plus_face_frac": two_plus_frac,
    })

    # Brightness gates
    if br_mean < bright_lo:
        reasons.append(f"too dark (brightness_mean={br_mean:.1f} < {bright_lo})")
    if br_mean > bright_hi:
        reasons.append(f"too bright/washed (brightness_mean={br_mean:.1f} > {bright_hi})")

    # Face gates
    if scenario != "MULTIFACE":
        if one_face_frac < min_one_face_frac:
            reasons.append(f"not enough 1-face frames (one_face_frac={one_face_frac:.2f} < {min_one_face_frac})")
    else:
        if two_plus_frac < min_two_face_frac:
            reasons.append(f"not enough 2-face frames (two_plus_face_frac={two_plus_frac:.2f} < {min_two_face_frac})")

    # Blur gates
    if scenario == "BLUR":
        # For BLUR we want low Laplacian variance (blurry)
        if blur_med > blur_max_blur:
            reasons.append(f"BLUR not blurry enough (blur_median={blur_med:.1f} > {blur_max_blur})")
    else:
        # For non-BLUR, require sharp enough
        if blur_med < blur_min_sharp:
            reasons.append(f"too blurry (blur_median={blur_med:.1f} < {blur_min_sharp})")

    # Motion gates
    if scenario == "NO_MOTION":
        if motion_mean > motion_nomotion_max:
            reasons.append(f"too much motion for NO_MOTION (motion_mean={motion_mean:.3f} > {motion_nomotion_max})")
    elif scenario in ("IDLE", "BLINK_TRY"):
        if motion_mean < motion_idle_min:
            reasons.append(f"too static for {scenario} (motion_mean={motion_mean:.3f} < {motion_idle_min})")
    elif scenario == "POSE":
        if motion_mean < motion_pose_min:
            reasons.append(f"POSE motion too small (motion_mean={motion_mean:.3f} < {motion_pose_min})")
    elif scenario == "HEAD_SHAKE":
        if motion_mean < motion_headshake_min:
            reasons.append(f"HEAD_SHAKE motion too small (motion_mean={motion_mean:.3f} < {motion_headshake_min})")

    ok = (len(reasons) == 0)
    return ValidateResult(ok=ok, reasons=reasons, metrics=metrics)


# =============================
# Camera control
# =============================
def open_camera(cam_index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    """
    DirectShow first (Windows), fallback otherwise.
    CAP_PROP_FPS is often ignored; we record fixed frame counts anyway.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {cam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def record_clip(cap: cv2.VideoCapture,
                scenario: str,
                instruction: str,
                seconds: float,
                fps: float,
                preview_window: str) -> List[np.ndarray]:
    """
    Preview until SPACE, then record fixed number of frames.
    """
    frames: List[np.ndarray] = []
    frames_target = int(round(seconds * fps))
    frame_period = 1.0 / float(fps)

    # Preview loop
    while True:
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")

        overlay = fr.copy()
        put_multiline_text(
            overlay,
            [
                f"SCENARIO: {scenario}",
                instruction,
                "",
                "SPACE = start recording",
                "ESC   = quit",
            ],
            color=(0, 255, 0),
        )
        cv2.imshow(preview_window, overlay)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            raise KeyboardInterrupt
        if k == 32:  # SPACE
            break

    # Record loop
    next_t = time.perf_counter()
    for i in range(frames_target):
        ok, fr = cap.read()
        if not ok:
            break

        frames.append(fr)

        overlay = fr.copy()
        put_multiline_text(
            overlay,
            [
                f"RECORDING {scenario}   {i+1}/{frames_target}",
                "Follow the instruction.",
            ],
            color=(0, 0, 255),
        )
        cv2.imshow(preview_window, overlay)
        cv2.waitKey(1)

        next_t += frame_period
        while time.perf_counter() < next_t:
            cv2.waitKey(1)

    return frames


def review_clip(frames: List[np.ndarray], fps: float, title: str) -> bool:
    """
    Playback once. Press Y to accept, N to reject.
    If no key pressed, asks in console.
    """
    if not frames:
        return False

    delay_ms = int(max(1, round(1000.0 / fps)))
    win = "review_clip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for fr in frames:
        overlay = fr.copy()
        put_multiline_text(
            overlay,
            [
                f"REVIEW: {title}",
                "Press Y accept / N reject (during playback).",
            ],
            color=(255, 255, 0),
        )
        cv2.imshow(win, overlay)
        k = cv2.waitKey(delay_ms) & 0xFF
        if k in (ord("y"), ord("Y")):
            cv2.destroyWindow(win)
            return True
        if k in (ord("n"), ord("N")):
            cv2.destroyWindow(win)
            return False

    cv2.destroyWindow(win)
    ans = input("Accept this clip? (y/n): ").strip().lower()
    return ans == "y"


# =============================
# Main workflow
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="clips", help="Output folder")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index (0/1/2...)")
    ap.add_argument("--seconds", type=float, default=3.0, help="Seconds per clip")
    ap.add_argument("--fps", type=float, default=20.0, help="Target FPS")
    ap.add_argument("--width", type=int, default=640, help="Capture width")
    ap.add_argument("--height", type=int, default=480, help="Capture height")

    # Resume/overwrite controls
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Resume from already-saved mp4 files in out-dir (default: ON).")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Disable resume; always record everything.")
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, re-record and overwrite existing scenario mp4s.")

    # Validation knobs
    ap.add_argument("--min-one-face-frac", type=float, default=0.50,
                    help="Min fraction of frames with exactly 1 face (non-MULTIFACE).")
    ap.add_argument("--min-two-face-frac", type=float, default=0.25,
                    help="For MULTIFACE: min fraction with >=2 faces.")
    ap.add_argument("--bright-lo", type=float, default=35.0, help="Min mean brightness (ROI).")
    ap.add_argument("--bright-hi", type=float, default=220.0, help="Max mean brightness (ROI).")

    # Blur thresholds (ROI Laplacian var)
    ap.add_argument("--blur-min-sharp", type=float, default=80.0,
                    help="Non-BLUR: min median Laplacian var on ROI.")
    ap.add_argument("--blur-max-blur", type=float, default=60.0,
                    help="BLUR: max median Laplacian var on ROI.")

    # Motion thresholds (ROI frame-diff mean)
    ap.add_argument("--motion-idle-min", type=float, default=0.012,
                    help="Min motion for IDLE/BLINK_TRY.")
    ap.add_argument("--motion-nomotion-max", type=float, default=0.05,
                    help="Max motion for NO_MOTION.")
    ap.add_argument("--motion-pose-min", type=float, default=0.04,
                    help="Min motion for POSE.")
    ap.add_argument("--motion-headshake-min", type=float, default=0.022,
                    help="Min motion for HEAD_SHAKE.")

    # Control
    ap.add_argument("--force-accept", action="store_true",
                    help="If set, saves even if validation fails (still shows review).")

    # BLUR handling
    ap.add_argument("--auto-generate-blur", action="store_true",
                    help="Generate BLUR.mp4 automatically from an accepted source clip (default IDLE).")
    ap.add_argument("--blur-from", default="IDLE",
                    help="Scenario name to use as source for BLUR generation (default IDLE).")
    ap.add_argument("--blur-gauss", type=int, default=11,
                    help="Gaussian kernel size for blur generation (odd int recommended).")
    ap.add_argument("--blur-motion", type=int, default=17,
                    help="Motion blur kernel size for blur generation (odd int recommended).")
    ap.add_argument("--blur-downscale", type=float, default=0.0,
                    help="Optional downscale factor (0.3–0.9) before upscaling, adds softness.")

    # Output metrics
    ap.add_argument("--write-metrics-json", action="store_true",
                    help="Write per-scenario metrics JSON beside the mp4 files.")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    progress_path = os.path.join(out_dir, "progress.json")
    progress = load_progress(progress_path)

    print(f"[OUT] Folder: {out_dir}")
    print(f"[OUT] Progress: {progress_path}")
    print("[UI] Preview controls: SPACE=start, ESC=quit. Review: Y accept / N reject.\n")

    cascade = load_haar()
    cap = open_camera(args.cam, args.width, args.height, args.fps)

    win = "record_liveness_clips"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def scenario_path(s: str) -> str:
        return os.path.join(out_dir, f"{s}.mp4")

    def print_metrics(vr: ValidateResult):
        m = vr.metrics
        print("Metrics:",
              f"frames={int(m['frames'])}/{int(m['expected_frames'])}",
              f"bright={m['brightness_mean']:.1f}",
              f"blurMed={m['blur_median']:.1f}",
              f"motion={m['motion_mean']:.3f}",
              f"oneFaceFrac={m['one_face_frac']:.2f}",
              f"twoPlusFrac={m['two_plus_face_frac']:.2f}",
              sep=" | ")
        if not vr.ok:
            print("Validation: FAIL")
            for r in vr.reasons:
                print(" -", r)
        else:
            print("Validation: PASS")

    def validate_frames(scenario: str, frames: List[np.ndarray]) -> ValidateResult:
        return validate_clip(
            frames_bgr=frames,
            fps_target=args.fps,
            seconds_target=args.seconds,
            scenario=scenario,
            cascade=cascade,
            min_one_face_frac=args.min_one_face_frac,
            min_two_face_frac=args.min_two_face_frac,
            bright_lo=args.bright_lo,
            bright_hi=args.bright_hi,
            blur_min_sharp=args.blur_min_sharp,
            blur_max_blur=args.blur_max_blur,
            motion_idle_min=args.motion_idle_min,
            motion_nomotion_max=args.motion_nomotion_max,
            motion_pose_min=args.motion_pose_min,
            motion_headshake_min=args.motion_headshake_min,
        )

    def save_scenario(scenario: str, frames: List[np.ndarray], vr: ValidateResult):
        out_path = scenario_path(scenario)
        save_mp4_atomic(out_path, frames, fps=args.fps)
        update_progress(progress_path, progress, scenario, out_path, vr)
        print(f"[SAVED] {out_path}")

        if args.write_metrics_json:
            jpath = os.path.join(out_dir, f"{scenario}.metrics.json")
            atomic_write_json(jpath, {"ok": vr.ok, "reasons": vr.reasons, "metrics": vr.metrics})
            print(f"[SAVED] {jpath}")

    def already_done(scenario: str) -> bool:
        """
        Resume logic:
          - If mp4 exists AND not overwrite, treat as done.
          - progress.json is advisory (used for metrics/history), but mp4 presence is the primary signal.
        """
        out_path = scenario_path(scenario)
        if args.overwrite:
            return False
        if args.resume and os.path.exists(out_path):
            return True
        return False

    try:
        # 1) Record all live capture scenarios, but SKIP already-saved ones when resuming
        for scenario, instruction in SCENARIOS_DEFAULT:
            if already_done(scenario):
                print(f"[RESUME] Skipping {scenario} (already exists): {scenario_path(scenario)}")
                continue

            while True:
                print(f"\n=== Scenario: {scenario} ===")
                print(f"Instruction: {instruction}")

                frames = record_clip(
                    cap=cap,
                    scenario=scenario,
                    instruction=instruction,
                    seconds=args.seconds,
                    fps=args.fps,
                    preview_window=win,
                )

                vr = validate_frames(scenario, frames)
                print_metrics(vr)

                # Manual review always
                if not review_clip(frames, fps=args.fps, title=scenario):
                    print("Re-recording this scenario...")
                    continue

                # Save if valid OR force-accept
                if vr.ok or args.force_accept:
                    save_scenario(scenario, frames, vr)
                    break

                print("[NOT SAVED] You accepted, but validation failed and --force-accept is not set.")
                print("Re-recording this scenario...")

        # 2) Handle BLUR (manual or auto-generated)
        if already_done("BLUR"):
            print(f"[RESUME] Skipping BLUR (already exists): {scenario_path('BLUR')}")
        else:
            if args.auto_generate_blur:
                src = args.blur_from.upper().strip()
                src_path = scenario_path(src)
                if not os.path.exists(src_path):
                    raise RuntimeError(f"BLUR generation requested, but source clip missing: {src_path}")

                print(f"\n=== Scenario: BLUR (auto-generated from {src}) ===")
                src_frames, _fps = load_mp4(src_path)

                gen = generate_blur_clip(
                    src_frames,
                    gaussian_ksize=args.blur_gauss,
                    motion_ksize=args.blur_motion,
                    downscale=args.blur_downscale,
                )

                # Validate and save BLUR; if too sharp, strengthen once and retry.
                for attempt in range(2):
                    vr = validate_frames("BLUR", gen)
                    print_metrics(vr)

                    # Review is useful even for generated clips (you can visually confirm it's blurred)
                    if not review_clip(gen, fps=args.fps, title="BLUR (generated)"):
                        print("Regenerating BLUR with stronger blur...")
                        # strengthen and regenerate
                        args.blur_gauss = max(args.blur_gauss, 11) + 4
                        args.blur_motion = max(args.blur_motion, 17) + 6
                        gen = generate_blur_clip(
                            src_frames,
                            gaussian_ksize=args.blur_gauss,
                            motion_ksize=args.blur_motion,
                            downscale=args.blur_downscale,
                        )
                        continue

                    if vr.ok or args.force_accept:
                        save_scenario("BLUR", gen, vr)
                        break

                    # If not ok, strengthen blur and retry once
                    print("[INFO] BLUR validation failed; strengthening blur and retrying...")
                    args.blur_gauss = max(args.blur_gauss, 11) + 4
                    args.blur_motion = max(args.blur_motion, 17) + 6
                    gen = generate_blur_clip(
                        src_frames,
                        gaussian_ksize=args.blur_gauss,
                        motion_ksize=args.blur_motion,
                        downscale=args.blur_downscale,
                    )

                if not os.path.exists(scenario_path("BLUR")):
                    print("[WARN] BLUR not saved. Re-run with stronger --blur-gauss/--blur-motion or use --force-accept.")
            else:
                # Manual BLUR recording
                while True:
                    print("\n=== Scenario: BLUR (manual) ===")
                    print("Instruction: Create blur (move quickly / slightly defocus).")

                    frames = record_clip(
                        cap=cap,
                        scenario="BLUR",
                        instruction="Create blur (move quickly / slightly defocus).",
                        seconds=args.seconds,
                        fps=args.fps,
                        preview_window=win,
                    )

                    vr = validate_frames("BLUR", frames)
                    print_metrics(vr)

                    if not review_clip(frames, fps=args.fps, title="BLUR"):
                        print("Re-recording BLUR...")
                        continue

                    if vr.ok or args.force_accept:
                        save_scenario("BLUR", frames, vr)
                        break

                    print("[NOT SAVED] You accepted, but validation failed and --force-accept is not set.")
                    print("Re-recording BLUR...")

        print("\n[DONE] All requested clips produced.")
        print(f"[DONE] Clips folder: {out_dir}")
        print(f"[DONE] Progress file: {progress_path}")

    except KeyboardInterrupt:
        print("\n[EXIT] User aborted.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()