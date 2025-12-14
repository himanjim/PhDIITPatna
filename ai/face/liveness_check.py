# live_benchmark.py
"""
End-to-end liveness benchmark from two still photos.

What it does
------------
• Generates short "video-like" sequences from 2 input faces (A,B).
• Runs liveness_check() with:
    - Anti-coercion (enforce ~one face),
    - Passive anti-spoof (ONNX; optional),
    - Active prompt (blink / pose / none),
    - Micro-motion (optical flow) & blur checks.
• Benchmarks each stage and aggregates stats across multiple runs.

New in this version
-------------------
• FIX: Optical flow uses a fixed downscaled size every step (no shape mismatch).
• --min-oneface-frac gate so tests like REPLAY/PRINT don't fail too early.
• --det-size to try larger detector input (e.g., 800x800) on hard frames.
• --replay-check heuristic (scanline periodicity) to flag “replay-like” inputs.
• Multi-run benchmarking (--runs N) with mean/std/p95 and optional CSV export.

NOTE: The exported ONNX anti-spoof (when --export-antispoof) uses a tiny
stub model with deterministic bias (p_live≈0.88) for plumbing/timing only.
Replace with a real model to evaluate spoof robustness.
"""
import os, time, argparse, random, warnings, csv, math
from statistics import mean, pstdev
import numpy as np
import cv2, onnxruntime as ort, mediapipe as mp
from insightface.app import FaceAnalysis
from insightface.utils import face_align

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

# -------------------- GPU/Provider detection --------------------
ORT_PROVIDERS = ort.get_available_providers()
USE_CUDA = "CUDAExecutionProvider" in ORT_PROVIDERS
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_CUDA else ["CPUExecutionProvider"]
CTX_ID = 0 if USE_CUDA else -1

# -------------------- Optional: export a tiny anti-spoof model to ONNX --------------------
def maybe_export_antispoof_onnx(out_path="antispoof_cdcn.onnx", device_pref="cuda"):
    """
    Export a tiny CDCN-inspired CNN (random convs) to ONNX so you can test end-to-end.
    Deterministic head: p_spoof = sigmoid(-2) ≈ 0.119 → p_live ≈ 0.881 (passes at 0.5 threshold).
    """
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if (device_pref=="cuda" and torch.cuda.is_available()) else "cpu")

    class DepthwiseCD(nn.Module):
        def __init__(self, c, k=3):
            super().__init__()
            self.dw = nn.Conv2d(c, c, k, padding=k//2, groups=c, bias=False)
            self.pw = nn.Conv2d(c, c, 1, bias=False)
        def forward(self, x):
            y = self.dw(x); mean = torch.mean(x, dim=(2,3), keepdim=True)
            y = y - mean
            return self.pw(nn.ReLU(inplace=True)(y))

    class TinyCDCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
            self.block1 = nn.Sequential(DepthwiseCD(16), nn.ReLU(inplace=True), nn.MaxPool2d(2))
            self.block2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
                                        DepthwiseCD(32), nn.ReLU(inplace=True), nn.MaxPool2d(2))
            self.block3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
                                        DepthwiseCD(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 1), nn.Sigmoid())  # p_spoof
        def forward(self, x):
            x = self.stem(x); x = self.block1(x); x = self.block2(x); x = self.block3(x)
            return self.head(x)

    model = TinyCDCN().to(device).eval()
    with torch.no_grad():
        lin = model.head[1]  # Linear(64,1)
        lin.weight.zero_()
        lin.bias.fill_(-2.0)  # sigmoid(-2)=0.119 → p_live≈0.881

    dummy = torch.randn(1, 3, 112, 112, device=device)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["p_spoof"],
        dynamic_axes={"input": {0: "batch"}, "p_spoof": {0: "batch"}},
        opset_version=12
    )
    return out_path

# -------------------- Core helpers --------------------
def align112(bgr, face):
    """InsightFace 5-pt alignment → 112×112 crop expected by ArcFace-style backbones."""
    return face_align.norm_crop(bgr, landmark=face.kps, image_size=112)

def lap_var(gray):
    """Focus metric: variance of Laplacian (lower → blurrier)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def ear(lm):
    """Eye-Aspect-Ratio using MediaPipe landmark indices (approx). Lower → blink."""
    L = [33,160,158,133,153,144]; p=lambda i: np.array([lm[i].x, lm[i].y])
    A = np.linalg.norm(p(160)-p(144)) + np.linalg.norm(p(158)-p(153))
    B = np.linalg.norm(p(33)-p(133))
    return A / (2.0*B + 1e-6)

def head_pose(lm, w, h):
    """Rough yaw/pitch via PnP on a sparse 3D face template."""
    idx=[33,263,1,61,291,199]
    img=np.array([[lm[i].x*w, lm[i].y*h] for i in idx], np.float32)
    mdl=np.array([[-30,30,30],[30,30,30],[0,0,60],[-25,-30,20],[25,-30,20],[0,-65,0]], np.float32)
    K=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], np.float32)
    ok,rvec,tvec=cv2.solvePnP(mdl,img,K,np.zeros(4),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0,0.0
    R,_=cv2.Rodrigues(rvec); sy=np.sqrt(R[0,0]**2+R[1,0]**2)
    yaw=np.degrees(np.arctan2(R[2,0], sy)); pitch=np.degrees(np.arctan2(-R[2,1], R[2,2]))
    return yaw,pitch

def run_antispoof(sess, bgr112):
    """Run ONNX anti-spoof; return (p_live, elapsed_ms). If sess is None → pass-through."""
    if sess is None: return 1.0, 0.0
    t0=time.perf_counter()
    x=bgr112.astype(np.float32); x=(x-127.5)/128.0
    x=np.transpose(x,(2,0,1))[None,...]  # 1×3×112×112
    p_spoof=float(sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})[0][0][0])
    dt=(time.perf_counter()-t0)*1000
    return 1.0 - p_spoof, dt

# -------------------- Replay heuristic --------------------
def scanline_score(gray):
    """
    Simple replay cue: horizontal scanline periodicity.
    We compute an FFT per row (after mean removal) and average magnitudes in a mid/high band.
    Higher average ⇒ more stripe-like content typical of screen replays.
    """
    s = gray.astype(np.float32)
    mag = np.abs(np.fft.rfft(s - s.mean(axis=1, keepdims=True), axis=1))
    band = mag[:, mag.shape[1]//6 : mag.shape[1]//2]  # drop DC/very low freq
    return float(band.mean())

# -------------------- Liveness check --------------------
def liveness_check(frames_bgr, faces_app, mesh, spoof_sess,
                   prompt="blink", live_thr=0.5,
                   motion_thr=0.4, blur_thr=60.0, flow_scale=1.0,
                   min_oneface_frac=0.6, replay_check=False, replay_thr=2.5):
    """
    frames_bgr     : list of BGR frames
    prompt         : 'blink' | 'smile' | 'pose' | 'none'
    live_thr       : anti-spoof live probability threshold
    motion_thr     : mean optical-flow magnitude threshold
    blur_thr       : median Laplacian variance threshold
    flow_scale     : downscale factor for optical flow (0.5 ≈ 2× faster)
    min_oneface_frac: require at least this fraction of frames to have exactly one face
    replay_check   : if True, enable scanline periodicity heuristic
    replay_thr     : threshold for replay score to trigger a fail

    NOTE: We compute optical flow on a fixed-size downscaled full frame for robustness.
    """
    metrics = {"det_align_ms":0.0, "antispoof_ms":0.0, "landmarks_ms":0.0,
               "flow_blur_ms":0.0, "total_ms":0.0}
    T0=time.perf_counter()

    # 0) Anti-coercion + aligned crops
    t=time.perf_counter()
    crops=[]; counts=[]
    for f in frames_bgr:
        det=faces_app.get(f)
        counts.append(len(det))
        if len(det)==1:
            face=max(det, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            crops.append(align112(f, face))
    metrics["det_align_ms"]=(time.perf_counter()-t)*1000

    oneface_frac = float(np.mean(np.array(counts) == 1))
    if len(crops) < 3 or oneface_frac < min_oneface_frac:
        metrics["total_ms"]=(time.perf_counter()-T0)*1000
        return False, f"multiple/no faces (one-face frames={oneface_frac:.2f})", metrics

    # 1) Passive anti-spoof on central crop
    live_p, dt = run_antispoof(spoof_sess, crops[len(crops)//2])
    metrics["antispoof_ms"] = dt
    if live_p < live_thr:
        metrics["total_ms"]=(time.perf_counter()-T0)*1000
        return False, f"anti-spoof low (p={live_p:.2f})", metrics

    # 2) Active prompts + capture-time checks (landmarks, flow, blur)
    t=time.perf_counter()
    blink=smile=pose=False; blurs=[]; flows=[]; replay_scores=[]
    prev_small=None; flow_size=None

    for i, f in enumerate(frames_bgr):
        # Blur metric on full frame (before resizes)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        blurs.append(lap_var(gray))

        # Optional replay heuristic
        if replay_check:
            replay_scores.append(scanline_score(gray))

        # Landmarks for blink/smile/pose
        res=mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0].landmark
            blink |= (ear(lm) < 0.18)
            mw=np.linalg.norm(np.array([lm[61].x,lm[61].y])-np.array([lm[291].x,lm[291].y]))
            mh=np.linalg.norm(np.array([lm[13].x,lm[13].y])-np.array([lm[14].x,lm[14].y]))
            smile |= (mw/(mh+1e-6) > 2.2)
            yaw,pitch=head_pose(lm, f.shape[1], f.shape[0]); pose |= (abs(yaw)>8)

        # --- Fixed-size optical flow on the full frame ---
        if flow_size is None:
            H, W = gray.shape[:2]
            flow_size = (max(16, int(W*flow_scale)), max(16, int(H*flow_scale)))  # (W,H)
        gray_small = cv2.resize(gray, flow_size, interpolation=cv2.INTER_AREA)

        if prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_small, gray_small,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag  = cv2.magnitude(flow[...,0], flow[...,1])  # (H,W)
            flows.append(float(mag.mean()))
        prev_small = gray_small
    metrics["landmarks_ms"]=(time.perf_counter()-t)*1000

    t=time.perf_counter()
    motion_ok=(np.mean(flows)>motion_thr)
    blur_ok=(np.median(blurs)>blur_thr)
    metrics["flow_blur_ms"]=(time.perf_counter()-t)*1000

    # Decision: prompts → replay → blur → motion
    if prompt == "blink" and not blink:
        ok, why = False, "blink not observed"
    elif prompt == "smile" and not smile:
        ok, why = False, "smile not observed"
    elif prompt == "pose" and not pose:
        ok, why = False, "pose not observed"
    elif replay_check and (len(replay_scores) >= 3) and (np.mean(replay_scores) > replay_thr):
        ok, why = False, "replay-like scanline periodicity"
    elif not blur_ok:
        ok, why = False, "excessive blur"
    elif not motion_ok:
        ok, why = False, "insufficient micro-motion"
    else:
        detail = "no active prompt" if prompt in ("none", None) else f"{prompt} ok"
        ok, why = True, f"live (p={live_p:.2f}), {detail}"

    metrics["total_ms"]=(time.perf_counter()-T0)*1000
    return ok, why, metrics

# -------------------- Landmark-aware fake blink --------------------
def eye_poly_points(lm, w, h, right=False):
    L = [33,160,158,133,153,144]       # left
    R = [362,385,387,263,373,380]      # right
    idx = R if right else L
    return np.array([[lm[i].x*w, lm[i].y*h] for i in idx], dtype=np.int32)

def draw_closed_eyelids(img, mesh):
    """Simulate a blink by collapsing eyelid polygons to their midlines."""
    out = img.copy()
    res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return out
    lm = res.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]
    for right in (False, True):
        pts = eye_poly_points(lm, w, h, right=right)
        ymin, ymax = np.min(pts[:,1]), np.max(pts[:,1])
        midy = int(0.5*(ymin + ymax))
        pts_closed = pts.copy(); pts_closed[:,1] = midy
        cv2.fillConvexPoly(out, pts_closed, color=(0,0,0))
        cv2.polylines(out, [pts_closed], True, (0,0,0), thickness=2)
    return out

# -------------------- Synthetic sequence primitives --------------------
def jitter(img, dx, dy):
    """Small translation + brightness jitter."""
    a = 0.95 + 0.1*np.random.rand(); b = np.random.randint(-6,7)
    out = cv2.convertScaleAbs(img, alpha=a, beta=b)
    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(out, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def shear_yaw_like(img, shear=0.28):
    """Perspective skew to mimic yawing left/right."""
    h,w=img.shape[:2]
    src=np.float32([[0,0],[w,0],[0,h],[w,h]])
    dst=np.float32([[0,0],[w,0],[int(shear*h),h],[w-int(shear*h),h]])
    H=cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img,H,(w,h),borderMode=cv2.BORDER_REFLECT_101)

def add_second_face(base, other, x=10, y=10, scale=0.35):
    """
    Overlay a second face onto `base` at (x,y), blending 60/40.
    Safe against edges; always matches ROI and overlay shapes.
    """
    out = base.copy()
    h, w = out.shape[:2]

    # resize the second face
    ow, oh = other.shape[1], other.shape[0]
    tw, th = max(1, int(ow * scale)), max(1, int(oh * scale))
    sf = cv2.resize(other, (tw, th), interpolation=cv2.INTER_LINEAR)

    # clamp top-left and compute bottom-right inside base
    x = int(max(0, min(x, w - 1)))
    y = int(max(0, min(y, h - 1)))
    x2 = min(w, x + sf.shape[1])
    y2 = min(h, y + sf.shape[0])

    # nothing to do if target box is degenerate
    if x2 <= x or y2 <= y:
        return out

    # crop overlay to exactly fit ROI
    sf_crop = sf[0:(y2 - y), 0:(x2 - x)]
    roi     = out[y:y2, x:x2]

    # blend and write back
    blended = (0.6 * sf_crop.astype(np.float32) + 0.4 * roi.astype(np.float32)).astype(np.uint8)
    out[y:y2, x:x2] = blended
    return out


def add_scanlines(img, period=4, strength=0.35):
    """Mimic screen replay with horizontal scanlines."""
    out = img.copy().astype(np.float32)
    for r in range(0, out.shape[0], period):
        out[r,:,:] *= (1.0 - strength)
    return np.clip(out,0,255).astype(np.uint8)

def paper_like(img):
    """Mimic printed photo: desaturate + contrast + slight blur + noise."""
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    col=cv2.GaussianBlur(col,(3,3),0)
    col=cv2.convertScaleAbs(col, alpha=1.3, beta=10)
    noise=np.random.normal(0,8, size=col.shape).astype(np.float32)
    out=np.clip(col.astype(np.float32)+noise,0,255).astype(np.uint8)
    return out

def mask_lower_face(img, frac=0.38):
    """Overlay a 'mask' over the lower face region."""
    h,w=img.shape[:2]; y1=int(h*(1.0-frac))
    out=img.copy()
    cv2.rectangle(out, (int(0.25*w), y1), (int(0.75*w), h-2), (30,30,30), thickness=-1)
    return out

# -------------------- Build all sequences --------------------
def make_sequences(faceA, faceB, mesh, N=30):
    """Return a dict: test_name → list-of-frames (BGR)."""
    seqs = {
        "IDLE":        [jitter(faceA, np.random.randint(-2,3), np.random.randint(-2,3)) for _ in range(N)],
        "POSE":        [shear_yaw_like(faceA, shear=0.28) if 8 <= i < 20 else
                        jitter(faceA, np.random.randint(-2,3), np.random.randint(-2,3)) for i in range(N)],
        "HEAD_SHAKE":  [jitter(faceA, (-1)**i * 4, 0) for i in range(N)],
        "NO_MOTION":   [faceA.copy() for _ in range(N)],
        "BLUR":        [cv2.GaussianBlur(faceA, (21,21), 0) for _ in range(N)],
        "MULTIFACE":   [add_second_face(faceA, faceB, x=10+2*i, y=10) for i in range(N)],
        "BLINK_TRY":   [draw_closed_eyelids(faceA, mesh) if 10 <= i < 14 else
                        jitter(faceA, np.random.randint(-2,3), np.random.randint(-2,3)) for i in range(N)],
        "REPLAY":      [add_scanlines(jitter(faceA, 0, 0), period=4, strength=0.35) for _ in range(N)],
        "PRINT":       [paper_like(faceA) for _ in range(N)],
        "MASK":        [mask_lower_face(faceA) for _ in range(N)],
    }
    return seqs

# -------------------- Aggregation helpers --------------------
def p95(xs):
    if not xs: return 0.0
    arr=np.sort(np.array(xs,dtype=np.float64)); k=int(math.ceil(0.95*(len(arr)-1)))
    return float(arr[k])

def summarize(rows, key):
    vals=[r[key] for r in rows]
    return {"mean":mean(vals), "std":pstdev(vals) if len(vals)>1 else 0.0, "p95":p95(vals)}

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face1", required=True, help="Path to main face image")
    ap.add_argument("--face2", required=False, help="Path to second face (for multiface); defaults to face1")
    ap.add_argument("--antispoof", default="", help="Path to anti-spoof ONNX (if provided, we use it)")
    ap.add_argument("--export-antispoof", action="store_true",
                    help="Export a tiny PyTorch anti-spoof model to ONNX (antispoof_cdcn.onnx) and use it")
    ap.add_argument("--spoof-thr", type=float, default=None,
                    help="Live prob threshold for anti-spoof. Default: 0.5 if model is used, else 0.0")
    ap.add_argument("--motion-thr", type=float, default=0.4, help="Optical-flow mean magnitude threshold")
    ap.add_argument("--blur-thr", type=float, default=60.0, help="Median Laplacian variance threshold")
    ap.add_argument("--flow-scale", type=float, default=1.0, help="Downscale factor for optical flow (0.5=2x faster)")
    ap.add_argument("--min-oneface-frac", type=float, default=0.6,
                    help="Require at least this fraction of frames to have exactly one face (default 0.6)")
    ap.add_argument("--replay-check", action="store_true",
                    help="Enable simple scanline periodicity heuristic (replay detection)")
    ap.add_argument("--replay-thr", type=float, default=2.5, help="Replay score threshold")
    ap.add_argument("--det-size", type=str, default="640x640", help="Detector input WxH (e.g., 800x800)")
    ap.add_argument("--frames", type=int, default=30, help="frames per sequence")
    ap.add_argument("--runs", type=int, default=3, help="repeat each test N times and aggregate")
    ap.add_argument("--save-csv", action="store_true", help="write per-run and summary CSVs")
    ap.add_argument("--seed", type=int, default=123, help="random seed for jitter")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # Prepare ONNX model if requested
    if args.export_antispoof and not args.antispoof:
        try:
            onnx_path = maybe_export_antispoof_onnx("antispoof_cdcn.onnx", device_pref="cuda" if USE_CUDA else "cpu")
            args.antispoof = onnx_path
            print(f"[EXPORT] Wrote ONNX anti-spoof → {onnx_path}")
        except Exception as e:
            print(f"[EXPORT] Failed to export ONNX: {e}. Continuing without anti-spoof.")
            args.antispoof = ""

    img1=cv2.imread(args.face1);  img2=cv2.imread(args.face2) if args.face2 else img1
    if img1 is None or img2 is None:
        raise SystemExit("Could not read input images.")
    print(f"[ENV] ORT providers: {ORT_PROVIDERS} | Using: {'CUDA' if USE_CUDA else 'CPU'}")

    # Init models
    faces_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    dw, dh = (int(v) for v in args.det_size.lower().split("x"))
    faces_app.prepare(ctx_id=CTX_ID, det_size=(dw, dh))
    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.6, min_tracking_confidence=0.6)
    spoof_sess = None
    if args.antispoof:
        spoof_sess = ort.InferenceSession(args.antispoof, providers=PROVIDERS)
        print(f"[ANTI-SPOOF] Loaded {args.antispoof} with {spoof_sess.get_providers()}")

    # Determine thresholds
    live_thr = (args.spoof_thr if args.spoof_thr is not None else (0.5 if spoof_sess else 0.0))
    print(f"[THR] spoof={live_thr:.2f}  motion={args.motion_thr:.2f}  blur={args.blur_thr:.1f}  "
          f"min_oneface={args.min_oneface_frac:.2f}  flow_scale={args.flow_scale:g}  det={dw}x{dh}  "
          f"replay_check={'on' if args.replay_check else 'off'}")

    # Build sequences factory (reseeds per run for variety)
    def build_seqs():
        return make_sequences(img1, img2, mesh, N=args.frames)

    # Warm-up pass to stabilize kernels / caches
    _ = liveness_check(build_seqs()["IDLE"], faces_app, mesh, spoof_sess,
                       prompt="none", live_thr=live_thr,
                       motion_thr=args.motion_thr, blur_thr=args.blur_thr, flow_scale=args.flow_scale,
                       min_oneface_frac=args.min_oneface_frac,
                       replay_check=args.replay_check, replay_thr=args.replay_thr)

    PROMPT_FOR = {
        "IDLE": "none", "POSE": "pose", "HEAD_SHAKE": "none",
        "NO_MOTION": "none", "BLUR": "none", "MULTIFACE": "none",
        "BLINK_TRY": "blink", "REPLAY": "none", "PRINT": "none", "MASK": "none",
    }

    # -------- per-run results (rows) and per-test summaries --------
    all_rows = []
    print("\n=== Liveness benchmark ({} frames/seq, {} runs) ===".format(args.frames, args.runs))
    print("{:<12} {:<4} {:<6} {:<40} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "TEST","run","OK?","WHY","det","spoof","landmk","flow","total"))

    for run_id in range(1, args.runs+1):
        # vary jitter each run
        random.seed(args.seed + run_id); np.random.seed(args.seed + run_id)
        seqs = build_seqs()

        for name, frames in seqs.items():
            prompt = PROMPT_FOR[name]
            ok, why, m = liveness_check(
                frames, faces_app, mesh, spoof_sess,
                prompt=prompt, live_thr=live_thr,
                motion_thr=args.motion_thr, blur_thr=args.blur_thr, flow_scale=args.flow_scale,
                min_oneface_frac=args.min_oneface_frac,
                replay_check=args.replay_check, replay_thr=args.replay_thr
            )
            print("{:<12} {:<4} {:<6} {:<40} {:8.1f} {:8.1f} {:8.1f} {:8.1f} {:8.1f}".format(
                name, run_id, ("PASS" if ok else "FAIL"),
                (why[:37]+"...") if len(why)>40 else why,
                m["det_align_ms"], m["antispoof_ms"], m["landmarks_ms"], m["flow_blur_ms"], m["total_ms"]
            ))
            all_rows.append({
                "test": name, "run": run_id, "ok": int(ok), "why": why,
                "det_ms": m["det_align_ms"], "spoof_ms": m["antispoof_ms"],
                "landmarks_ms": m["landmarks_ms"], "flow_ms": m["flow_blur_ms"],
                "total_ms": m["total_ms"]
            })

    # Aggregate per test
    tests = sorted(set(r["test"] for r in all_rows))
    summary = []
    print("\n--- Summary across runs ---")
    print("{:<12} {:>5} {:>7} {:>7} {:>7}   {:>7} {:>7} {:>7}".format(
        "TEST","pass%","totµ","totσ","totp95","detµ","lmµ","flowµ"))
    for t in tests:
        rows = [r for r in all_rows if r["test"] == t]
        pass_rate = 100.0 * sum(r["ok"] for r in rows) / len(rows)
        s_total = summarize(rows, "total_ms")
        s_det   = summarize(rows, "det_ms")
        s_lm    = summarize(rows, "landmarks_ms")
        s_flow  = summarize(rows, "flow_ms")
        summary.append({
            "test": t,
            "pass_rate": pass_rate,
            "total_mean": s_total["mean"], "total_std": s_total["std"], "total_p95": s_total["p95"],
            "det_mean": s_det["mean"], "lm_mean": s_lm["mean"], "flow_mean": s_flow["mean"]
        })
        print("{:<12} {:5.1f} {:7.1f} {:7.1f} {:7.1f}   {:7.1f} {:>7.1f} {:>7.1f}".format(
            t, pass_rate, s_total["mean"], s_total["std"], s_total["p95"],
            s_det["mean"], s_lm["mean"], s_flow["mean"]
        ))

    # CSV export (optional)
    if args.save_csv and all_rows:
        with open("liveness_runs.csv","w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
        with open("liveness_summary.csv","w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
        print("\n[OUT] wrote liveness_runs.csv and liveness_summary.csv")

if __name__ == "__main__":
    main()
