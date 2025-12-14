#!/usr/bin/env python3
"""
Calibrate face-match threshold τ for InsightFace embeddings with on-the-fly
face detection + 5-point alignment (112×112 crops) before embedding.

Outputs:
  • A sweep table (threshold_sweep.csv) with τ_L2, τ^2 (FAISS squared L2), FMR, FNMR, ACC, F1
  • Suggested operating points (EER-ish and target-FMR)
Notes:
  • Embeddings are L2-normalized before distance calculations.
  • FAISS IndexFlatL2 returns **squared L2** distance; compare to τ^2 unless your service sqrt()s first.
"""

import os, glob, argparse, random, csv
from collections import Counter, defaultdict
import numpy as np
import cv2

from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

# ---- on-the-fly detection + alignment (InsightFace) ----
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ---- Defaults (match your Triton config) ----
MODEL_NAME  = "buffalo_l"
INPUT_NAME  = "input.1"
OUTPUT_NAME = "683"              # 512-D embedding tensor name
IMAGE_SIZE  = (112, 112)         # (W, H)
TRITON_URL  = "localhost:8001"
DET_SIZE    = (640, 640)         # detector input size

# --------------------- Utilities --------------------- #

def list_images_with_labels(root_dir):
    paths, labels = [], []
    for person in sorted(os.listdir(root_dir)):
        pdir = os.path.join(root_dir, person)
        if not os.path.isdir(pdir):
            continue
        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            imgs.extend(glob.glob(os.path.join(pdir, ext)))
        for p in sorted(imgs):
            paths.append(p); labels.append(person)
    return paths, labels

def init_aligner(det_size=DET_SIZE):
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=det_size)
    return app

def detect_and_align_112(img_bgr, app):
    faces = app.get(img_bgr)
    if not faces:
        return None
    # choose largest face
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    crop = face_align.norm_crop(img_bgr, landmark=f.kps, image_size=IMAGE_SIZE[0])  # 112×112 BGR
    return crop

def preprocess_InsightFace(img_bgr, use_rgb=False):
    """Color order + pixel norm expected by InsightFace: (x - 127.5)/128."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img_bgr, IMAGE_SIZE)
    if use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    img = (img - 127.5) / 128.0
    chw = img.transpose(2, 0, 1)  # HWC→CHW
    return chw

def get_embedding(triton: InferenceServerClient, chw_img: np.ndarray,
                  model_name: str, input_name: str, output_name: str):
    batch = np.expand_dims(chw_img, axis=0).astype(np.float32)  # [1,3,H,W]
    infer_input = InferInput(input_name, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    infer_output = InferRequestedOutput(output_name)
    result = triton.infer(model_name, inputs=[infer_input], outputs=[infer_output])
    vec = result.as_numpy(output_name)[0]  # [512]
    return vec.astype(np.float32)

def save_cache(cache_path, embeddings, labels, paths):
    np.savez_compressed(cache_path,
        embeddings=embeddings.astype("float32"),
        labels=np.array(labels, dtype=object),
        paths=np.array(paths, dtype=object),
    )
    print(f"[CACHE] Saved: {embeddings.shape[0]} embeddings → {cache_path}")

def load_cache(cache_path):
    data = np.load(cache_path, allow_pickle=True)
    X = data["embeddings"].astype("float32")
    labels = data["labels"].tolist()
    paths = data["paths"].tolist()
    print(f"[CACHE] Loaded: {X.shape[0]} embeddings from {cache_path}")
    return X, labels, paths

def build_pairs(labels, max_pos=200_000, max_neg=200_000, seed=123):
    random.seed(seed)
    by_lab = defaultdict(list)
    for i, lab in enumerate(labels):
        by_lab[lab].append(i)

    pos_pairs = []
    for lab, idxs in by_lab.items():
        if len(idxs) < 2: continue
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                pos_pairs.append((idxs[a], idxs[b]))
                if len(pos_pairs) >= max_pos: break
            if len(pos_pairs) >= max_pos: break

    labs = list(by_lab.keys())
    neg_pairs = []
    while len(neg_pairs) < max_neg and len(labs) >= 2:
        la, lb = random.sample(labs, 2)
        ia = random.choice(by_lab[la]); ib = random.choice(by_lab[lb])
        neg_pairs.append((ia, ib))

    return pos_pairs, neg_pairs

def l2_distance(a, b): return float(np.linalg.norm(a - b))

def quick_separation_check(X, labels, sample=200, seed=42):
    rnd = random.Random(seed)
    by_lab = defaultdict(list)
    for i, lab in enumerate(labels): by_lab[lab].append(i)
    pos, neg = [], []
    for lab, idxs in by_lab.items():
        if len(idxs) >= 2 and len(pos) < sample:
            a,b = rnd.sample(idxs, 2); pos.append((a,b))
    labs = list(by_lab.keys())
    while len(neg) < sample and len(labs) >= 2:
        la,lb = rnd.sample(labs,2); ia,ib = rnd.choice(by_lab[la]), rnd.choice(by_lab[lb])
        neg.append((ia,ib))
    pd = [l2_distance(X[i],X[j]) for (i,j) in pos] if pos else [0.0]
    nd = [l2_distance(X[i],X[j]) for (i,j) in neg] if neg else [0.0]
    print(f"[CHECK] mean L2: pos={np.mean(pd):.3f}  neg={np.mean(nd):.3f}  "
          f"min/max pos=({np.min(pd):.3f},{np.max(pd):.3f})  "
          f"min/max neg=({np.min(nd):.3f},{np.max(nd):.3f})")

def sweep_thresholds(X, labels, grid=(0.6, 1.4, 0.02), target_fmr=None,
                     max_pos=200_000, max_neg=200_000):
    pos_pairs, neg_pairs = build_pairs(labels, max_pos=max_pos, max_neg=max_neg)
    print(f"[PAIRS] positives: {len(pos_pairs)}   negatives: {len(neg_pairs)}")
    if len(pos_pairs) == 0 or len(neg_pairs) == 0:
        raise SystemExit("[PAIRS] Need at least one positive and one negative pair.")

    pos_d = np.fromiter((l2_distance(X[i],X[j]) for (i,j) in pos_pairs), dtype=np.float32)
    neg_d = np.fromiter((l2_distance(X[i],X[j]) for (i,j) in neg_pairs), dtype=np.float32)

    s,e,step = grid
    s = max(0.0, s); e = min(e, float(np.sqrt(2.0)))
    if s >= e: raise SystemExit(f"[GRID] invalid range: start={s} end={e}")
    thresholds = np.arange(s, e + 1e-12, step, dtype=np.float32)

    rows = []; best_eer_gap, eer_row, target_row = 1e9, None, None
    for t in thresholds:
        tp = np.sum(pos_d <= t); fn = len(pos_d) - tp
        fp = np.sum(neg_d <= t); tn = len(neg_d) - fp
        fmr  = fp / max(1, len(neg_d))
        fnmr = fn / max(1, len(pos_d))
        acc  = (tp + tn) / max(1, len(pos_d)+len(neg_d))
        prec = tp / max(1, tp+fp)
        rec  = tp / max(1, tp+fn)
        f1   = 0.0 if (prec+rec)==0 else (2*prec*rec)/(prec+rec)

        rows.append({
            "tau_L2": float(t),
            "tau_squared": float(t*t),
            "cosine_equiv": float(1.0 - (t*t)/2.0),
            "FMR": float(fmr), "FNMR": float(fnmr),
            "ACC": float(acc), "F1": float(f1),
        })

        gap = abs(fmr - fnmr)
        if gap < best_eer_gap: best_eer_gap, eer_row = gap, rows[-1]
        # pick the largest threshold that still satisfies FMR <= target_fmr (minimizes FNMR)
        if target_fmr is not None and (fmr <= target_fmr) and (target_row is None or t > target_row["tau_L2"]):
            target_row = rows[-1]

    print("\n=== Threshold sweep (true L2) ===")
    print("tau_L2  tau^2    cosine_eq  FMR      FNMR     ACC      F1")
    for r in rows:
        print(f"{r['tau_L2']:.3f}   {r['tau_squared']:.3f}   {r['cosine_equiv']:.3f}   "
              f"{r['FMR']:.4f}   {r['FNMR']:.4f}   {r['ACC']:.4f}   {r['F1']:.4f}")

    print("\n--- Suggested operating points ---")
    if target_fmr is not None and target_row is not None:
        r = target_row
        print(f"Target-FMR≈{target_fmr:g} → τ_L2={r['tau_L2']:.3f} , τ^2={r['tau_squared']:.3f} , "
              f"cos≈{r['cosine_equiv']:.3f} | FMR={r['FMR']:.5f}, FNMR={r['FNMR']:.5f}, ACC={r['ACC']:.4f}")
    r = eer_row
    print(f"EER-ish (FMR≈FNMR) → τ_L2={r['tau_L2']:.3f} , τ^2={r['tau_squared']:.3f} , "
          f"cos≈{r['cosine_equiv']:.3f} | FMR={r['FMR']:.5f}, FNMR={r['FNMR']:.5f}, ACC={r['ACC']:.4f}")
    return rows, eer_row, target_row

# --------------------- Main --------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root (folders per identity).")
    ap.add_argument("--triton-url", type=str, default=TRITON_URL)
    ap.add_argument("--model", type=str, default=MODEL_NAME)
    ap.add_argument("--input-name", type=str, default=INPUT_NAME)
    ap.add_argument("--output-name", type=str, default=OUTPUT_NAME)
    ap.add_argument("--cache", type=str, default="embeddings_cache.npz")
    ap.add_argument("--recompute", action="store_true", help="Force re-embed even if cache exists.")
    ap.add_argument("--grid", type=str, default="0.6:1.6:0.02",
                    help="Threshold grid over true L2 as start:end:step")
    ap.add_argument("--target-fmr", type=float, default=0.001, help="e.g., 0.001 = 0.1%%")
    ap.add_argument("--max-pos", type=int, default=200_000)
    ap.add_argument("--max-neg", type=int, default=200_000)
    ap.add_argument("--rgb", action="store_true", help="Convert BGR→RGB before Triton")
    ap.add_argument("--det-size", type=str, default="640x640", help="Detector input, e.g., 640x640")
    args = ap.parse_args()

    # Enumerate dataset
    paths, labels = list_images_with_labels(args.root)
    if not paths: raise SystemExit(f"No images found under {args.root}")
    counts = Counter(labels)
    num_ids = len(counts)
    num_ids_with_2plus = sum(1 for c in counts.values() if c >= 2)
    if num_ids < 2 or num_ids_with_2plus == 0:
        raise SystemExit("[DATA] Need ≥2 identities and ≥2 images for at least one identity "
                         f"(found {num_ids} ids; {num_ids_with_2plus} ids with ≥2 images).")
    print(f"[DATA] Found {len(paths)} images across {num_ids} identities.")

    # Cache (trust only if filenames match)
    X = None
    if (not args.recompute) and os.path.exists(args.cache):
        X, labels_cached, paths_cached = load_cache(args.cache)
        same_len = (len(labels_cached) == len(labels))
        same_set = (sorted(os.path.basename(p) for p in paths_cached) ==
                    sorted(os.path.basename(p) for p in paths))
        if not (same_len and same_set):
            print("[WARN] Cache does not match current dataset; recomputing.")
            X = None

    # Init aligner
    w,h = (int(v) for v in args.det_size.lower().split("x"))
    aligner = init_aligner(det_size=(w,h))
    print("[ALIGN] InsightFace detector ready.")

    # Compute embeddings if needed
    if X is None:
        print(f"[TRITON] Connecting to {args.triton_url} ...")
        triton = InferenceServerClient(args.triton_url)
        # health checks
        try:
            if not triton.is_server_live():  raise RuntimeError("Triton is not live.")
            if not triton.is_server_ready(): raise RuntimeError("Triton is not ready.")
            if not triton.is_model_ready(args.model):
                raise RuntimeError(f"Model '{args.model}' is not ready on Triton.")
        except Exception as e:
            raise SystemExit(f"[TRITON] health check failed: {e}")

        vecs, kept_paths, kept_labels = [], [], []
        skipped = 0
        for i,(p,lab) in enumerate(zip(paths,labels),1):
            img = cv2.imread(p)
            if img is None: skipped += 1; continue
            crop = detect_and_align_112(img, aligner)
            if crop is None: skipped += 1; continue
            chw = preprocess_InsightFace(crop, use_rgb=args.rgb)
            v = get_embedding(triton, chw, args.model, args.input_name, args.output_name)
            vecs.append(v); kept_paths.append(p); kept_labels.append(lab)
            if i % 100 == 0: print(f"  processed {i}/{len(paths)}")
        if skipped: print(f"[ALIGN] Skipped {skipped} images (no detection / IO fail).")
        if not vecs: raise SystemExit("[EMBED] No embeddings computed. Check alignment/dataset.")
        X = np.vstack(vecs).astype("float32")
        save_cache(args.cache, X, kept_labels, kept_paths)
        labels = kept_labels  # keep only those we embedded

    # Normalize to unit sphere (ArcFace geometry)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # Quick separation sanity check
    quick_separation_check(X, labels)

    # Parse & clamp grid
    s,e,st = [float(x) for x in args.grid.split(":")]
    s = max(0.0, s); e = min(e, float(np.sqrt(2.0)))
    if s >= e: raise SystemExit(f"[GRID] invalid range: start={s} end={e}")

    # Sweep thresholds
    rows, eer_row, target_row = sweep_thresholds(
        X, labels, grid=(s, e, st),
        target_fmr=args.target_fmr,
        max_pos=args.max_pos, max_neg=args.max_neg
    )

    # Save CSV
    out_csv = "threshold_sweep.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tau_L2","tau_squared","cosine_equiv","FMR","FNMR","ACC","F1"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"\n[OUT] Wrote sweep table → {out_csv}")

    # FAISS units note
    print("\n[NOTE] Your FAISS service returns squared L2 distance.")
    print("      → If you keep comparing FAISS 'distance' directly, use τ_squared (not τ_L2).")
    print("      → If you change your service to sqrt the distance, use τ_L2.")

if __name__ == "__main__":
    main()
