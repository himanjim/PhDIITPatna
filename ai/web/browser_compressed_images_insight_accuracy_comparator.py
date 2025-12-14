
"""
BrowserCompressedImagesAccuracyComparator (InsightFace-only, Triton gRPC)
-------------------------------------------------------------------------
Nearest-neighbor evaluation (simple & practical)

For EACH original image:
  1) Compare against ALL compressed images.
  2) Pick the single NEAREST compressed image (smallest L2).
  3) Predict "Same" if nearest distance <= threshold; else "Different".

What it does
- Reads images from two folders:
    Downloads/voter_images_faces                 -> "original" images
    Downloads/voter_images_faces_compressed      -> "compressed" images
- Gets embeddings from Triton (gRPC) using InsightFace model (e.g., buffalo_l).
- L2-normalizes embeddings once, then uses L2 distance with threshold (default τ = 1.03).
- Prints metrics, confusion matrix, and sample FP/FN pairs.

Config (env or CLI)
- TRITON_URL  : gRPC endpoint (default: localhost:8001)
- MODEL_NAME  : Triton model name (default: buffalo_l)
- IN_TENSOR   : input tensor name (default: input.1)
- OUT_TENSOR  : output tensor name (default: 683)
- IMG_SIZE    : (112, 112)
- L2_THRESHOLD: default 1.03 (L2 on L2-normalized vectors)
- ORIG_DIR / COMP_DIR: override default folders

CLI examples
  python BrowserCompressedImagesAccuracyComparator.py --threshold 1.03
  python BrowserCompressedImagesAccuracyComparator.py --orig "C:/A" --comp "C:/B" --batch 32

Dependencies
  pip install tritonclient[grpc] opencv-python numpy scikit-learn pandas
"""

import os
import re
import cv2
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Triton gRPC client
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

# ───────── Logging ───────── #
LOG = logging.getLogger("Comparator")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ==============================
#         Configuration
# ==============================
def get_downloads_path() -> Path:
    if "USERPROFILE" in os.environ:
        return Path(os.path.join(os.environ["USERPROFILE"], "Downloads"))
    return Path.home() / "Downloads"


DEFAULT_DOWNLOADS = get_downloads_path()

# Folders (override with CLI / env)
DEFAULT_ORIG_DIR = os.environ.get("ORIG_DIR", str(DEFAULT_DOWNLOADS / "voter_images_faces"))
DEFAULT_COMP_DIR = os.environ.get("COMP_DIR", str(DEFAULT_DOWNLOADS / "voter_images_faces_compressed"))

# Triton (aligned with locust_benchmark.py)
TRITON_URL  = os.environ.get("TRITON_URL", "localhost:8001")
MODEL_NAME  = os.environ.get("MODEL_NAME", "buffalo_l")
IN_TENSOR   = os.environ.get("IN_TENSOR", "input.1")
OUT_TENSOR  = os.environ.get("OUT_TENSOR", "683")
IMG_SIZE    = (112, 112)  # width, height

# Threshold (L2 on normalized embeddings)
DEFAULT_L2_THRESHOLD = float(os.environ.get("L2_THRESHOLD", "1.15"))


# ==============================
#       Helper Functions
# ==============================
def collect_images(folder: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    if not os.path.isdir(folder):
        LOG.error("Folder not found: %s", folder)
        return []
    files = []
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and f.lower().endswith(exts):
            files.append(p)
    return files


def extract_identity(filename: str) -> str:
    """
    Example: "Ravi_001.png" -> "Ravi"
             "Anita-2.jpg"  -> "Anita" (letters prefix)
    """
    basename = os.path.basename(filename)
    m = re.match(r"^([A-Za-z]+)", basename)
    return m.group(1) if m else basename


def preprocess_image(path: str) -> np.ndarray:
    """
    cv2.imread -> float32/255 -> resize to 112x112 -> CHW -> NCHW
    Mirrors your earlier locust_benchmark.py behavior.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"cv2.imread failed for: {path}")
    img = img.astype("float32") / 255.0
    img = cv2.resize(img, IMG_SIZE).transpose(2, 0, 1)  # CHW
    img = np.expand_dims(img, 0)  # [1,3,112,112]
    return img


def l2_normalize(vecs: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        n = np.linalg.norm(arr) + eps
        return arr / n
    norms = np.linalg.norm(arr, axis=axis, keepdims=True) + eps
    return arr / norms


def triton_embed(client: InferenceServerClient, batch: np.ndarray) -> np.ndarray:
    """
    Sends a batch [N,3,112,112] FP32 to Triton and returns embeddings [N, D].
    Compatible with tritonclient versions that *do* or *do not* accept binary_data.
    """
    inp = InferInput(IN_TENSOR, batch.shape, "FP32")
    try:
        inp.set_data_from_numpy(batch, binary_data=True)
    except TypeError:
        inp.set_data_from_numpy(batch)

    try:
        out = InferRequestedOutput(OUT_TENSOR, binary_data=True)
    except TypeError:
        out = InferRequestedOutput(OUT_TENSOR)

    resp = client.infer(MODEL_NAME, inputs=[inp], outputs=[out])
    emb = resp.as_numpy(OUT_TENSOR)
    if emb is None:
        raise RuntimeError("Triton response has no output tensor")
    return emb.astype(np.float32)


def embed_all(paths: List[str], client: InferenceServerClient, batch_size: int = 32) -> np.ndarray:
    """
    Efficiently embed a list of image paths using mini-batches.
    Returns array [N, D], L2-normalized.
    """
    embs = []
    N = len(paths)
    done = 0
    for i in range(0, N, batch_size):
        batch_paths = paths[i:i+batch_size]
        arrs = []
        for p in batch_paths:
            try:
                arrs.append(preprocess_image(p))
            except Exception as e:
                LOG.warning("Skipping %s (%s)", p, e)
        if not arrs:
            continue
        batch = np.concatenate(arrs, axis=0)  # [B,3,112,112]
        E = triton_embed(client, batch)       # [B,D]
        embs.append(E)
        done += len(arrs)
        LOG.info("Embedded %d/%d images", done, N)

    if not embs:
        raise RuntimeError("No embeddings produced (check image folders)")
    allE = np.vstack(embs).astype(np.float32)
    allE = l2_normalize(allE, axis=1)
    return allE


# ==============================
#            Main
# ==============================
def main():
    ap = argparse.ArgumentParser(description="Nearest-neighbor accuracy check for original vs compressed faces using Triton (InsightFace).")
    ap.add_argument("--orig", default=DEFAULT_ORIG_DIR, help="Original faces folder")
    ap.add_argument("--comp", default=DEFAULT_COMP_DIR, help="Compressed faces folder")
    ap.add_argument("--threshold", type=float, default=DEFAULT_L2_THRESHOLD, help="L2 threshold on normalized embeddings (default: 1.03)")
    ap.add_argument("--batch", type=int, default=32, help="Triton batch size")
    args = ap.parse_args()

    orig_dir = args.orig
    comp_dir = args.comp
    L2_THR   = args.threshold

    LOG.info("Original dir:   %s", orig_dir)
    LOG.info("Compressed dir: %s", comp_dir)
    LOG.info("Triton URL:     %s  | model=%s  in=%s  out=%s", TRITON_URL, MODEL_NAME, IN_TENSOR, OUT_TENSOR)
    LOG.info("L2 threshold:   %.6f (on normalized embeddings)", L2_THR)

    # --- Collect images ---
    original_paths   = collect_images(orig_dir)
    compressed_paths = collect_images(comp_dir)

    if not original_paths:
        LOG.error("No images in original folder. Exiting.")
        sys.exit(2)
    if not compressed_paths:
        LOG.error("No images in compressed folder. Exiting.")
        sys.exit(2)

    # Map to identities (stable order)
    orig_identities = [extract_identity(p) for p in original_paths]
    comp_identities = [extract_identity(p) for p in compressed_paths]

    # --- Triton client ---
    TRITON = InferenceServerClient(TRITON_URL)

    # --- Embed all ---
    LOG.info("Generating embeddings via Triton ...")
    orig_embs = embed_all(original_paths, TRITON, batch_size=args.batch)   # [N1,D], L2-normalized
    comp_embs = embed_all(compressed_paths, TRITON, batch_size=args.batch) # [N2,D], L2-normalized
    LOG.info("Done. Shapes: orig=%s  comp=%s", orig_embs.shape, comp_embs.shape)

    # --- For each original: pick nearest compressed, then threshold
    LOG.info("Nearest-neighbor evaluation (one decision per original) ...")
    y_true, y_pred = [], []
    false_positives, false_negatives = [], []

    comp_arr = comp_embs  # [N2, D]
    for i, (orig_vec, orig_path, orig_name) in enumerate(zip(orig_embs, original_paths, orig_identities)):
        diffs = comp_arr - orig_vec            # [N2, D]
        dists = np.linalg.norm(diffs, axis=1)  # [N2]
        j = int(np.argmin(dists))
        best_dist = float(dists[j])
        comp_path = compressed_paths[j]
        comp_name = comp_identities[j]

        same_person = (orig_name == comp_name)
        is_match    = (best_dist <= L2_THR)

        y_true.append(1 if same_person else 0)
        y_pred.append(1 if is_match else 0)

        if is_match and not same_person:
            false_positives.append((orig_path, comp_path, best_dist))
        elif (not is_match) and same_person:
            false_negatives.append((orig_path, comp_path, best_dist))

    # --- Metrics ---
    LOG.info("Computing metrics ...")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n✅ Evaluation Metrics (L2 on normalized embeddings):")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    print("\n📋 Classification Report (one decision per original):")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Different", "Same"], zero_division=0))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 0 = Different, 1 = Same
    cm_df = pd.DataFrame(cm, index=["True: Different", "True: Same"], columns=["Pred: Different", "Pred: Same"])
    print("\n📊 Confusion Matrix:")
    print(cm_df.to_string())

    # --- Print mismatches ---
    def basename(p): return os.path.basename(p)

    print("\n❌ False Positives (Matched but different identities):")
    for orig, comp, dist in false_positives[:100]:
        print(f"FP: {basename(orig)} ↔ {basename(comp)} | L2: {dist:.4f}")
    if len(false_positives) > 100:
        print(f"... and {len(false_positives)-100} more")

    print("\n❌ False Negatives (Same identity but not matched):")
    for orig, comp, dist in false_negatives[:100]:
        print(f"FN: {basename(orig)} ↔ {basename(comp)} | L2: {dist:.4f}")
    if len(false_negatives) > 100:
        print(f"... and {len(false_negatives)-100} more")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
