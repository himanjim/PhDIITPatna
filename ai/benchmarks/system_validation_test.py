#!/usr/bin/env python3
"""
This script performs a two-phase validation of a Triton-plus-FAISS face
matching pipeline. In the first phase, it traverses a directory of
identity-organised images, computes one embedding per image through
Triton, and submits each embedding to the FAISS service so that the
index is populated. In the second phase, it queries the FAISS service
again with the same embeddings and checks whether each vector retrieves
itself as the nearest neighbour within a predefined acceptance threshold.

The script is intended as a functional and consistency check for the
combined embedding-and-search workflow rather than as a full benchmark
or production verification service. Its main value is diagnostic: it
confirms that preprocessing, embedding generation, FAISS insertion,
response parsing, and threshold-based self-match validation are all
aligned with the assumptions of the larger system.
"""

import os
import glob
import cv2
import numpy as np
import requests
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

# ───────────────────────── CONFIG ───────────────────────── #
# Root folder with subfolders per person, each containing images
ROOT_DIR         = "/mnt/c/Users/himan/Documents/Image_Train"

# Service endpoints
TRITON_URL       = "localhost:8001"                # Triton gRPC
FAISS_SEARCH_URL = "http://localhost:9000/search"  # FAISS search+add

# Triton model I/O details (change if your model uses different names)
MODEL_NAME  = "buffalo_l"
INPUT_NAME  = "input.1"
OUTPUT_NAME = "683"                 # embedding tensor name
IMAGE_SIZE  = (112, 112)            # width, height expected by the model

# FAISS IndexFlatL2 reports squared L2 distance, so the operational
# threshold must also be expressed in squared form.
TAU_L2 = 1.150
DISTANCE_THRESHOLD = TAU_L2 * TAU_L2   # 1.322

# Optional HTTP timeout for FAISS requests (seconds)
HTTP_TIMEOUT = 30

# ───────────────────── IMAGE/TRITON HELPERS ───────────────────── #
# Read one image from disk and convert it into the tensor layout expected
# by the Triton-served face model. The function loads the image with
# OpenCV, rescales pixel values to floating-point form, resizes it to the
# configured model input size, and converts the layout from HWC to CHW.
# This keeps model-facing preprocessing explicit and consistent across the
# ingest and verification phases.
def preprocess(image_path: str) -> np.ndarray:
    """
    Load an image and convert to model's expected format.
    Returns a float32 array of shape (3, H, W) in [0, 1].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read {image_path}")
    img = img.astype("float32") / 255.0
    img = cv2.resize(img, IMAGE_SIZE)
    # Convert HWC -> CHW (channels first) as most ONNX face models expect NCHW
    return img.transpose(2, 0, 1)


# Submit one preprocessed image tensor to Triton and return the resulting
# one-dimensional embedding vector. A batch dimension is added because the
# deployed model expects batched input, even for single-image inference.
# The returned embedding is L2-normalised before use so that downstream
# distance comparisons remain consistent with the geometry typically
# assumed by InsightFace-style representations.
def get_embedding(triton_client: InferenceServerClient, chw_img: np.ndarray) -> np.ndarray:
    """
    Send one preprocessed image (CHW) to Triton and return the 1-D embedding.
    """
    # Triton expects a batch dimension: [1, 3, H, W]
    batch = np.expand_dims(chw_img, axis=0).astype(np.float32)

    infer_input = InferInput(INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    infer_output = InferRequestedOutput(OUTPUT_NAME)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output]
    )
    vec = result.as_numpy(OUTPUT_NAME)[0].astype(np.float32)
    # L2-normalise the embedding so that similarity behaviour remains
    # consistent with the model's intended representation space.
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec

# ───────────────────────── FAISS HELPERS ───────────────────────── #
# Submit one embedding to the FAISS service during the ingest phase. Under
# the service contract assumed here, the /search endpoint both performs a
# lookup and inserts the supplied vector into the index. This function is
# therefore used as an index-population step, while any returned search
# information is treated as incidental rather than analytically important.
def faiss_add(voter_id: int, embedding: np.ndarray):
    """
    Phase 1 "add" call.
    Per service contract, /search both searches and adds the vector.
    Here we call once to ensure the vector is inserted; we ignore the result.
    """
    payload = {"voter_id": int(voter_id), "vector": embedding.tolist()}
    r = requests.post(FAISS_SEARCH_URL, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()  # returned in case you want to inspect/log

# Submit one embedding to the FAISS service during the verification phase
# and extract the nearest-neighbour outcome from the response. The purpose
# of this call is to test whether an embedding already inserted into the
# index can retrieve itself as the closest match, subject to the chosen
# acceptance threshold. The full raw response is also returned so that
# unexpected service behaviour can be inspected if needed.
def faiss_search_once(voter_id: int, embedding: np.ndarray):
    """
    Phase 2 "verify" call.
    Calls /search again and extracts nearest neighbor id and distance.
    """
    payload = {"voter_id": int(voter_id), "vector": embedding.tolist()}
    r = requests.post(FAISS_SEARCH_URL, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    rsp = r.json()
    nn_id, dist = extract_result(rsp)
    return nn_id, dist, rsp

# Parse the nearest-neighbour identifier and distance value from the FAISS
# service response. The function accepts several plausible response shapes
# so that the validation script remains usable across small variations in
# service implementation. If none of the expected patterns is present, it
# raises an error immediately rather than silently accepting an ambiguous
# or malformed response.
def extract_result(rsp: dict):
    """
    Extract nearest-neighbor ID and distance from FAISS response.
    Supports several common shapes; adjust if your service differs.
    """
    if "nearest_id" in rsp and "distance" in rsp:
        return int(rsp["nearest_id"]), float(rsp["distance"])
    if "ids" in rsp and "distances" in rsp and rsp["ids"] and rsp["distances"]:
        return int(rsp["ids"][0]), float(rsp["distances"][0])
    if "id" in rsp and "dist" in rsp:
        return int(rsp["id"]), float(rsp["dist"])
    raise KeyError(f"Cannot extract id/distance from FAISS response: {rsp}")

# ───────────────────────────── MAIN ───────────────────────────── #
# Run the complete two-phase validation workflow. The function first
# enumerates all images beneath the configured root directory, assigns a
# deterministic integer identifier to each image, computes embeddings
# through Triton, and inserts them into the FAISS-backed service. After a
# deliberate pause for manual inspection of logs or service metrics, it
# re-queries the service with the same embeddings and checks whether each
# vector returns a self-match within the calibrated squared-L2 threshold.
# The final summary reports how many embeddings passed or failed this
# consistency check.
def main():
    # ---------- Phase 0: enumerate all images under ROOT_DIR ----------
    image_paths = []
    for sub in sorted(os.listdir(ROOT_DIR)):
        psub = os.path.join(ROOT_DIR, sub)
        if not os.path.isdir(psub):
            continue
        for ext in ("jpg", "jpeg", "png", "bmp"):
            image_paths.extend(glob.glob(os.path.join(psub, f"*.{ext}")))
    if not image_paths:
        print("No images found under", ROOT_DIR)
        return

    # Assign deterministic integer IDs to each image (1..N)
    id_map = {i: path for i, path in enumerate(sorted(image_paths), start=1)}
    print(f"Discovered {len(id_map)} images across {ROOT_DIR}")

    # ---------- Connect to Triton once ----------
    print(f"Connecting to Triton at {TRITON_URL} ...")
    triton_client = InferenceServerClient(TRITON_URL)

    # ---------- Phase 1: compute all embeddings ----------
    print("Computing embeddings via Triton ...")
    embeddings = {}
    for vid, img_path in id_map.items():
        chw = preprocess(img_path)
        emb = get_embedding(triton_client, chw)
        embeddings[vid] = emb
    print(f"Computed {len(embeddings)} embeddings.")

    # ---------- Phase 1b: add ALL embeddings to FAISS ----------
    print("Adding ALL embeddings to FAISS (ingest/build index) ...")
    for vid, emb in embeddings.items():
        try:
            _ = faiss_add(vid, emb)  # response ignored; purpose is to insert
        except Exception as e:
            print(f"  [ADD ERROR] vid={vid}: {e}")
    print("Finished adding to FAISS.\n")

    # ---------- Pause so you can inspect FAISS service (logs/metrics) ----------
    input("Press ENTER to start verification searches... ")

    # ---------- Phase 2: verify each embedding exists in FAISS ----------
    print("Verifying each embedding (expect self‑match & dist < threshold) ...")
    passed, failed = [], []
    for vid, emb in embeddings.items():
        try:
            found_id, dist, _ = faiss_search_once(vid, emb)
            if found_id == vid and dist <= DISTANCE_THRESHOLD:  # τ² comparison
                passed.append((vid, dist))
                print(f"  ✔ vid={vid:4d}  dist={dist:.4f}")
            else:
                failed.append((vid, found_id, dist))
                print(f"  ✘ vid={vid:4d}  found={found_id}  dist={dist:.4f}")
        except Exception as e:
            failed.append((vid, None, None))
            print(f"  [VERIFY ERROR] vid={vid}: {e}")

    # ---------- Summary ----------
    print("\n=== SUMMARY ===")
    print(f"Total images tested: {len(id_map)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failures detail:")
        for vid, found, dist in failed:
            print(f" - vid={vid}  got={found}  dist={dist}")

# Standard script entry point for direct command-line execution.
if __name__ == "__main__":
    main()
