# -*- coding: utf-8 -*-
#
# test_faiss_clip_embedding.py
# -------------------------
# Optional end-to-end integration test: MP4 clip → InsightFace embedding → FAISS upsert/search.
#
# Rationale:
#   - Unit tests validate API contracts and numeric behaviour with synthetic vectors.
#   - This test validates that the service also works with *real* face embeddings produced by
#     the same family of models used in the LVS stack (InsightFace 'buffalo_l').
#
# Execution model:
#   - Disabled by default to keep CI lightweight.
#   - Enable by setting: RUN_FAISS_CLIP_TEST=1 and FAISS_TEST_CLIP=/abs/path/to/clip.mp4
#
# Operational notes:
#   - On first run, InsightFace may download model assets into the user profile directory.
#   - The clip should contain a clear, single face for at least one frame.
#
import os
import importlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


# Guardrails to avoid heavy dependencies in routine test runs.
RUN = os.getenv("RUN_FAISS_CLIP_TEST", "0") == "1"
CLIP = os.getenv("FAISS_TEST_CLIP", "").strip()

@pytest.mark.skipif(not RUN, reason="Set RUN_FAISS_CLIP_TEST=1 to enable")
@pytest.mark.skipif(not CLIP, reason="Set FAISS_TEST_CLIP=/abs/path/to/clip.mp4")

# The test intentionally searches for a frame with *exactly one* detected face to avoid
# ambiguous embeddings (multi-face) and to keep the assertion deterministic.
def test_faiss_with_clip_embedding():
    # Heavy deps imported only when enabled.
    # Heavy dependencies are imported lazily so that the module can be imported without them.
    import cv2
    from insightface.app import FaceAnalysis

    # Configure FAISS service for normal 512-d embeddings.
    os.environ["INTERNAL_AUTH_TOKEN"] = "test-token"
    os.environ["USE_GPU"] = "0"
    os.environ["INDEX_TYPE"] = "flat"
    os.environ["BOOTSTRAP_MODE"] = "empty"
    os.environ["ENABLE_UPDATES"] = "1"
    os.environ["EMB_DIM"] = "512"
    # Threshold value is aligned with the calibrated L2 threshold used in the paper.
    os.environ["TAU_L2"] = "1.150"  # typical threshold for L2-normalized InsightFace


    # Import the service as a module and test it in-process using FastAPI's TestClient.
    mod = importlib.import_module("faiss_service")
    app = mod.app
    c = TestClient(app)
    hdr = {"X-Internal-Auth": "test-token"}

    # Build InsightFace detector/recognizer.
    # Note: First run may download models into ~/.insightface
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])

    # Configure detector/recogniser. det_size trades accuracy vs speed; here we keep a standard
    # setting that is robust for typical webcam clips.
    fa.prepare(ctx_id=-1, det_size=(640, 640))

    cap = cv2.VideoCapture(str(CLIP))
    assert cap.isOpened(), f"Cannot open clip: {CLIP}"
    emb = None

    # Frame scan is capped to bound runtime and to avoid dependence on clip length.
    # Scan up to ~120 frames to find exactly one face.
    for _ in range(120):
        ok, frame = cap.read()
        if not ok:
            break
        faces = fa.get(frame)
        if len(faces) == 1 and getattr(faces[0], "embedding", None) is not None:
            e = faces[0].embedding.astype(np.float32, copy=False)
            n = float(np.linalg.norm(e) + 1e-12)
            # Normalise embeddings to unit norm so that L2 distances are comparable across frames.
            emb = (e / n).astype(np.float32, copy=False)
            break

    cap.release()
    assert emb is not None, "Could not extract a single-face embedding from the clip (try a clearer clip/frame)."


    # Self-match: searching for the exact inserted embedding must return the same ID with
    # near-zero distance (numerical noise only).
    # Upsert and self-search
    r = c.post("/v1/upsert_batch", json={"ids":[123], "vectors":[emb.tolist()]}, headers=hdr)
    assert r.status_code == 200, r.text

    r = c.post("/v1/search", json={"subject_ref":"clip","vector": emb.tolist()}, headers=hdr)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["nearest_id"] == 123
    assert float(j["distance_l2"]) <= 1e-4
    assert j["is_match"] is True
