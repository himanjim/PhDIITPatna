# This module defines an optional end-to-end integration test for the
# FAISS service using a real face embedding extracted from an MP4 clip.
# Unlike the synthetic-vector unit tests, this test verifies that the
# service accepts and returns embeddings from the same model family used by
# the wider liveness and face-verification stack. It remains disabled by
# default so that routine test runs do not depend on heavy model assets or
# a local clip file.


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

# Extract one normalised InsightFace embedding from a real clip, insert it
# into the FAISS service, and verify that the same embedding is returned
# as a near-zero-distance self-match.
def test_faiss_with_clip_embedding():
    # Import the heavier media and face-analysis dependencies only when the
    # optional integration test has actually been enabled.
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

    # Build the InsightFace analysis object used to locate a frame
    # containing exactly one face and to compute its embedding.
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])

    # Configure detector/recogniser. det_size trades accuracy vs speed; here we keep a standard
    # setting that is robust for typical webcam clips.
    fa.prepare(ctx_id=-1, det_size=(640, 640))

    cap = cv2.VideoCapture(str(CLIP))
    assert cap.isOpened(), f"Cannot open clip: {CLIP}"
    emb = None

    # Scan a bounded number of frames to find one unambiguous single-face
    # embedding, which keeps the subsequent FAISS assertions deterministic.
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


    # Insert the extracted embedding under a fixed identifier and then
    # verify that the same embedding returns that identifier as its nearest
    # neighbour.
    r = c.post("/v1/upsert_batch", json={"ids":[123], "vectors":[emb.tolist()]}, headers=hdr)
    assert r.status_code == 200, r.text

    r = c.post("/v1/search", json={"subject_ref":"clip","vector": emb.tolist()}, headers=hdr)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["nearest_id"] == 123
    assert float(j["distance_l2"]) <= 1e-4
    assert j["is_match"] is True
