# -*- coding: utf-8 -*-
#
# test_faiss_unit.py
# -----------------
# Unit-level regression tests for the internal FAISS similarity service.
#
# Scope:
#   - Validates fail-closed internal authentication.
#   - Exercises the canonical policy endpoint used by audit/public dashboard tooling.
#   - Confirms correctness of core endpoints: upsert_batch, search, search_batch.
#   - Confirms legacy compatibility endpoints (/search, /search_batch) for older harnesses.
#
# Design notes:
#   - Tests run against FastAPI TestClient (in-process), so they measure functional behaviour
#     rather than network performance.
#   - The index is initialised with a tiny deterministic configuration (EMB_DIM=8) to keep
#     runtime low and make assertions easy to interpret.
#

# Standard library
import os
import importlib

# Third-party dependencies
import numpy as np
import pytest
from fastapi.testclient import TestClient


# The fixture configures the service using environment variables and imports the module
# under test. This mirrors typical runtime configuration without needing a live server.
@pytest.fixture(scope="module")
def faiss_app():
    # Deterministic tiny CPU-only index for fast tests.
    os.environ["INTERNAL_AUTH_TOKEN"] = "test-token"
    os.environ["USE_GPU"] = "0"
    os.environ["INDEX_TYPE"] = "flat"
    os.environ["BOOTSTRAP_MODE"] = "empty"
    os.environ["ENABLE_UPDATES"] = "1"
    os.environ["EMB_DIM"] = "8"
    os.environ["TAU_L2"] = "0.50"

    mod = importlib.import_module("faiss_service")
    return mod.app


# Helper: the service expects an internal auth token in the X-Internal-Auth header.
# HTTP header lookup in Starlette is case-insensitive, so this spelling is sufficient.
def _hdr():
    # faiss_service reads header 'x-internal-auth' (case-insensitive)
    return {"X-Internal-Auth": "test-token"}


# Basic liveness check: validates that the app starts and exposes a health endpoint.
def test_health(faiss_app):
    c = TestClient(faiss_app)
    r = c.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# Security invariant: requests without a valid internal token must be rejected.
def test_auth_fail_closed(faiss_app):
    c = TestClient(faiss_app)
    r = c.get("/v1/policy")
    assert r.status_code == 403


# Policy snapshot: used for audit tooling to verify that the service is running with the
# expected configuration (thresholds, dimensions, index type).
def test_policy_ok(faiss_app):
    c = TestClient(faiss_app)
    r = c.get("/v1/policy", headers=_hdr())
    assert r.status_code == 200
    j = r.json()
    assert "policy" in j and "policy_hash_sha256" in j
    assert isinstance(j["policy_hash_sha256"], str) and len(j["policy_hash_sha256"]) == 64


# Core functional path: insert a few vectors, then verify self-match and non-match behaviour
# under a strict (small) threshold.
def test_upsert_and_search(faiss_app):
    c = TestClient(faiss_app)

    vecs = np.array([
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    ], dtype=np.float32)

    r = c.post("/v1/upsert_batch", json={"ids":[1,2,3], "vectors": vecs.tolist()}, headers=_hdr())
    assert r.status_code == 200
    assert r.json()["added"] == 3

    # Self-match should be within tau.
    r = c.post("/v1/search", json={"subject_ref":"s1","vector": vecs[0].tolist()}, headers=_hdr())
    assert r.status_code == 200
    j = r.json()
    assert j["is_match"] is True
    assert j["nearest_id"] in (1,2)

    # Far query: use a vector NOT present in index.
    # Nearest will be id=3 ([10,0,...]) with L2 distance ~= 1.0, which is > tau(0.50) => non-match.
    far = [9.0] + [0.0] * 7
    r = c.post("/v1/search", json={"subject_ref": "s2", "vector": far}, headers=_hdr())
    assert r.status_code == 200, r.text
    j2 = r.json()
    assert j2["is_match"] is False
    assert float(j2["distance_l2"]) > 0.50


# Batch search: validates the request/response schema and that per-item subject refs are
# preserved in order for downstream correlation.
def test_search_batch(faiss_app):
    c = TestClient(faiss_app)

    # Correct request model for /v1/search_batch (QueryBatch):
    payload = {
        "subject_refs": ["q1", "q2"],
        "vectors": [
            [0.0] * 8,
            [0.1] + [0.0] * 7,
        ],
    }

    r = c.post("/v1/search_batch", json=payload, headers=_hdr())
    assert r.status_code == 200, r.text

    j = r.json()
    assert "items" in j and isinstance(j["items"], list) and len(j["items"]) == 2
    assert j["items"][0]["subject_ref"] == "q1"
    assert j["items"][1]["subject_ref"] == "q2"
    assert "faiss_search_time_ms" in j


# Legacy endpoints: maintained to keep older benchmarking harnesses and scripts working.
def test_legacy_endpoints(faiss_app):
    c = TestClient(faiss_app)
    r = c.post("/search", json={"voter_id": 99, "vector":[0.0]*8}, headers=_hdr())
    assert r.status_code == 200
    j = r.json()
    assert "nearest_id" in j and "distance" in j

    r = c.post("/search_batch", json={"voter_ids":[1,2], "vectors":[[0.0]*8, [0.1]+[0.0]*7]}, headers=_hdr())
    assert r.status_code == 200
    j2 = r.json()
    assert isinstance(j2, list) and len(j2) == 2
