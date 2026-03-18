# This module defines unit-level regression tests for the internal FAISS
# similarity service exposed by `faiss_service.py`. The tests run against
# FastAPI's in-process TestClient, so they validate service behaviour,
# request contracts, authentication, and deterministic similarity results
# without introducing network transport effects. A small fixed-dimension
# index is used so that expected nearest-neighbour outcomes remain easy to
# interpret and assert.

# Standard library
import os
import importlib

# Third-party dependencies
import numpy as np
import pytest
from fastapi.testclient import TestClient


# Configure a deterministic, CPU-only FAISS service instance for the test
# module and return the FastAPI application object under test.
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


# Return the internal-auth header required by protected endpoints in the
# service.
def _hdr():
    # faiss_service reads header 'x-internal-auth' (case-insensitive)
    return {"X-Internal-Auth": "test-token"}


# Verify that the service starts successfully and exposes a basic health
# endpoint.
def test_health(faiss_app):
    c = TestClient(faiss_app)
    r = c.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# Verify that protected endpoints reject unauthenticated requests rather
# than defaulting to permissive behaviour.
def test_auth_fail_closed(faiss_app):
    c = TestClient(faiss_app)
    r = c.get("/v1/policy")
    assert r.status_code == 403


# Verify that the policy endpoint returns both the active policy object and
# a stable SHA-256 hash suitable for audit and dashboard tooling.
def test_policy_ok(faiss_app):
    c = TestClient(faiss_app)
    r = c.get("/v1/policy", headers=_hdr())
    assert r.status_code == 200
    j = r.json()
    assert "policy" in j and "policy_hash_sha256" in j
    assert isinstance(j["policy_hash_sha256"], str) and len(j["policy_hash_sha256"]) == 64


# Verify the core functional path: explicit vector insertion followed by
# nearest-neighbour search under a deliberately strict threshold.
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


# Verify the legacy compatibility endpoints retained for older benchmark
# harnesses and scripts.
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
