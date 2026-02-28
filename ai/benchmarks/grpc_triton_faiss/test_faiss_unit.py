# -*- coding: utf-8 -*-
#
# test_faiss_unit.py  (gRPC edition)
# ---------------------------------
# Regression tests for:
#   - faiss_grpc_server.py (FaissDedup) + admin HTTP endpoints
#   - faiss_aggregator_grpc.py (FaissAggregator) + admin HTTP endpoints
#
# These tests replace the old FastAPI in-process tests for faiss_service.py.
#
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple
import urllib.request
import json

import numpy as np
import pytest
import grpc
import faiss

# Use existing running services instead of spawning subprocesses (recommended for your setup)
FAISS_ADDR = os.getenv("FAISS_ADDR", "").strip()          # e.g. "127.0.0.1:50051"
AGG_ADDR = os.getenv("AGG_ADDR", "").strip()              # e.g. "127.0.0.1:51052"
FAISS_ADMIN_URL = os.getenv("FAISS_ADMIN_URL", "").strip()  # e.g. "http://127.0.0.1:9100"
AGG_ADMIN_URL = os.getenv("AGG_ADMIN_URL", "").strip()      # e.g. "http://127.0.0.1:9200"
TEST_TOKEN = os.getenv("TEST_TOKEN", "").strip()          # must match INTERNAL_AUTH_TOKEN in running services


def _import_stubs():
    here = Path(__file__).resolve().parent
    for base in (here, here.parent, Path.cwd()):
        gen = base / "gen"
        if (gen / "dedup_pb2.py").exists():
            sys.path.insert(0, str(gen))
            sys.path.insert(0, str(base))
            break
    import dedup_pb2, dedup_pb2_grpc  # type: ignore
    return dedup_pb2, dedup_pb2_grpc


dedup_pb2, dedup_pb2_grpc = _import_stubs()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_port(host: str, port: int, timeout_s: float = 10.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"port not ready: {host}:{port}")


def _http_get_json(url: str, token: str = "") -> Tuple[int, dict]:
    req = urllib.request.Request(url)
    if token:
        req.add_header("X-Internal-Auth", token)
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return e.code, json.loads(body)


def _build_tiny_index(path: Path, dim: int) -> None:
    idx = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
    vecs = np.array(
        [
            [0.0] * dim,                   # id=1
            [0.1] + [0.0] * (dim - 1),     # id=2
            [10.0] + [0.0] * (dim - 1),    # id=3
        ],
        dtype=np.float32,
    )
    ids = np.array([1, 2, 3], dtype=np.int64)
    idx.add_with_ids(vecs, ids)
    faiss.write_index(idx, str(path))


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
        # --- External mode: test already-running services ---
    if FAISS_ADDR and AGG_ADDR:
        if not TEST_TOKEN:
            pytest.fail("TEST_TOKEN is not set. Set TEST_TOKEN to match INTERNAL_AUTH_TOKEN of running services.")

        if not FAISS_ADMIN_URL:
            pytest.fail("FAISS_ADMIN_URL is not set. Set FAISS_ADMIN_URL (e.g. http://127.0.0.1:9100).")

        # Aggregator admin is optional in your current run; fail with a clear message if missing.
        if not AGG_ADMIN_URL:
            pytest.fail("AGG_ADMIN_URL is not set. Start aggregator with ADMIN_HTTP_LISTEN and set AGG_ADMIN_URL.")

        yield {
            "dim": int(os.getenv("EMB_DIM", "512")),
            "token": TEST_TOKEN,
            "faiss": FAISS_ADDR,
            "agg": AGG_ADDR,
            "faiss_admin": FAISS_ADMIN_URL,
            "agg_admin": AGG_ADMIN_URL,
            "p_faiss": None,
            "p_agg": None,
        }
        return
    tmp = tmp_path_factory.mktemp("faiss_stack_test")
    dim = 8
    token = "test-token"

    idx_path = tmp / "tiny.index"
    _build_tiny_index(idx_path, dim)

    faiss_port = _free_port()
    agg_port = _free_port()
    faiss_admin = _free_port()
    agg_admin = _free_port()

    # Adjust these paths if you keep tests in a different folder.
    faiss_srv = str(Path(__file__).resolve().parent / "faiss_grpc_server.py")
    agg_srv = str(Path(__file__).resolve().parent / "faiss_aggregator_grpc.py")

    env_base = os.environ.copy()
    env_base.update(
        {
            "INTERNAL_AUTH_TOKEN": token,
            "EMB_DIM": str(dim),
            "TAU_L2": "0.50",
            "USE_GPU": "0",
            "INDEX_TYPE": "flat",
            "BOOTSTRAP_MODE": "empty",
            "ALLOW_EMPTY_INDEX": "0",
            "LOAD_INDEX_PATH": str(idx_path),
            "ENABLE_UPDATES": "0",
            "FAISS_EXECUTOR_WORKERS": "8",
        }
    )

    # Start FAISS server with admin HTTP enabled
    env_faiss = env_base.copy()
    env_faiss["ADMIN_HTTP_LISTEN"] = f"127.0.0.1:{faiss_admin}"
    p_faiss = subprocess.Popen(
        [sys.executable, faiss_srv, "--listen", f"127.0.0.1:{faiss_port}"],
        env=env_faiss,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    _wait_port("127.0.0.1", faiss_port)
    _wait_port("127.0.0.1", faiss_admin)

    # Start aggregator with admin HTTP enabled
    env_agg = env_base.copy()
    env_agg["ADMIN_HTTP_LISTEN"] = f"127.0.0.1:{agg_admin}"
    p_agg = subprocess.Popen(
        [sys.executable, agg_srv, "--listen", f"127.0.0.1:{agg_port}", "--faiss", f"127.0.0.1:{faiss_port}"],
        env=env_agg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    _wait_port("127.0.0.1", agg_port)
    _wait_port("127.0.0.1", agg_admin)

    yield {
        "dim": dim,
        "token": token,
        "faiss": f"127.0.0.1:{faiss_port}",
        "agg": f"127.0.0.1:{agg_port}",
        "faiss_admin": f"http://127.0.0.1:{faiss_admin}",
        "agg_admin": f"http://127.0.0.1:{agg_admin}",
        "p_faiss": p_faiss,
        "p_agg": p_agg,
    }

    for p in (p_agg, p_faiss):
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()


def _emb_bytes(dim: int, xs) -> bytes:
    v = np.asarray(xs, dtype=np.float32).reshape(1, dim)
    return v.tobytes()


def test_admin_health(stack):
    # FAISS admin must be reachable
    code, j = _http_get_json(stack["faiss_admin"] + "/v1/health")
    assert code == 200 and j.get("status") == "ok"

    # Aggregator admin must be reachable (otherwise fail with an actionable message)
    try:
        code, j = _http_get_json(stack["agg_admin"] + "/v1/health")
    except Exception as e:
        pytest.fail(
            "Aggregator admin endpoint is not reachable. "
            "Start aggregator with ADMIN_HTTP_LISTEN=127.0.0.1:9200 and set AGG_ADMIN_URL accordingly. "
            f"Underlying error: {e}"
        )
    assert code == 200 and j.get("status") == "ok"


def test_admin_policy_fail_closed(stack):
    code, _ = _http_get_json(stack["faiss_admin"] + "/v1/policy")  # no token
    assert code == 403

    code, j = _http_get_json(stack["faiss_admin"] + "/v1/policy", token=stack["token"])
    assert code == 200
    assert "policy" in j and "policy_hash_sha256" in j
    assert len(j["policy_hash_sha256"]) == 64


@pytest.mark.asyncio
async def test_grpc_auth_fail_closed(stack):
    ch = grpc.aio.insecure_channel(stack["faiss"])
    await ch.channel_ready()
    stub = dedup_pb2_grpc.FaissDedupStub(ch)

    req = dedup_pb2.SearchRequest(query_id=1, embedding_f32=_emb_bytes(stack["dim"], [0.0] * stack["dim"]))
    with pytest.raises(grpc.aio.AioRpcError) as e:
        await stub.Search(req, timeout=2.0)  # missing metadata token
    assert e.value.code() == grpc.StatusCode.PERMISSION_DENIED
    await ch.close()


@pytest.mark.asyncio
async def test_search_direct_faiss(stack):
    md = (("x-internal-auth", stack["token"]),)
    ch = grpc.aio.insecure_channel(stack["faiss"])
    await ch.channel_ready()
    stub = dedup_pb2_grpc.FaissDedupStub(ch)

    # exact match for id=1
    req = dedup_pb2.SearchRequest(query_id=10, embedding_f32=_emb_bytes(stack["dim"], [0.0] * stack["dim"]))
    r = await stub.Search(req, timeout=2.0, metadata=md)
    # Detect external mode (no spawned process handles)
    external = stack["p_faiss"] is None

    if external:
        # Live index: just sanity-check that service returns a valid response quickly.
        assert r.nearest_id != -1
        assert r.distance_sq >= 0.0
    else:
        # Tiny index mode: deterministic IDs
        assert r.nearest_id == 1
        assert r.is_match is True

    # far query should be nearest id=3 but outside tau(0.5)
    req2 = dedup_pb2.SearchRequest(query_id=11, embedding_f32=_emb_bytes(stack["dim"], [9.0] + [0.0] * (stack["dim"] - 1)))
    r2 = await stub.Search(req2, timeout=2.0, metadata=md)
    if external:
        assert r2.nearest_id != -1
    else:
        assert r2.nearest_id == 3
        assert r2.is_match is False
        await ch.close()


@pytest.mark.asyncio
async def test_search_via_aggregator(stack):
    md = (("x-internal-auth", stack["token"]),)
    ch = grpc.aio.insecure_channel(stack["agg"])
    await ch.channel_ready()
    stub = dedup_pb2_grpc.FaissAggregatorStub(ch)

    req = dedup_pb2.SearchRequest(query_id=20, embedding_f32=_emb_bytes(stack["dim"], [0.0] * stack["dim"]))
    r = await stub.Search(req, timeout=5.0, metadata=md)
    # Instead of expecting a specific id (not meaningful for random 1M index):
    assert r.nearest_id != -1
    assert r.distance_sq >= 0.0
    assert r.batch_size >= 1
    assert r.aggregator_wait_ms >= 0.0

    await ch.close()