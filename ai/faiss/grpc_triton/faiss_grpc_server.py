#!/usr/bin/env python3
"""
This module implements a gRPC-based FAISS similarity service for internal
de-duplication and benchmarking workloads. It exposes single-query search,
batch search, and append-only batch ingest operations over embeddings
transmitted as raw float32 bytes. The service is designed to support
high-throughput nearest-neighbour retrieval while keeping the wire format
compact and avoiding JSON-related transport overhead.

Operationally, the implementation separates the latency-critical search
path from buffered index updates, supports both CPU and GPU FAISS
backends, and can optionally replicate accepted ingest operations from a
primary node to follower replicas. The code therefore serves two roles at
once: it is a functional search service and a systems component whose
timing and queueing behaviour can be studied under benchmark load.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import faiss
import grpc

# ---- Admin policy/metrics endpoints (HTTP) -------------------------------
import hashlib
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# Locate the generated protobuf stubs from project-local paths so that the
# service can run from a checked-out source tree without requiring manual
# PYTHONPATH configuration. This improves reproducibility across local and
# experimental deployments.
ROOT = Path(__file__).resolve().parent
GEN = ROOT / "gen"
if GEN.exists():
    sys.path.insert(0, str(GEN))
    sys.path.insert(0, str(ROOT))

try:
    import dedup_pb2, dedup_pb2_grpc  # type: ignore
except Exception:
    # Fallback when using package-style imports
    from gen import dedup_pb2, dedup_pb2_grpc  # type: ignore


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EMB_DIM = int(os.getenv("EMB_DIM", "512"))
TAU_L2 = float(os.getenv("TAU_L2", "1.150"))
TAU_SQ = TAU_L2 * TAU_L2

INTERNAL_AUTH_TOKEN = os.getenv("INTERNAL_AUTH_TOKEN", "").strip()
if not INTERNAL_AUTH_TOKEN:
    raise RuntimeError("INTERNAL_AUTH_TOKEN must be set (fail-closed).")

USE_GPU = os.getenv("USE_GPU", "1") == "1"
GPU_ID = int(os.getenv("GPU_ID", "0"))
INDEX_TYPE = os.getenv("INDEX_TYPE", "flat").lower()  # flat|hnsw

HNSW_M = int(os.getenv("HNSW_M", "32"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "64"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))

BOOTSTRAP_MODE = os.getenv("BOOTSTRAP_MODE", "empty").lower()  # empty|random
RANDOM_N = int(os.getenv("RANDOM_N", "0"))
RANDOM_CHUNK = int(os.getenv("RANDOM_CHUNK", "100000"))

ALLOW_EMPTY_INDEX = os.getenv("ALLOW_EMPTY_INDEX", "1") == "1"
LOAD_INDEX_PATH = os.getenv("LOAD_INDEX_PATH", "").strip()

ID_STORE = os.getenv("ID_STORE", "1") == "1"

# Threading defaults: GPU indices are typically safest with workers=1.
_default_workers = 1 if USE_GPU else 64
FAISS_EXECUTOR_WORKERS = int(os.getenv("FAISS_EXECUTOR_WORKERS", str(_default_workers)))

# CPU OpenMP tuning
FAISS_OMP_THREADS = os.getenv("FAISS_OMP_THREADS", "").strip()
if FAISS_OMP_THREADS:
    try:
        faiss.omp_set_num_threads(int(FAISS_OMP_THREADS))
    except Exception:
        pass

# Updates (append-only)
ENABLE_UPDATES = os.getenv("ENABLE_UPDATES", "1") == "1"
UPDATE_BATCH_SIZE = int(os.getenv("UPDATE_BATCH_SIZE", "4096"))
UPDATE_BATCH_MS = int(os.getenv("UPDATE_BATCH_MS", "100"))
UPDATE_MAX_QUEUE = int(os.getenv("UPDATE_MAX_QUEUE", "200000"))

# Replica replication (PRIMARY fan-out)
IS_PRIMARY = os.getenv("IS_PRIMARY", "0") == "1"
REPLICA_FANOUT = [s.strip() for s in os.getenv("REPLICA_FANOUT", "").split(",") if s.strip()]
SYNC_REPLICATION = os.getenv("SYNC_REPLICATION", "0") == "1"
STRICT_REPLICATION = os.getenv("STRICT_REPLICATION", "0") == "1"
REPLICA_TIMEOUT_S = float(os.getenv("REPLICA_TIMEOUT_S", "2.0"))

# gRPC limits
GRPC_MAX_MB = int(os.getenv("GRPC_MAX_MB", "64"))

# Admin HTTP endpoints (internal only; same token as gRPC metadata)
ADMIN_HTTP_LISTEN = os.getenv("ADMIN_HTTP_LISTEN", "").strip()  # e.g. "127.0.0.1:9100"
METRICS_WINDOW = int(os.getenv("METRICS_WINDOW", "2048"))


# Return the gRPC transport options used by the service. These settings
# bound message sizes and apply conservative keepalive behaviour so that
# long-running internal workloads remain stable without excessive control-
# plane traffic.
def _grpc_options():
    """
    Return gRPC transport options with conservative keepalive behaviour and
    message-size limits suitable for high-throughput internal workloads.
    """
    return [
        ("grpc.max_receive_message_length", GRPC_MAX_MB * 1024 * 1024),
        ("grpc.max_send_message_length", GRPC_MAX_MB * 1024 * 1024),

        ("grpc.keepalive_time_ms", 120_000),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.keepalive_permit_without_calls", 0),
        ("grpc.http2.max_pings_without_data", 2),
        ("grpc.http2.min_time_between_pings_ms", 10_000),
        ("grpc.http2.min_ping_interval_without_data_ms", 10_000),
    ]
    
# ------------------------- Policy snapshot + metrics -------------------------
# Serialize a dictionary into a deterministic JSON byte sequence so that
# hashes remain stable across runs and across non-semantic variations in
# key order or whitespace. This is used for policy reporting rather than
# for the search path itself.
def _canon_json_bytes(obj: dict) -> bytes:
    """
    Deterministic JSON for stable hashing.
    (Sorted keys + no whitespace so the hash is reproducible across runs.)
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

# Compute a SHA-256 digest for the supplied byte sequence and return the
# hexadecimal form used by the administrative policy endpoint.
def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# Maintain a bounded rolling window of recent latency observations and
# provide compact percentile summaries for the administrative metrics
# interface. The class is intentionally lightweight and is not intended to
# replace a full telemetry system.
class _LatencyStats:
    """Fixed-window latency summary used for lightweight administrative metrics."""
    def __init__(self, window: int):
        self._q = deque(maxlen=window)

    def add_ms(self, ms: float) -> None:
        self._q.append(float(ms))

    def summary(self) -> dict:
        if not self._q:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        xs = sorted(self._q)
        n = len(xs)
        return {
            "count": n,
            "p50_ms": xs[int(0.50 * (n - 1))],
            "p95_ms": xs[int(0.95 * (n - 1))],
            "p99_ms": xs[int(0.99 * (n - 1))],
            "max_ms": xs[-1],
        }

# Build the policy snapshot exposed by the administrative API. The object
# describes the active service configuration in a form suitable for review
# and reproducibility checks, while deliberately excluding secrets and
# authentication material.
def _policy_obj_faiss(mgr: FaissManager) -> dict:
    # The policy object is intended for external inspection and therefore
    # must not contain secrets or authentication material.
    return {
        "service": "faiss_grpc_server",
        "version": "v3.2",
        "emb_dim": EMB_DIM,
        "metric": "L2",
        "tau_l2": TAU_L2,
        "index_type": INDEX_TYPE,
        "use_gpu": bool(mgr.use_gpu),
        "gpu_id": GPU_ID if mgr.use_gpu else None,
        "id_store": bool(ID_STORE),
        "bootstrap_mode": BOOTSTRAP_MODE,
        "random_n": RANDOM_N,
        "random_chunk": RANDOM_CHUNK,
        "allow_empty_index": bool(ALLOW_EMPTY_INDEX),
        "updates_enabled": bool(ENABLE_UPDATES),
        "update_batch_size": UPDATE_BATCH_SIZE,
        "update_batch_ms": UPDATE_BATCH_MS,
        "update_max_queue": UPDATE_MAX_QUEUE,
        "grpc_max_mb": GRPC_MAX_MB,
        "executor_workers": FAISS_EXECUTOR_WORKERS,
    }

# Global admin state set by serve() so the handler can read metrics safely.
_ADMIN_STATE = {
    "mgr": None,
    "svc": None,
    "started_ts": time.time(),
}

# Expose lightweight administrative HTTP endpoints for health, policy, and
# recent service metrics. These endpoints are operational aids and are
# intentionally kept separate from the gRPC search and ingest interface.
class _AdminHandler(BaseHTTPRequestHandler):
    server_version = "faiss-admin/1.0"

    # Return a compact JSON response with the supplied HTTP status code.
    # This helper keeps the administrative interface simple and machine-
    # readable.
    def _send_json(self, code: int, obj: dict) -> None:
        b = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    # Validate the internal administrative token for endpoints that expose
    # policy or metrics information. Health-style checks remain separately
    # controlled by the request path.
    def _auth_ok(self) -> bool:
        # Match old service semantics: X-Internal-Auth header, fail-closed.
        tok = self.headers.get("X-Internal-Auth", "")
        return tok == INTERNAL_AUTH_TOKEN

    # Serve the supported administrative GET endpoints. Depending on the
    # path and authentication state, the method returns health information,
    # a policy snapshot, or recent service metrics derived from in-process
    # state.
    def do_GET(self):  # noqa: N802
        if self.path in ("/v1/health", "/ping"):
            mgr = _ADMIN_STATE["mgr"]
            nt = int(mgr.ntotal()) if mgr else 0
            return self._send_json(200, {"status": "ok", "ntotal": nt})

        if not self._auth_ok():
            return self._send_json(403, {"detail": "forbidden"})

        mgr = _ADMIN_STATE["mgr"]
        svc = _ADMIN_STATE["svc"]
        if mgr is None or svc is None:
            return self._send_json(503, {"detail": "not_ready"})

        if self.path == "/v1/policy":
            pol = _policy_obj_faiss(mgr)
            canon = _canon_json_bytes(pol)
            return self._send_json(200, {"policy": pol, "policy_hash_sha256": _sha256_hex(canon)})

        if self.path == "/v1/metrics":
            return self._send_json(
                200,
                {
                    "ntotal": int(mgr.ntotal()),
                    "search_rpc_ms": svc.lat_search_rpc.summary(),
                    "ingest_enqueue_ms": svc.lat_ingest_enq.summary(),
                    "update_queue_depth": int(svc._upd_q.qsize()) if ENABLE_UPDATES else 0,
                    "started_ts": _ADMIN_STATE["started_ts"],
                },
            )

        return self._send_json(404, {"detail": "not_found"})

# Start the auxiliary administrative HTTP server in a daemon thread so
# that operational inspection can run alongside the asyncio-based gRPC
# service without interfering with its event loop.
def _start_admin_http(listen: str) -> None:
    # Start the administrative HTTP server in a daemon thread. This
    # interface is intended only for localhost or other controlled
    # internal-network deployment.
    host, port_s = listen.rsplit(":", 1)
    httpd = ThreadingHTTPServer((host, int(port_s)), _AdminHandler)
    t = threading.Thread(target=httpd.serve_forever, name="faiss-admin-http", daemon=True)
    t.start()


# Enforce a fail-closed internal authentication boundary for all gRPC
# methods. Requests must present the expected shared token in metadata or
# they are rejected before reaching the service implementation.
class InternalAuthInterceptor(grpc.aio.ServerInterceptor):
    """Fail-closed internal auth via metadata x-internal-auth."""

    def __init__(self, expected_token: str):
        self.expected = expected_token

    # Intercept each incoming RPC and validate its internal-auth metadata.
    # On failure, return an aborting handler so that unauthorised traffic
    # is rejected consistently and early in the serving path.
    async def intercept_service(self, continuation, handler_call_details):
        md = dict(handler_call_details.invocation_metadata or [])
        tok = md.get("x-internal-auth", "")
        if tok != self.expected:
            async def abort_behavior(request, context):
                # grpc.aio: abort() is a coroutine and must be awaited
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "forbidden")
            return grpc.aio.unary_unary_rpc_method_handler(abort_behavior)
        return await continuation(handler_call_details)


# -----------------------------------------------------------------------------
# Index build helpers
# -----------------------------------------------------------------------------
# Construct the configured CPU FAISS index. CPU-backed indices are wrapped
# in IndexIDMap2 so that application-level identifiers can be stored and
# retrieved directly alongside vector positions.
def _make_cpu_index() -> faiss.Index:
    """
    CPU indices are wrapped in IndexIDMap2 so we can store application-level ids directly.
    """
    if INDEX_TYPE == "hnsw":
        base = faiss.IndexHNSWFlat(EMB_DIM, HNSW_M)
        base.hnsw.efSearch = HNSW_EF_SEARCH
        base.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        return faiss.IndexIDMap2(base)

    return faiss.IndexIDMap2(faiss.IndexFlatL2(EMB_DIM))

# Attempt to construct a GPU-backed FlatL2 FAISS index using FP16 storage.
# GPU mode is enabled only for flat indices here, and identifier mapping is
# maintained separately because direct GPU-side ID mapping can be less
# stable under concurrent service workloads.

def _make_gpu_index_flat_fp16() -> Optional[faiss.Index]:
    """
    GPU FlatL2 index with FP16 storage.

    We DO NOT wrap in IndexIDMap2 on GPU because FAISS GPU ID mapping has historically
    caused stability issues under concurrency. Instead we keep a parallel id-store array
    (mgr.ids) aligned with vector positions.

    Returns None if GPU unavailable or INDEX_TYPE != flat.
    """
    if not USE_GPU or INDEX_TYPE != "flat":
        return None

    try:
        res = faiss.StandardGpuResources()

        tmp = os.getenv("FAISS_GPU_TEMP_BYTES", "").strip()
        if tmp:
            try:
                res.setTempMemory(int(tmp))
            except Exception:
                pass

        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = GPU_ID
        cfg.useFloat16 = True

        idx = faiss.GpuIndexFlatL2(res, EMB_DIM, cfg)
        _ = idx.search(np.zeros((1, EMB_DIM), dtype=np.float32), 1)  # warm-up
        return idx
    except Exception:
        return None

# Generate synthetic vectors and aligned integer identifiers for optional
# index bootstrapping. This path is intended to emulate index size and
# search load for systems benchmarking; it does not represent real face
# embeddings or biometric decision behaviour.
def _bootstrap_random_vectors(n: int, start_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic vectors used to emulate index size for benchmarking.
    (These are not real face embeddings.)
    """
    rng = np.random.default_rng(seed=42 + start_id)
    vecs = rng.standard_normal((n, EMB_DIM)).astype(np.float32, copy=False)
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    return vecs, ids


# -----------------------------------------------------------------------------
# Manager: index + optional id-store mapping for GPU-flat
# -----------------------------------------------------------------------------
# Hold the active FAISS index together with the metadata required to
# interpret its contents, including whether the index is GPU-backed and,
# when needed, the parallel identifier store aligned to vector positions.
@dataclass
class FaissManager:
    index_srv: faiss.Index
    use_gpu: bool
    # For GPU-flat, ids[pos] gives application-level id for vector at position pos.
    ids: Optional[np.ndarray]
    # Return the current number of indexed vectors as an integer suitable
    # for operational reporting and service checks.
    def ntotal(self) -> int:
        return int(self.index_srv.ntotal)

# Build and initialise the FAISS manager according to the configured
# environment. Depending on settings, this may load an existing CPU index,
# construct a new CPU or GPU index, and optionally bootstrap the index
# with synthetic vectors to emulate large-scale operation.
def build_manager() -> FaissManager:
    ids: Optional[np.ndarray] = None

    # 1) Load from disk (CPU index). Optionally move to GPU for flat.
    if LOAD_INDEX_PATH:
        cpu = faiss.read_index(LOAD_INDEX_PATH)
        if USE_GPU and INDEX_TYPE == "flat":
            try:
                res = faiss.StandardGpuResources()
                gpu = faiss.index_cpu_to_gpu(res, GPU_ID, cpu)
                return FaissManager(index_srv=gpu, use_gpu=True, ids=None)
            except Exception:
                return FaissManager(index_srv=cpu, use_gpu=False, ids=None)
        return FaissManager(index_srv=cpu, use_gpu=False, ids=None)

    # 2) Build new (GPU preferred if possible)
    gpu = _make_gpu_index_flat_fp16()
    if gpu is not None:
        idx = gpu
        use_gpu = True
        ids = np.empty((0,), dtype=np.int64)
    else:
        idx = _make_cpu_index()
        use_gpu = False

    # 3) Optional synthetic bootstrap for scale realism
    if BOOTSTRAP_MODE == "random":
        n_total = RANDOM_N if RANDOM_N > 0 else 1_000_000
        start = 0
        while start < n_total:
            n = min(RANDOM_CHUNK, n_total - start)
            vecs, vid = _bootstrap_random_vectors(n, start)
            if use_gpu:
                idx.add(vecs)
                ids = np.concatenate([ids, vid]) if ids is not None else None
            else:
                # CPU: IndexIDMap2 supports add_with_ids
                idx.add_with_ids(vecs, vid)
            start += n

    return FaissManager(index_srv=idx, use_gpu=use_gpu, ids=ids)


# -----------------------------------------------------------------------------
# Embedding payload decoding
# -----------------------------------------------------------------------------
_EXPECT_BYTES = EMB_DIM * 4

# Decode a batch of raw embedding byte strings into a float32 matrix of
# shape [N, EMB_DIM]. The function validates payload lengths first and then
# performs a single concatenation so that decoding remains efficient for
# batch RPCs.
def _bytes_list_to_mat(bs: List[bytes]) -> np.ndarray:
    """
    Efficient batch decode:
      - validate each element length
      - concatenate once
      - view as float32 matrix [N, EMB_DIM]
    """
    for b in bs:
        if len(b) != _EXPECT_BYTES:
            raise ValueError(f"embedding_f32 must be {_EXPECT_BYTES} bytes, got {len(b)}")

    blob = b"".join(bs)
    return np.frombuffer(blob, dtype=np.float32).reshape(len(bs), EMB_DIM)

# Decode one raw embedding payload into a float32 matrix with a leading
# batch dimension of size one, after validating the expected byte length.
def _bytes_to_single(b: bytes) -> np.ndarray:
    if len(b) != _EXPECT_BYTES:
        raise ValueError(f"embedding_f32 must be {_EXPECT_BYTES} bytes, got {len(b)}")
    return np.frombuffer(b, dtype=np.float32).reshape(1, EMB_DIM)


# -----------------------------------------------------------------------------
# Update queue entries
# -----------------------------------------------------------------------------
# Represent one pending append-only index update in the in-memory ingest
# queue, consisting of the decoded vector and its application-level
# identifier.
@dataclass
class _Upd:
    vec: np.ndarray
    vid: int


# -----------------------------------------------------------------------------
# Service implementation
# -----------------------------------------------------------------------------
# Implement the gRPC search and ingest interface exposed by the FAISS
# service. The class keeps the search path fast and predictable while
# buffering update traffic and, when configured as a primary node,
# optionally replicating accepted ingest operations to follower replicas.
class FaissDedupServicer(dedup_pb2_grpc.FaissDedupServicer):
  
    # Initialise the service state, including the FAISS execution pool,
    # optional GPU-operation lock, buffered update queue, optional replica
    # fan-out stubs, and rolling latency summaries exposed through the
    # administrative metrics endpoint.
    def __init__(self, mgr: FaissManager):
        self.mgr = mgr
        self._exec = ThreadPoolExecutor(max_workers=FAISS_EXECUTOR_WORKERS)

        # Serialize GPU index ops to avoid allocator asserts and undefined behaviour.
        self._index_lock = asyncio.Lock() if mgr.use_gpu else None

        # Buffered update queue (append-only ingestion)
        self._upd_q: "asyncio.Queue[_Upd]" = asyncio.Queue(maxsize=UPDATE_MAX_QUEUE)

        # PRIMARY-only: outbound channels/stubs for follower replication (IngestBatch fan-out).
        self._replica_channels: List[grpc.aio.Channel] = []
        self._replica_stubs: List[dedup_pb2_grpc.FaissDedupStub] = []
        
        self.lat_search_rpc = _LatencyStats(METRICS_WINDOW)   # wall time per FAISS call
        self.lat_ingest_enq = _LatencyStats(METRICS_WINDOW)   # enqueue time in IngestBatch
        if IS_PRIMARY and REPLICA_FANOUT:
            for addr in REPLICA_FANOUT:
                ch = grpc.aio.insecure_channel(addr, options=_grpc_options())
                self._replica_channels.append(ch)
                self._replica_stubs.append(dedup_pb2_grpc.FaissDedupStub(ch))

    # Execute an asynchronous FAISS operation under the GPU lock when the
    # active index is GPU-backed. This prevents overlapping search and add
    # calls from entering the GPU index concurrently.
    async def _with_lock(self, coro):
        """Execute coro under the GPU lock if needed."""
        if self._index_lock is None:
            return await coro()
        async with self._index_lock:
            return await coro()

    # Execute a FAISS search in the thread pool and return distances,
    # indices, and the wall-clock time observed by the service. The timing
    # includes executor dispatch as well as FAISS runtime and is therefore
    # suitable for service-level latency reporting.
    async def _search(self, vecs: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Runs FAISS search in a threadpool and returns (D, I, wall_ms).
        wall_ms includes executor scheduling + FAISS runtime; this is what callers experience.
        """
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        async def _do():
            return await loop.run_in_executor(self._exec, self.mgr.index_srv.search, vecs, k)

        D, I = await self._with_lock(_do)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.lat_search_rpc.add_ms(dt_ms)
        return D, I, dt_ms

    # Append vectors to the active FAISS index and maintain identifier
    # alignment where necessary. CPU-backed indices store application-level
    # identifiers directly, whereas the GPU flat index updates a separate
    # parallel identifier array after a successful add.
    async def _add(self, vecs: np.ndarray, ids: np.ndarray) -> None:
        """
        Append vectors to index and maintain id-store (GPU) if enabled.

        CPU (IndexIDMap2): add_with_ids(vecs, ids)
        GPU (Flat): add(vecs) and append ids to mgr.ids (aligned by position)
        """
        loop = asyncio.get_running_loop()

        def _do_add():
            if self.mgr.use_gpu:
                self.mgr.index_srv.add(vecs)
            else:
                self.mgr.index_srv.add_with_ids(vecs, ids)

        await self._with_lock(lambda: loop.run_in_executor(self._exec, _do_add))

        # Maintain id store for GPU-flat after successful add.
        if self.mgr.use_gpu:
            if self.mgr.ids is None:
                self.mgr.ids = ids.copy()
            else:
                self.mgr.ids = np.concatenate([self.mgr.ids, ids])

    # Periodically drain the buffered ingest queue and apply queued updates
    # to the index in batches. This separates the append-only update path
    # from the latency-critical search path so that frequent small ingests
    # do not dominate search tail latency.
    async def _flush_updates_loop(self) -> None:
        """
        Background update flusher:
        - wakes every UPDATE_BATCH_MS
        - drains up to UPDATE_BATCH_SIZE updates
        - calls _add() once per flush
        """
        if not ENABLE_UPDATES:
            return

        while True:
            await asyncio.sleep(UPDATE_BATCH_MS / 1000.0)

            vecs = []
            vids = []

            while not self._upd_q.empty() and len(vecs) < UPDATE_BATCH_SIZE:
                u = self._upd_q.get_nowait()
                vecs.append(u.vec)
                vids.append(int(u.vid))

            if not vecs:
                continue

            mat = np.stack(vecs, axis=0).astype(np.float32, copy=False)
            ids = np.asarray(vids, dtype=np.int64)

            try:
                await self._add(mat, ids)
            except Exception:
                # Suppress flush failures so that the benchmark service
                # remains available, while acknowledging that a production
                # system should log, alert, and apply stronger failure
                # handling at this point.
                pass

    # Replicate an accepted ingest payload from the primary node to any
    # configured follower replicas. The method supports both asynchronous
    # and synchronous fan-out modes and reports how many followers accepted
    # the forwarded update.
    async def _fanout_ingest_to_followers(self, accepted_ids: List[int], accepted_embs: List[bytes]) -> Tuple[int, int, str]:
        """
        Replicate an accepted IngestBatch payload to follower replicas.

        Returns:
          (ok_count, total_followers, note_suffix)

        Behavior is controlled by SYNC_REPLICATION / STRICT_REPLICATION.
        """
        if not (IS_PRIMARY and self._replica_stubs):
            return (0, 0, "")

        md = (("x-internal-auth", INTERNAL_AUTH_TOKEN),)
        fwd_req = dedup_pb2.IngestBatchRequest(ids=accepted_ids, embeddings_f32=accepted_embs)

        async def _call_all():
            calls = [
                stub.IngestBatch(fwd_req, timeout=REPLICA_TIMEOUT_S, metadata=md)
                for stub in self._replica_stubs
            ]
            return await asyncio.gather(*calls, return_exceptions=True)

        if not SYNC_REPLICATION:
            # Fire-and-forget replication maximizes throughput; followers may lag briefly.
            asyncio.create_task(_call_all())
            return (0, len(self._replica_stubs), "replication_async")

        res = await _call_all()
        ok = 0
        for r in res:
            if isinstance(r, Exception):
                continue
            if getattr(r, "accepted", 0) == len(accepted_ids):
                ok += 1

        if ok == len(self._replica_stubs):
            return (ok, len(self._replica_stubs), "replication_ok")
        return (ok, len(self._replica_stubs), f"partial_replication:{ok}/{len(self._replica_stubs)}")

    # -------------------------------------------------------------------------
    # RPCs
    # -------------------------------------------------------------------------
    # Handle a single-query nearest-neighbour search. The method validates
    # the embedding payload, executes a k=1 FAISS lookup, maps internal
    # positions to application identifiers when required, and returns both
    # distance values and service-level timing metadata.
    async def Search(self, request, context):
        if self.mgr.ntotal() == 0:
            if not ALLOW_EMPTY_INDEX:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "empty_index")
            return dedup_pb2.SearchResponse(
                query_id=request.query_id,
                nearest_id=-1,
                distance_sq=float("inf"),
                distance_l2=float("inf"),
                is_match=False,
                faiss_ms=0.0,
                faiss_batch_ms=0.0,
                batch_size=0,
                aggregator_wait_ms=0.0,
                note="empty_index",
            )

        try:
            vec = _bytes_to_single(request.embedding_f32)
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        D, I, batch_ms = await self._search(vec, 1)
        dist_sq = float(D[0][0])
        pos = int(I[0][0])

        if pos < 0 or (not math.isfinite(dist_sq)):
            return dedup_pb2.SearchResponse(
                query_id=request.query_id,
                nearest_id=-1,
                distance_sq=float("inf"),
                distance_l2=float("inf"),
                is_match=False,
                faiss_ms=batch_ms,
                faiss_batch_ms=batch_ms,
                batch_size=1,
                aggregator_wait_ms=0.0,
                note="faiss_invalid_result",
            )

        # Map GPU position to application id if available
        nn_id = pos
        if self.mgr.use_gpu and self.mgr.ids is not None:
            nn_id = int(self.mgr.ids[pos]) if 0 <= pos < self.mgr.ids.shape[0] else -1

        dist_l2 = float(math.sqrt(dist_sq))
        return dedup_pb2.SearchResponse(
            query_id=request.query_id,
            nearest_id=nn_id,
            distance_sq=dist_sq,
            distance_l2=dist_l2,
            is_match=(dist_sq <= TAU_SQ),
            faiss_ms=batch_ms,
            faiss_batch_ms=batch_ms,
            batch_size=1,
            aggregator_wait_ms=0.0,
            note="ok",
        )

    # Handle a batch nearest-neighbour search. The method validates request
    # shape, decodes all embeddings into one matrix, performs a single FAISS
    # batch lookup, and returns one SearchResponse per submitted query
    # together with the shared batch timing metadata.
    async def SearchBatch(self, request, context):
        if len(request.query_ids) != len(request.embeddings_f32):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query_ids and embeddings_f32 length mismatch")

        n = len(request.query_ids)
        if n == 0:
            return dedup_pb2.SearchBatchResponse(results=[], faiss_batch_ms=0.0, batch_size=0)

        if self.mgr.ntotal() == 0:
            if not ALLOW_EMPTY_INDEX:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "empty_index")
            results = [
                dedup_pb2.SearchResponse(
                    query_id=qid,
                    nearest_id=-1,
                    distance_sq=float("inf"),
                    distance_l2=float("inf"),
                    is_match=False,
                    faiss_ms=0.0,
                    faiss_batch_ms=0.0,
                    batch_size=n,
                    aggregator_wait_ms=0.0,
                    note="empty_index",
                )
                for qid in request.query_ids
            ]
            return dedup_pb2.SearchBatchResponse(results=results, faiss_batch_ms=0.0, batch_size=n)

        try:
            mat = _bytes_list_to_mat(list(request.embeddings_f32))
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        D, I, batch_ms = await self._search(mat, 1)
        per_query_ms = batch_ms / max(1, n)

        results = []
        for qid, d, i in zip(request.query_ids, D, I):
            dist_sq = float(d[0])
            pos = int(i[0])

            if pos < 0 or (not math.isfinite(dist_sq)):
                results.append(
                    dedup_pb2.SearchResponse(
                        query_id=qid,
                        nearest_id=-1,
                        distance_sq=float("inf"),
                        distance_l2=float("inf"),
                        is_match=False,
                        faiss_ms=per_query_ms,
                        faiss_batch_ms=batch_ms,
                        batch_size=n,
                        aggregator_wait_ms=0.0,
                        note="faiss_invalid_result",
                    )
                )
                continue

            nn_id = pos
            if self.mgr.use_gpu and self.mgr.ids is not None:
                nn_id = int(self.mgr.ids[pos]) if 0 <= pos < self.mgr.ids.shape[0] else -1

            dist_l2 = float(math.sqrt(dist_sq))
            results.append(
                dedup_pb2.SearchResponse(
                    query_id=qid,
                    nearest_id=nn_id,
                    distance_sq=dist_sq,
                    distance_l2=dist_l2,
                    is_match=(dist_sq <= TAU_SQ),
                    faiss_ms=per_query_ms,
                    faiss_batch_ms=batch_ms,
                    batch_size=n,
                    aggregator_wait_ms=0.0,
                    note="ok",
                )
            )

        return dedup_pb2.SearchBatchResponse(results=results, faiss_batch_ms=batch_ms, batch_size=n)

    # Handle append-only batch ingestion of new vectors. The method
    # validates the payload, decodes embeddings once, enqueues accepted
    # updates for later flushing, and optionally replicates the accepted
    # portion of the batch to follower replicas when this node acts as the
    # configured primary.
    async def IngestBatch(self, request, context):
        """
        Append-only ingestion path (not part of the hot search benchmark).

        - Validates payload
        - Enqueues updates into _upd_q
        - Optionally replicates accepted updates to follower replicas (PRIMARY only)

        Note: "accepted" means accepted into the in-memory queue, not necessarily flushed to the index yet.
        """
        if not ENABLE_UPDATES:
            return dedup_pb2.IngestBatchResponse(
                accepted=0,
                queued=int(self._upd_q.qsize()),
                enqueue_ms=0.0,
                note="disabled",
            )

        if len(request.embeddings_f32) != len(request.ids):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "ids and embeddings_f32 length mismatch")

        n = len(request.ids)
        if n == 0:
            return dedup_pb2.IngestBatchResponse(
                accepted=0,
                queued=int(self._upd_q.qsize()),
                enqueue_ms=0.0,
                note="ok",
            )

        t0 = time.perf_counter()

        # Decode once, then enqueue rows. We copy each row to avoid referencing the same backing buffer.
        try:
            mat = _bytes_list_to_mat(list(request.embeddings_f32))
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        accepted = 0
        for row, vid in zip(mat, request.ids):
            try:
                self._upd_q.put_nowait(_Upd(vec=row.copy(), vid=int(vid)))
                accepted += 1
            except asyncio.QueueFull:
                break

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.lat_ingest_enq.add_ms(dt_ms)
        note = "ok" if accepted == n else "overloaded"

        # PRIMARY fan-out replication (optional)
        if IS_PRIMARY and self._replica_stubs and accepted > 0:
            accepted_ids = list(request.ids[:accepted])
            accepted_embs = list(request.embeddings_f32[:accepted])
            ok_cnt, total, rep_note = await self._fanout_ingest_to_followers(accepted_ids, accepted_embs)
            if rep_note:
                note = f"{note};{rep_note}"

            if STRICT_REPLICATION and SYNC_REPLICATION and total > 0 and ok_cnt != total:
                await context.abort(grpc.StatusCode.UNAVAILABLE, f"replication_failed:{ok_cnt}/{total}")

        return dedup_pb2.IngestBatchResponse(
            accepted=accepted,
            queued=int(self._upd_q.qsize()),
            enqueue_ms=dt_ms,
            note=note,
        )



# -----------------------------------------------------------------------------
# Server bootstrap
# -----------------------------------------------------------------------------
# Build the FAISS manager, create the gRPC server, register the service
# implementation, and optionally start the buffered update loop and the
# auxiliary administrative HTTP interface. This function is the operational
# entry point for running the service process.
async def serve(listen: str):
    mgr = build_manager()

    server = grpc.aio.server(
        interceptors=[InternalAuthInterceptor(INTERNAL_AUTH_TOKEN)],
        options=_grpc_options(),
    )
    svc = FaissDedupServicer(mgr)
    dedup_pb2_grpc.add_FaissDedupServicer_to_server(svc, server)

    server.add_insecure_port(listen)
    await server.start()

    if ENABLE_UPDATES:
        asyncio.create_task(svc._flush_updates_loop())

    print(
        "[FAISS gRPC] "
        f"listen={listen} ntotal={mgr.ntotal()} use_gpu={mgr.use_gpu} index_type={INDEX_TYPE} "
        f"workers={FAISS_EXECUTOR_WORKERS} updates={ENABLE_UPDATES} "
        f"primary={IS_PRIMARY} fanout={len(REPLICA_FANOUT)} sync_rep={SYNC_REPLICATION} strict_rep={STRICT_REPLICATION}"
    )
    
    # Expose the policy/metrics endpoints if requested.
    if ADMIN_HTTP_LISTEN:
        _ADMIN_STATE["mgr"] = mgr
        _ADMIN_STATE["svc"] = svc
        _start_admin_http(ADMIN_HTTP_LISTEN)
        print(f"[FAISS ADMIN] http://{ADMIN_HTTP_LISTEN} (auth via X-Internal-Auth)")

    await server.wait_for_termination()

# Parse command-line arguments and launch the asynchronous FAISS gRPC
# service on the configured listening address.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:50051")
    args = ap.parse_args()
    asyncio.run(serve(args.listen))

# Standard command-line entry point.
if __name__ == "__main__":
    main()
