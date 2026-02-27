#!/usr/bin/env python3
"""
faiss_grpc_server.py (v3.2)
===========================

What this service is
--------------------
A gRPC FAISS similarity microservice that supports:

  1) Search()       : single-vector nearest-neighbour lookup (k=1)
  2) SearchBatch()  : batched nearest-neighbour lookup (k=1)  [hot path for throughput]
  3) IngestBatch()  : append-only index updates, buffered + flushed asynchronously

This v3.2 update adds *optional primary→follower replication* for IngestBatch so that
multiple FAISS replicas can behave like one logical "master index" (e.g., "already-voted"
set) while still scaling read/search throughput.

Key correctness/scalability idea
--------------------------------
- There is ONE logical index (the global set of embeddings to compare against).
- We run multiple PHYSICAL replicas of that index for capacity and availability.
- To keep replicas consistent, every accepted IngestBatch update must be replicated
  to followers (either by the upstream coordinator, or by the PRIMARY fan-out in this file).

This file implements the *minimum* server-side support for that: PRIMARY fan-out.

Compatibility note (important)
------------------------------
To use IngestBatch, your protobuf service definition must include the IngestBatch RPC and
message types. If your current dedup.proto only has Search/SearchBatch, add IngestBatch and
re-run gen_proto.sh to regenerate stubs.

Design principles (reviewer-friendly)
-------------------------------------
- Embeddings are transported as raw float32 bytes (512-D → 2048 bytes) to avoid JSON/base64 overhead.
- GPU FAISS operations are serialized with an asyncio.Lock to avoid FAISS GPU allocator asserts.
- Ingest is decoupled from Search via an in-memory queue and periodic flush (so updates do not
  dominate search tail latency).
- Internal auth is fail-closed using gRPC metadata header: x-internal-auth

Environment variables (core)
----------------------------
INTERNAL_AUTH_TOKEN   (required) shared secret; client must send metadata x-internal-auth
EMB_DIM               default 512
TAU_L2                default 1.150 (L2 threshold; compare squared distance to TAU_L2^2)

USE_GPU               1/0 (default 1)
GPU_ID                default 0
INDEX_TYPE            flat|hnsw (default flat). GPU supported only for flat.

BOOTSTRAP_MODE        empty|random (default empty)
RANDOM_N              number of synthetic vectors (default 1_000_000 when random)
RANDOM_CHUNK          chunk size for bootstrap (default 100000)
ALLOW_EMPTY_INDEX     1/0 (default 1)

FAISS_OMP_THREADS     (CPU only) OpenMP threads
FAISS_EXECUTOR_WORKERS threadpool size for FAISS native calls (CPU default 64, GPU default 1)
FAISS_GPU_TEMP_BYTES  optional temp memory for StandardGpuResources (bytes)

Updates (append-only)
---------------------
ENABLE_UPDATES         1/0 (default 1)
UPDATE_BATCH_SIZE      default 4096
UPDATE_BATCH_MS        default 100
UPDATE_MAX_QUEUE       default 200000

Replica-consistency (PRIMARY fan-out for IngestBatch)
-----------------------------------------------------
IS_PRIMARY=1|0            default 0
REPLICA_FANOUT            comma-separated follower addrs, e.g. "10.0.0.2:50051,10.0.0.3:50051"
SYNC_REPLICATION=1|0      default 0 (async by default)
STRICT_REPLICATION=1|0    default 0 (only applies when SYNC_REPLICATION=1)
REPLICA_TIMEOUT_S         default 2.0

Operational guidance
--------------------
- For a real election system, treat FAISS replicas as read replicas and either:
    (A) upstream coordinator broadcasts IngestBatch to all replicas, or
    (B) send ingest to PRIMARY only and let PRIMARY fan-out to followers (this file).
- If STRICT_REPLICATION=1, a partial fan-out fails the ingest (caller must retry or remove
  lagging replicas from the read pool).

Run
---
ulimit -n 200000
export INTERNAL_AUTH_TOKEN=devtoken
export USE_GPU=1
export BOOTSTRAP_MODE=random
export RANDOM_N=1000000

python faiss_grpc_server.py --listen 0.0.0.0:50051
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


# -----------------------------------------------------------------------------
# Make ./gen importable (avoids requiring users to export PYTHONPATH explicitly)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# gRPC options
# -----------------------------------------------------------------------------
def _grpc_options():
    """
    Conservative keepalive to avoid GOAWAY ENHANCE_YOUR_CALM for too many pings.
    For high-throughput workloads, there is constant data flow and keepalive is rarely needed.
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


# -----------------------------------------------------------------------------
# Auth interceptor
# -----------------------------------------------------------------------------
class InternalAuthInterceptor(grpc.aio.ServerInterceptor):
    """Fail-closed internal auth via metadata x-internal-auth."""

    def __init__(self, expected_token: str):
        self.expected = expected_token

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
@dataclass
class FaissManager:
    index_srv: faiss.Index
    use_gpu: bool
    # For GPU-flat, ids[pos] gives application-level id for vector at position pos.
    ids: Optional[np.ndarray]

    def ntotal(self) -> int:
        return int(self.index_srv.ntotal)


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


def _bytes_to_single(b: bytes) -> np.ndarray:
    if len(b) != _EXPECT_BYTES:
        raise ValueError(f"embedding_f32 must be {_EXPECT_BYTES} bytes, got {len(b)}")
    return np.frombuffer(b, dtype=np.float32).reshape(1, EMB_DIM)


# -----------------------------------------------------------------------------
# Update queue entries
# -----------------------------------------------------------------------------
@dataclass
class _Upd:
    vec: np.ndarray
    vid: int


# -----------------------------------------------------------------------------
# Service implementation
# -----------------------------------------------------------------------------
class FaissDedupServicer(dedup_pb2_grpc.FaissDedupServicer):
    """
    Search/SearchBatch are the latency-critical APIs used in benchmarks.
    IngestBatch is buffered and flushed periodically to avoid inflating search tail latency.

    GPU safety:
      - FAISS GPU indices are not safe under concurrent search/add calls.
      - We serialize all GPU index operations using an asyncio.Lock.
    """

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
        if IS_PRIMARY and REPLICA_FANOUT:
            for addr in REPLICA_FANOUT:
                ch = grpc.aio.insecure_channel(addr, options=_grpc_options())
                self._replica_channels.append(ch)
                self._replica_stubs.append(dedup_pb2_grpc.FaissDedupStub(ch))

    async def _with_lock(self, coro):
        """Execute coro under the GPU lock if needed."""
        if self._index_lock is None:
            return await coro()
        async with self._index_lock:
            return await coro()

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
        return D, I, dt_ms

    async def _add(self, vecs: np.ndarray, ids: np.ndarray) -> None:
        """
        Append vectors to index and maintain id-store (GPU) if enabled.

        CPU (IndexIDMap2): add_with_ids(vecs, ids)
        GPU (Flat): add(vecs) and append ids to mgr.ids (aligned by position)
        """
        loop = asyncio.get_running_loop()

        async def _do_add():
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
                # For benchmark harnesses, swallow errors to keep service alive.
                # Production systems should log + alert + trip a circuit breaker.
                pass

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

    await server.wait_for_termination()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:50051")
    args = ap.parse_args()
    asyncio.run(serve(args.listen))


if __name__ == "__main__":
    main()
