#!/usr/bin/env python3
"""
faiss_grpc_server.py (v3.1)
---------------------------
FAISS gRPC server with:
- Search / SearchBatch for online similarity queries
- IngestBatch for index updates (queued + flushed asynchronously)

This design mirrors the intent of your older FastAPI+FAISS service, where updates are
buffered in an async queue and flushed periodically in a background task.  (See faiss_ms (1).py.)  fileciteturn4file0

Why you saw RPC ~3,000 ms while FAISS ~1–6 ms
---------------------------------------------
Your benchmark summaries show FAISS compute is fast, but RPC latency ~3 s and many DEADLINE_EXCEEDED.
That is queueing + a short per-RPC deadline, typically caused by:
- server instability (GPU crash under concurrent searches), and/or
- offered load exceeding service capacity (requests sit in queue until deadline triggers).

This version focuses on STABILITY first:
- FAISS GPU search/add are serialized using an asyncio.Lock (GPU indices are not safe under concurrent calls).
- For GPU, default FAISS_EXECUTOR_WORKERS=1.

Then it provides SAFE, BACKGROUND updates:
- IngestBatch enqueues embeddings; a background flusher calls add() in batches.

Environment variables (key)
---------------------------
INTERNAL_AUTH_TOKEN (required)
EMB_DIM=512
TAU_L2=1.150

USE_GPU=1/0 (default 1)
GPU_ID=0
INDEX_TYPE=flat|hnsw (default flat). GPU supports flat only.

BOOTSTRAP_MODE=empty|random
RANDOM_N (default 1_000_000 when random and RANDOM_N=0)
RANDOM_CHUNK=100000

LOAD_INDEX_PATH optional

ENABLE_UPDATES=1/0 (default 1)
UPDATE_BATCH_SIZE=4096
UPDATE_BATCH_MS=100
UPDATE_MAX_QUEUE=200000

FAISS_OMP_THREADS (CPU)
FAISS_EXECUTOR_WORKERS (CPU default 64; GPU default 1)

ID_STORE=1/0 (default 1)  # maintain app-level ids even on GPU-flat

GRPC_MAX_MB=64
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
from typing import List, Optional

import numpy as np
import faiss
import grpc


ROOT = Path(__file__).resolve().parent
GEN = ROOT / "gen"
if GEN.exists():
    sys.path.insert(0, str(GEN))
    sys.path.insert(0, str(ROOT))

try:
    import dedup_pb2, dedup_pb2_grpc  # type: ignore
except Exception:
    from gen import dedup_pb2, dedup_pb2_grpc  # type: ignore


EMB_DIM = int(os.getenv("EMB_DIM", "512"))
TAU_L2 = float(os.getenv("TAU_L2", "1.150"))
TAU_SQ = TAU_L2 * TAU_L2


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# This server exposes a minimal similarity API around a FAISS index.
# - Search / SearchBatch: read path (nearest-neighbour query).
# - IngestBatch: update path (optional; enqueues additions and flushes asynchronously).
#
# Key implementation assumptions:
# - When USE_GPU=1, FAISS GPU indices are treated as not thread-safe for concurrent search/add.
#   Therefore, GPU operations are serialized via an asyncio.Lock to prevent instability.
# - For GPU, FAISS_EXECUTOR_WORKERS defaults to 1 (to avoid hidden concurrency in the thread pool).
# - If ID_STORE=1, we maintain an application-level id array for GPU-flat indices because
#   GPU-flat indices do not naturally preserve application ids the same way as IndexIDMap2.
# -----------------------------------------------------------------------------
INTERNAL_AUTH_TOKEN = os.getenv("INTERNAL_AUTH_TOKEN", "").strip()
if not INTERNAL_AUTH_TOKEN:
    raise RuntimeError("INTERNAL_AUTH_TOKEN must be set (fail-closed).")

INDEX_TYPE = os.getenv("INDEX_TYPE", "flat").lower()
USE_GPU = os.getenv("USE_GPU", "1") == "1"
GPU_ID = int(os.getenv("GPU_ID", "0"))

BOOTSTRAP_MODE = os.getenv("BOOTSTRAP_MODE", "empty").lower()
RANDOM_N = int(os.getenv("RANDOM_N", "0"))
RANDOM_CHUNK = int(os.getenv("RANDOM_CHUNK", "100000"))
LOAD_INDEX_PATH = os.getenv("LOAD_INDEX_PATH", "").strip()

ALLOW_EMPTY_INDEX = os.getenv("ALLOW_EMPTY_INDEX", "1") == "1"

GRPC_MAX_MB = int(os.getenv("GRPC_MAX_MB", "64"))

FAISS_OMP_THREADS = os.getenv("FAISS_OMP_THREADS", "").strip()
if FAISS_OMP_THREADS:
    try:
        faiss.omp_set_num_threads(int(FAISS_OMP_THREADS))
    except Exception:
        pass

_default_workers = 1 if USE_GPU else 64
FAISS_EXECUTOR_WORKERS = int(os.getenv("FAISS_EXECUTOR_WORKERS", str(_default_workers)))

ENABLE_UPDATES = os.getenv("ENABLE_UPDATES", "1") == "1"
UPDATE_BATCH_SIZE = int(os.getenv("UPDATE_BATCH_SIZE", "4096"))
UPDATE_BATCH_MS = int(os.getenv("UPDATE_BATCH_MS", "100"))
UPDATE_MAX_QUEUE = int(os.getenv("UPDATE_MAX_QUEUE", "200000"))

ID_STORE = os.getenv("ID_STORE", "1") == "1"


class InternalAuthInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, expected_token: str):
        self.expected = expected_token

    async def intercept_service(self, continuation, handler_call_details):
        md = dict(handler_call_details.invocation_metadata or [])
        tok = md.get("x-internal-auth", "")
        if tok != self.expected:
            async def abort_behavior(request, context):
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "forbidden")
            return grpc.aio.unary_unary_rpc_method_handler(abort_behavior)
        return await continuation(handler_call_details)


def _make_cpu_index() -> faiss.Index:
    if INDEX_TYPE == "hnsw":
        m = int(os.getenv("HNSW_M", "32"))
        efS = int(os.getenv("HNSW_EF_SEARCH", "64"))
        efC = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
        base = faiss.IndexHNSWFlat(EMB_DIM, m)
        base.hnsw.efSearch = efS
        base.hnsw.efConstruction = efC
        return faiss.IndexIDMap2(base)
    return faiss.IndexIDMap2(faiss.IndexFlatL2(EMB_DIM))


def _make_gpu_index_flat_fp16() -> Optional[faiss.Index]:
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
        _ = idx.search(np.zeros((1, EMB_DIM), dtype=np.float32), 1)
        return idx
    except Exception:
        return None


def _bootstrap_random(n: int, start: int) -> np.ndarray:
    rng = np.random.default_rng(seed=42 + start)
    return rng.standard_normal((n, EMB_DIM)).astype(np.float32, copy=False)


@dataclass
class FaissManager:
    index_srv: faiss.Index
    use_gpu: bool
    ids: Optional[np.ndarray]

    def ntotal(self) -> int:
        return int(self.index_srv.ntotal)



# -----------------------------------------------------------------------------
# Index construction and preload
# -----------------------------------------------------------------------------
# Startup modes:
# 1) LOAD_INDEX_PATH: load a serialized FAISS index from disk (fast start, reproducible).
# 2) BOOTSTRAP_MODE=random: generate synthetic vectors to emulate large gallery size.
# 3) default: start empty (ALLOW_EMPTY_INDEX controls whether queries are rejected).
# -----------------------------------------------------------------------------
def build_manager() -> FaissManager:
    ids = None

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

    gpu = _make_gpu_index_flat_fp16()
    if gpu is not None:
        idx = gpu
        use_gpu = True
        if ID_STORE:
            ids = np.empty((0,), dtype=np.int64)
    else:
        idx = _make_cpu_index()
        use_gpu = False

    if BOOTSTRAP_MODE == "random":
        n_total = RANDOM_N if RANDOM_N > 0 else 1_000_000
        start = 0
        while start < n_total:
            n = min(RANDOM_CHUNK, n_total - start)
            vecs = _bootstrap_random(n, start)
            if use_gpu:
                idx.add(vecs)
                if ids is not None:
                    ids = np.concatenate([ids, np.arange(start, start + n, dtype=np.int64)])
            else:
                idx.add_with_ids(vecs, np.arange(start, start + n, dtype=np.int64))
            start += n

    return FaissManager(index_srv=idx, use_gpu=use_gpu, ids=ids)


_EXPECT = None
def _exp_bytes() -> int:
    global _EXPECT
    if _EXPECT is None:
        _EXPECT = EMB_DIM * 4
    return _EXPECT


def _bytes_to_single(b: bytes) -> np.ndarray:
    if len(b) != _exp_bytes():
        raise ValueError(f"embedding_f32 must be {_exp_bytes()} bytes, got {len(b)}")
    return np.frombuffer(b, dtype=np.float32).reshape(1, -1)


def _bytes_list_to_mat(bs: List[bytes]) -> np.ndarray:
    exp = _exp_bytes()
    for b in bs:
        if len(b) != exp:
            raise ValueError(f"embedding_f32 must be {exp} bytes, got {len(b)}")
    # Fast path: concatenate byte payloads into one contiguous buffer, then view as float32 matrix.
    # This avoids per-vector allocations and reduces Python overhead for large batches.
    blob = b"".join(bs)
    return np.frombuffer(blob, dtype=np.float32).reshape(len(bs), EMB_DIM)


@dataclass
class _Upd:
    vec: np.ndarray
    vid: int



# -----------------------------------------------------------------------------
# Service implementation
# -----------------------------------------------------------------------------
# Search and SearchBatch are latency-critical and measured in benchmarks.
# IngestBatch is decoupled from the search path via an in-memory queue flushed periodically.
# -----------------------------------------------------------------------------
class FaissDedupServicer(dedup_pb2_grpc.FaissDedupServicer):
    def __init__(self, mgr: FaissManager):
        self.mgr = mgr
        self._exec = ThreadPoolExecutor(max_workers=FAISS_EXECUTOR_WORKERS)
        self._index_lock = asyncio.Lock() if mgr.use_gpu else None
        self._upd_q: "asyncio.Queue[_Upd]" = asyncio.Queue(maxsize=UPDATE_MAX_QUEUE)

    async def _with_lock(self, coro):
        if self._index_lock is None:
            return await coro()
        async with self._index_lock:
            return await coro()

    async def _search(self, vecs: np.ndarray, k: int = 1):
        loop = asyncio.get_running_loop()
        # Wall-clock time includes scheduling into the executor and the FAISS search itself.
        # This is the 'FAISS service time' reported to clients.
        t0 = time.perf_counter()

        async def _do():
            return await loop.run_in_executor(self._exec, self.mgr.index_srv.search, vecs, k)

        D, I = await self._with_lock(_do)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return D, I, dt_ms

    async def _add(self, vecs: np.ndarray, ids: Optional[np.ndarray]):
        loop = asyncio.get_running_loop()

        async def _do_add():
            if self.mgr.use_gpu:
                self.mgr.index_srv.add(vecs)
            else:
                if ids is None:
                    self.mgr.index_srv.add(vecs)
                else:
                    self.mgr.index_srv.add_with_ids(vecs, ids)

        await self._with_lock(lambda: loop.run_in_executor(self._exec, _do_add))


    # Update flusher: periodically drains queued updates and adds them to the index in batches.
    # This keeps Search latency predictable; update cost is amortized and can be rate-limited.
    async def _flush_updates_loop(self):
        if not ENABLE_UPDATES:
            return
        while True:
            await asyncio.sleep(UPDATE_BATCH_MS / 1000.0)

            vecs = []
            ids = []
            while not self._upd_q.empty() and len(vecs) < UPDATE_BATCH_SIZE:
                u = self._upd_q.get_nowait()
                vecs.append(u.vec)
                ids.append(int(u.vid))

            if not vecs:
                continue

            mat = np.stack(vecs, axis=0).astype(np.float32, copy=False)
            id_arr = np.asarray(ids, dtype=np.int64)

            try:
                await self._add(mat, id_arr)
                if self.mgr.use_gpu and self.mgr.ids is not None:
                    self.mgr.ids = np.concatenate([self.mgr.ids, id_arr])
            except Exception:
                pass

    async def Search(self, request, context):
        if self.mgr.ntotal() == 0:
            if not ALLOW_EMPTY_INDEX:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "empty_index")
            return dedup_pb2.SearchResponse(query_id=request.query_id, nearest_id=-1,
                                           distance_sq=float("inf"), distance_l2=float("inf"),
                                           is_match=False, faiss_ms=0.0, faiss_batch_ms=0.0,
                                           batch_size=0, note="empty_index")
        try:
            vec = _bytes_to_single(request.embedding_f32)
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        D, I, batch_ms = await self._search(vec, 1)
        dist_sq = float(D[0][0]); pos = int(I[0][0])

        if pos < 0 or (not math.isfinite(dist_sq)):
            return dedup_pb2.SearchResponse(query_id=request.query_id, nearest_id=-1,
                                           distance_sq=float("inf"), distance_l2=float("inf"),
                                           is_match=False, faiss_ms=batch_ms, faiss_batch_ms=batch_ms,
                                           batch_size=1, note="faiss_invalid_result")

        nn_id = pos
        if self.mgr.use_gpu and self.mgr.ids is not None:
            nn_id = int(self.mgr.ids[pos]) if 0 <= pos < self.mgr.ids.shape[0] else -1

        dist_l2 = float(math.sqrt(dist_sq))
        return dedup_pb2.SearchResponse(query_id=request.query_id, nearest_id=nn_id,
                                       distance_sq=dist_sq, distance_l2=dist_l2,
                                       is_match=(dist_sq <= TAU_SQ),
                                       faiss_ms=batch_ms, faiss_batch_ms=batch_ms,
                                       batch_size=1, note="ok")

    async def SearchBatch(self, request, context):
        if len(request.query_ids) != len(request.embeddings_f32):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query_ids and embeddings_f32 length mismatch")

        n = len(request.query_ids)
        if n == 0:
            return dedup_pb2.SearchBatchResponse(results=[], faiss_batch_ms=0.0, batch_size=0)

        if self.mgr.ntotal() == 0:
            if not ALLOW_EMPTY_INDEX:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "empty_index")
            results = [dedup_pb2.SearchResponse(query_id=qid, nearest_id=-1,
                                               distance_sq=float("inf"), distance_l2=float("inf"),
                                               is_match=False, faiss_ms=0.0, faiss_batch_ms=0.0,
                                               batch_size=n, note="empty_index")
                       for qid in request.query_ids]
            return dedup_pb2.SearchBatchResponse(results=results, faiss_batch_ms=0.0, batch_size=n)

        try:
            mat = _bytes_list_to_mat(list(request.embeddings_f32))
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        D, I, batch_ms = await self._search(mat, 1)
        per_query_ms = batch_ms / max(1, n)

        results = []
        for qid, d, i in zip(request.query_ids, D, I):
            dist_sq = float(d[0]); pos = int(i[0])
            if pos < 0 or (not math.isfinite(dist_sq)):
                results.append(dedup_pb2.SearchResponse(query_id=qid, nearest_id=-1,
                                                      distance_sq=float("inf"), distance_l2=float("inf"),
                                                      is_match=False, faiss_ms=per_query_ms,
                                                      faiss_batch_ms=batch_ms, batch_size=n,
                                                      note="faiss_invalid_result"))
                continue

            nn_id = pos
            if self.mgr.use_gpu and self.mgr.ids is not None:
                nn_id = int(self.mgr.ids[pos]) if 0 <= pos < self.mgr.ids.shape[0] else -1

            dist_l2 = float(math.sqrt(dist_sq))
            results.append(dedup_pb2.SearchResponse(query_id=qid, nearest_id=nn_id,
                                                  distance_sq=dist_sq, distance_l2=dist_l2,
                                                  is_match=(dist_sq <= TAU_SQ),
                                                  faiss_ms=per_query_ms, faiss_batch_ms=batch_ms,
                                                  batch_size=n, note="ok"))

        return dedup_pb2.SearchBatchResponse(results=results, faiss_batch_ms=batch_ms, batch_size=n)

    async def IngestBatch(self, request, context):
        if not ENABLE_UPDATES:
            return dedup_pb2.IngestBatchResponse(accepted=0, queued=int(self._upd_q.qsize()), enqueue_ms=0.0, note="disabled")

        if len(request.embeddings_f32) != len(request.ids):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "ids and embeddings_f32 length mismatch")

        n = len(request.ids)
        if n == 0:
            return dedup_pb2.IngestBatchResponse(accepted=0, queued=int(self._upd_q.qsize()), enqueue_ms=0.0, note="ok")

        t0 = time.perf_counter()

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
        return dedup_pb2.IngestBatchResponse(accepted=accepted, queued=int(self._upd_q.qsize()), enqueue_ms=dt_ms, note=note)


def _grpc_options():
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

    print(f"[FAISS gRPC] listen={listen} ntotal={mgr.ntotal()} use_gpu={mgr.use_gpu} index_type={INDEX_TYPE} workers={FAISS_EXECUTOR_WORKERS} updates={ENABLE_UPDATES}")
    await server.wait_for_termination()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:50051")
    args = ap.parse_args()
    asyncio.run(serve(args.listen))


if __name__ == "__main__":
    main()
