"""
faiss_bench.py
──────────────
Tiny FastAPI + Faiss micro-service for benchmark runs only.

• Builds a 1 000 000-vector Flat index (FP-16) directly on GPU each start-up
• Optional async batched inserts (disable with ENABLE_UPDATES=0)
• Never touches disk: no snapshots, no restores, no reserve() call
"""

from __future__ import annotations
import os, asyncio
from typing import List

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# ───────────── Config (all via env vars if desired) ───────────── #
EMB_DIM  = int(os.getenv("EMB_DIM", 512))
GPU_ID   = int(os.getenv("GPU_ID", 0))
ENABLE_UPDATES    = os.getenv("ENABLE_UPDATES", "1") == "1"   # 0 → search-only
BATCH_UPDATE_SIZE = int(os.getenv("BATCH_UPDATE_SIZE", 4096))
BATCH_UPDATE_MS   = int(os.getenv("BATCH_UPDATE_MS", 100))
K_NEIGHBOURS      = 1

# ───────────── Build GPU index once per run ───────────── #
def build_index() -> faiss.IndexIDMap2:
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device     = GPU_ID
    cfg.useFloat16 = True                        # half-precision storage
    base = faiss.GpuIndexFlatL2(res, EMB_DIM, cfg)

    idx = faiss.IndexIDMap2(base)

    rng  = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((1_000_000, EMB_DIM), dtype=np.float32)
    ids  = np.arange(1_000_000, dtype=np.int64)
    idx.add_with_ids(vecs, ids)
    print(f"[Init] GPU Flat-FP16 index ready – ntotal {idx.ntotal:,}")
    return idx

# ───────────── FastAPI models ───────────── #
class QueryVector(BaseModel):
    voter_id: int
    vector:   List[float]

class QueryBatch(BaseModel):
    voter_ids: List[int]
    vectors:   List[List[float]]

# ───────────── FastAPI app ───────────── #
app = FastAPI()
faiss_index = build_index()
update_q: asyncio.Queue[tuple[np.ndarray, int]] = asyncio.Queue()

# warm CUDA kernels
_ = faiss_index.search(np.zeros((1, EMB_DIM), dtype="float32"), 1)

# ───── Background flusher (only if updates enabled) ───── #
async def flush_updates():
    while True:
        await asyncio.sleep(BATCH_UPDATE_MS / 1000)
        vecs, ids = [], []
        while not update_q.empty() and len(vecs) < BATCH_UPDATE_SIZE:
            v, vid = update_q.get_nowait()
            vecs.append(v)
            ids.append(np.int64(vid))
        if vecs:
            faiss_index.add_with_ids(np.vstack(vecs).astype("float32"),
                                     np.asarray(ids, dtype="int64"))

@app.on_event("startup")
async def _startup():
    if ENABLE_UPDATES:
        asyncio.create_task(flush_updates())

# ───────────── Endpoints ───────────── #
@app.post("/search")
async def search(q: QueryVector):
    if len(q.vector) != EMB_DIM:
        raise HTTPException(400, f"vector must be length {EMB_DIM}")
    vec = np.asarray(q.vector, dtype="float32").reshape(1, -1)

    # Measure FAISS search time
    t0 = time.time()
    D, I = faiss_index.search(vec, K_NEIGHBOURS)

    dist_sq = float(D[0][0])
    dist_l2 = float(dist_sq ** 0.5)

    t1 = time.time()
    search_time_ms = (t1 - t0) * 1000

    # Measure FAISS update queue time (if enabled)
    update_time_ms = 0
    if ENABLE_UPDATES:
        t2 = time.time()
        await update_q.put((vec, q.voter_id))
        t3 = time.time()
        update_time_ms = (t3 - t2) * 1000

    return {
        "nearest_id": int(I[0][0]),
        "distance": dist_sq,               # back-compat: squared L2
        "distance_sq": dist_sq,            # explicit squared L2
        "distance_l2": dist_l2,            # true L2 (sqrt)
        "faiss_search_time_ms": round(search_time_ms, 3),
        "faiss_index_update_time_ms": round(update_time_ms, 3)
    }


@app.post("/search_batch")
async def search_batch(b: QueryBatch):
    if len(b.vectors) != len(b.voter_ids):
        raise HTTPException(400, "length mismatch")
    arr = np.asarray(b.vectors, dtype="float32")
    if arr.shape[1] != EMB_DIM:
        raise HTTPException(400, f"each vector length {EMB_DIM}")
    D, I = faiss_index.search(arr, K_NEIGHBOURS)
    if ENABLE_UPDATES:
        for v, vid in zip(arr, b.voter_ids):
            await update_q.put((v.reshape(1, -1), vid))
    return [{"voter_id": vid, "nearest_id": int(i[0]), "distance": float(d[0]),            # squared L2 (back-compat)
    "distance_sq": float(d[0]),
    "distance_l2": float(float(d[0]) ** 0.5)}
            for vid, i, d in zip(b.voter_ids, I, D)]

@app.get("/ping")
async def ping():
    return {"status": "ok", "ntotal": faiss_index.ntotal}