# -*- coding: utf-8 -*-
#
# bench_faiss_service_100k.py
# ------------------------
# Lightweight benchmarking client for the internal FAISS similarity service.
#
# What this script measures:
#   (i) Optional index build time by calling /v1/upsert_batch to add N vectors.
#   (ii) Search throughput and request-level latency using either:
#        - /v1/search       (single-query path), or
#        - /v1/search_batch (batched path; reduces HTTP and JSON overhead).
#
# Important constraints:
#   - Upsert requires the service to be started with ENABLE_UPDATES=1.
#   - For speed, the benchmark uses the binary-embedding fast path (base64 float32 bytes).
#   - This is a client-side benchmark; reported times include client scheduling + HTTP overhead.
#
# Interpretation:
#   - For kernel-only timing, use the server-reported field 'faiss_search_time_ms' returned by
#     the service, which isolates FAISS compute from transport overhead.
#
import argparse
import asyncio
import base64
import os
import random
import time
from statistics import mean
from typing import List, Tuple

import httpx
import numpy as np



# ---- Encoding utilities ---------------------------------------------------
def b64_f32_row(v: np.ndarray) -> str:
    """Encode one float32 embedding as base64 of raw bytes (len=dim*4)."""
    v = np.asarray(v, dtype=np.float32)
    if not v.flags["C_CONTIGUOUS"]:
        v = np.ascontiguousarray(v)
    return base64.b64encode(v.tobytes()).decode("ascii")



# ---- Synthetic vector generation -----------------------------------------
def make_vectors(rng: np.random.Generator, n: int, dim: int, normalize: bool) -> np.ndarray:
    """Generate random vectors (optionally L2-normalized)."""
    x = rng.standard_normal((n, dim), dtype=np.float32)
    if normalize:
        nrm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x = x / nrm
    return x



# ---- Index population (optional) -----------------------------------------
async def upsert_100k(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    nvec: int,
    dim: int,
    batch: int,
    normalize: bool,
    query_pool_size: int,
    seed: int,
) -> List[str]:
    """
    Upsert vectors into FAISS index via /v1/upsert_batch.
    Returns a small pool of vectors (base64 strings) to use as queries later.
    """
    rng = np.random.default_rng(seed)
    headers = {"X-Internal-Auth": token}

    pool_b64: List[str] = []

    t0 = time.perf_counter()
    added_total = 0
    batches = (nvec + batch - 1) // batch

    for bi in range(batches):
        start = bi * batch
        b = min(batch, nvec - start)

        vecs = make_vectors(rng, b, dim, normalize)
        ids = list(range(start, start + b))

        # Build query pool from the first few vectors (keeps memory low).
        if len(pool_b64) < query_pool_size:
            need = min(query_pool_size - len(pool_b64), b)
            for i in range(need):
                pool_b64.append(b64_f32_row(vecs[i]))

        payload = {
            "ids": ids,
            "vectors_b64_f32": [b64_f32_row(vecs[i]) for i in range(b)],
        }

        r = await client.post(base_url + "/v1/upsert_batch", json=payload, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"Upsert failed: {r.status_code} {r.text[:500]}")

        j = r.json()
        added_total += int(j.get("added", 0))

    wall = time.perf_counter() - t0
    print(f"[UPsert] Added={added_total}/{nvec}  batches={batches}  wall={wall:.3f}s  rate={added_total/wall:.1f} vec/s")

    return pool_b64



# ---- Benchmark modes ------------------------------------------------------
# Single-query mode: maximises HTTP overhead and is useful mainly for p95/p99 observations.
async def bench_search_single(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    total_queries: int,
    concurrency: int,
    pool_b64: List[str],
) -> None:
    """
    Benchmark /v1/search (single query).
    """
    headers = {"X-Internal-Auth": token}
    lat_ms: List[float] = []
    kernel_ms: List[float] = []

    per_worker = max(1, total_queries // concurrency)

    async def worker(wid: int):
        for i in range(per_worker):
            v = random.choice(pool_b64)
            payload = {"subject_ref": f"q-{wid}-{i}", "vector_b64_f32": v}
            t0 = time.perf_counter()
            r = await client.post(base_url + "/v1/search", json=payload, headers=headers)
            dt = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                raise RuntimeError(f"Search failed: {r.status_code} {r.text[:500]}")
            lat_ms.append(dt)
            kernel_ms.append(float(r.json().get("faiss_search_time_ms", 0.0)))

    t0 = time.perf_counter()
    await asyncio.gather(*[worker(w) for w in range(concurrency)])
    wall = time.perf_counter() - t0

    lat_sorted = sorted(lat_ms)
    p50 = lat_sorted[int(0.50 * (len(lat_sorted) - 1))]
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
    p99 = lat_sorted[int(0.99 * (len(lat_sorted) - 1))]

    print(f"[SEARCH single] queries={len(lat_ms)} conc={concurrency}")
    print(f"  Throughput: {len(lat_ms)/wall:.1f} qps   Wall: {wall:.3f}s")
    print(f"  Lat(ms): mean={mean(lat_ms):.3f} p50={p50:.3f} p95={p95:.3f} p99={p99:.3f} max={lat_sorted[-1]:.3f}")
    print(f"  FAISS kernel(ms): mean={mean(kernel_ms):.3f} (server-reported)")



# Batch mode: closer to production (gateway → FAISS) behaviour for high request volumes.
async def bench_search_batch(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    total_queries: int,
    concurrency: int,
    batch_q: int,
    pool_b64: List[str],
) -> None:
    """
    Benchmark /v1/search_batch (batched queries).
    More realistic for service use and reduces HTTP overhead.
    """
    headers = {"X-Internal-Auth": token}
    lat_ms: List[float] = []
    kernel_ms_per_vec: List[float] = []

    # Total HTTP requests = total_queries / batch_q (rounded up), then split across workers.
    total_reqs = (total_queries + batch_q - 1) // batch_q
    per_worker = max(1, total_reqs // concurrency)

    async def worker(wid: int):
        for bi in range(per_worker):
            b = batch_q
            subj = [f"bq-{wid}-{bi}-{i}" for i in range(b)]
            vecs = [random.choice(pool_b64) for _ in range(b)]

            payload = {"subject_refs": subj, "vectors_b64_f32": vecs}

            t0 = time.perf_counter()
            r = await client.post(base_url + "/v1/search_batch", json=payload, headers=headers)
            dt = (time.perf_counter() - t0) * 1000.0

            if r.status_code != 200:
                raise RuntimeError(f"SearchBatch failed: {r.status_code} {r.text[:500]}")

            lat_ms.append(dt)

            j = r.json()
            km = float(j.get("faiss_search_time_ms", 0.0))
            kernel_ms_per_vec.append(km / float(b))

    t0 = time.perf_counter()
    await asyncio.gather(*[worker(w) for w in range(concurrency)])
    wall = time.perf_counter() - t0

    # Each request carried batch_q queries.
    effective_queries = len(lat_ms) * batch_q

    lat_sorted = sorted(lat_ms)
    p50 = lat_sorted[int(0.50 * (len(lat_sorted) - 1))]
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
    p99 = lat_sorted[int(0.99 * (len(lat_sorted) - 1))]

    print(f"[SEARCH batch] effective_queries≈{effective_queries}  reqs={len(lat_ms)}  batch_q={batch_q}  conc={concurrency}")
    print(f"  Throughput: {effective_queries/wall:.1f} qps   Wall: {wall:.3f}s")
    print(f"  Req Lat(ms): mean={mean(lat_ms):.3f} p50={p50:.3f} p95={p95:.3f} p99={p99:.3f} max={lat_sorted[-1]:.3f}")
    print(f"  FAISS kernel per-vector(ms): mean={mean(kernel_ms_per_vec):.4f} (server-reported)")



# ---- CLI / orchestration --------------------------------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="FAISS service base URL, e.g. http://127.0.0.1:9010")
    ap.add_argument("--token", required=True, help="INTERNAL_AUTH_TOKEN")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--nvec", type=int, default=100_000, help="Index size to build via upsert")
    ap.add_argument("--upsert-batch", type=int, default=1000)
    ap.add_argument("--no-upsert", action="store_true", help="Skip upsert (assumes service already has index)")
    ap.add_argument("--queries", type=int, default=50_000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--mode", choices=["single", "batch"], default="batch")
    ap.add_argument("--batch-q", type=int, default=64, help="Queries per /v1/search_batch request")
    ap.add_argument("--pool", type=int, default=2048, help="How many vectors to keep as query pool")
    ap.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize generated vectors")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base_url = args.url.rstrip("/")
    normalize = not args.no_normalize

    # Keep payload sizes reasonable and avoid timeouts.
    # Connection pooling limits are set proportional to concurrency to avoid client-side throttling.
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency * 2)

    async with httpx.AsyncClient(timeout=60.0, limits=limits) as client:
        # Quick health check
        r = await client.get(base_url + "/v1/health")
        if r.status_code != 200:
            raise RuntimeError(f"Service not healthy: {r.status_code} {r.text[:200]}")

        # Build query pool
        pool_b64: List[str] = []

        if not args.no_upsert:
            print("[INFO] Upserting vectors. Ensure the service was started with ENABLE_UPDATES=1.")
            pool_b64 = await upsert_100k(
                client=client,
                base_url=base_url,
                token=args.token,
                nvec=args.nvec,
                dim=args.dim,
                batch=args.upsert_batch,
                normalize=normalize,
                query_pool_size=args.pool,
                seed=args.seed,
            )
        else:
            # If skipping upsert, generate a query pool locally (still good for benchmarking latency).
            rng = np.random.default_rng(args.seed)
            vecs = make_vectors(rng, min(args.pool, 4096), args.dim, normalize)
            pool_b64 = [b64_f32_row(vecs[i]) for i in range(vecs.shape[0])]

        if args.mode == "single":
            await bench_search_single(
                client=client,
                base_url=base_url,
                token=args.token,
                total_queries=args.queries,
                concurrency=args.concurrency,
                pool_b64=pool_b64,
            )
        else:
            await bench_search_batch(
                client=client,
                base_url=base_url,
                token=args.token,
                total_queries=args.queries,
                concurrency=args.concurrency,
                batch_q=args.batch_q,
                pool_b64=pool_b64,
            )


if __name__ == "__main__":
    asyncio.run(main())
