#!/usr/bin/env python3
"""
faiss_grpc_bench_optimized.py
-----------------------------
Synthetic embeddings and validity of results
--------------------------------------------
This benchmark generates synthetic float32 embeddings for throughput/latency measurement of the FAISS
serving tier. This is appropriate when the research question is service capacity (RPC overhead, queueing,
microbatching effectiveness, GPU/CPU utilisation), rather than biometric accuracy. If distributional
effects (e.g., distance concentration, thresholding behaviour) are under study, embeddings should be
derived from the same face model used in the full pipeline.

Deadlines and overload behaviour
--------------------------------
Per-RPC timeouts (deadlines) are treated as first-class signals of overload or network impairment.
A “fail-fast” design (explicit deadlines + bounded inflight) is preferable to unbounded queue growth,
because it stabilises p95/p99 and yields interpretable capacity curves for reviewers.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import grpc

# The benchmark is designed to run from a cloned folder without requiring the user to export PYTHONPATH.
# We therefore attempt to load generated protobuf stubs from ./gen (local, reproducible setup) and fall
# back to 'from gen import ...' if the user kept package-style imports.
# ------------------------- Import stubs without PYTHONPATH pain ------------------------- #
ROOT = Path(__file__).resolve().parent
GEN = ROOT / "gen"
if GEN.exists():
    # Make both ./gen and script dir importable
    sys.path.insert(0, str(GEN))
    sys.path.insert(0, str(ROOT))

try:
    import dedup_pb2, dedup_pb2_grpc  # type: ignore
except Exception:  # fallback if user kept "gen." package imports
    from gen import dedup_pb2, dedup_pb2_grpc  # type: ignore


# ------------------------- Config ------------------------- #
_RNG = np.random.default_rng(123)  # optional seed for reproducibility

INTERNAL_AUTH_TOKEN = os.getenv("INTERNAL_AUTH_TOKEN", "").strip()
if not INTERNAL_AUTH_TOKEN:
    raise RuntimeError("INTERNAL_AUTH_TOKEN must be set (fail-closed).")

EMB_DIM = int(os.getenv("EMB_DIM", "512"))

# gRPC message limits: direct-batch returns many results; keep generous headroom.
GRPC_MAX_MB = int(os.getenv("GRPC_MAX_MB", "64"))


# ------------------------- Stats helpers ------------------------- #

def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    k = max(0, min(len(ys) - 1, k))
    return ys[k]


def _stat(vals: List[float]) -> dict:
    return {
        "count": len(vals),
        "p50": _percentile(vals, 50),
        "p95": _percentile(vals, 95),
        "p99": _percentile(vals, 99),
        "mean": statistics.mean(vals) if vals else float("nan"),
        "max": max(vals) if vals else float("nan"),
    }

# Embedding wire format contract:
# - Raw little-endian float32 bytes, length = EMB_DIM * 4.
# - This avoids JSON float lists (large payload, expensive parsing) and keeps transport overhead small.
def _rand_emb_bytes() -> bytes:
    v = _RNG.standard_normal((EMB_DIM,), dtype=np.float32)
    return v.tobytes()


# ------------------------- Benchmark modes ------------------------- #

class Sample:
    __slots__ = ("ok", "rpc_ms", "faiss_ms", "agg_wait_ms", "note")
    def __init__(self, ok: bool, rpc_ms: float, faiss_ms: float, agg_wait_ms: float, note: str):
        self.ok = ok
        self.rpc_ms = rpc_ms
        self.faiss_ms = faiss_ms
        self.agg_wait_ms = agg_wait_ms
        self.note = note

# Channel readiness check prevents “silent zero-work” runs where no RPCs execute due to DNS/firewall/
# wrong address. If the channel cannot transition to READY within the timeout, fail loudly.
async def _channel_ready(ch: grpc.aio.Channel, timeout_s: float = 3.0) -> None:
    """Fail fast if channel cannot become READY."""
    await asyncio.wait_for(ch.channel_ready(), timeout=timeout_s)

# Sanity RPC rationale:
# This 1-item request validates end-to-end wiring (auth metadata, service availability, stub compatibility)
# before the timed run begins, so subsequent throughput/latency statistics cannot be attributed to a
# misconfiguration that prevented RPC execution.
async def bench_direct_batch(faiss_addr: str, items_per_s: float, duration_s: int, concurrency: int, batch_size: int):
    """
    Send SearchBatch requests directly to FaissDedup.
    - items_per_s is embeddings/sec.
    - batches/sec = items_per_s / batch_size
    """
    md = (("x-internal-auth", INTERNAL_AUTH_TOKEN),)

    ch = grpc.aio.insecure_channel(
        faiss_addr,
        options=[
            ("grpc.max_receive_message_length", GRPC_MAX_MB * 1024 * 1024),
            ("grpc.max_send_message_length", GRPC_MAX_MB * 1024 * 1024),
        ],
    )
    await _channel_ready(ch, 5.0)
    stub = dedup_pb2_grpc.FaissDedupStub(ch)

    # Sanity RPC: 1-item batch. If this fails, print error and exit.
    try:
        _ = await stub.SearchBatch(
            dedup_pb2.SearchBatchRequest(query_ids=[1], embeddings_f32=[_rand_emb_bytes()]),
            timeout=3.0,
            metadata=md,
        )
    except grpc.aio.AioRpcError as e:
        await ch.close()
        raise RuntimeError(f"Sanity SearchBatch failed: {e.code().name} {e.details()}") from e

    sem = asyncio.Semaphore(concurrency)
    samples: List[Sample] = []
    err_counts: Counter = Counter()

    batches_per_s = items_per_s / batch_size
    interval = 1.0 / batches_per_s if batches_per_s > 0 else 0.0

    t0 = time.perf_counter()
    end_t = t0 + duration_s

    attempted_batches = 0
    attempted_items = 0

    async def one_batch(batch_qids: List[int]) -> None:
        nonlocal attempted_items
        async with sem:
            t1 = time.perf_counter()
            try:
                breq = dedup_pb2.SearchBatchRequest(
                    query_ids=batch_qids,
                    embeddings_f32=[_rand_emb_bytes() for _ in batch_qids],
                )
                bresp = await stub.SearchBatch(breq, timeout=3.0, metadata=md)
                rpc_ms = (time.perf_counter() - t1) * 1000.0

                # treat ok if all items say note=="ok" (depends on your server)
                all_ok = True
                faiss_ms_sum = 0.0
                for r in bresp.results:
                    faiss_ms_sum += float(getattr(r, "faiss_ms", 0.0))
                    if getattr(r, "note", "") not in ("ok", ""):
                        all_ok = False

                faiss_ms = faiss_ms_sum / max(1, len(bresp.results))
                note = "ok" if all_ok else "partial_fail"
                samples.append(Sample(all_ok, rpc_ms, faiss_ms, 0.0, note))
                if not all_ok:
                    err_counts[note] += 1

                attempted_items += len(batch_qids)
            except grpc.aio.AioRpcError as e:
                rpc_ms = (time.perf_counter() - t1) * 1000.0
                note = f"rpc_error:{e.code().name}"
                samples.append(Sample(False, rpc_ms, 0.0, 0.0, note))
                err_counts[note] += 1
            except Exception as e:
                # Catch any client-side exception (prevents "silent zero")
                rpc_ms = (time.perf_counter() - t1) * 1000.0
                note = f"client_exception:{type(e).__name__}"
                samples.append(Sample(False, rpc_ms, 0.0, 0.0, note))
                err_counts[note] += 1
                
    # Scheduling model:
    # We use a best-effort constant-arrival scheduler by incrementing a target timestamp (next_t) and sleeping
    # until that target. If the event loop falls behind, the practical offered rate is limited by execution,
    # and achieved throughput should be read from the printed 'achieved' fields rather than assumed equal to
    # the configured --rps.
    # Schedule batches
    tasks = set()
    next_t = time.perf_counter()
    qid = 1

    while time.perf_counter() < end_t:
        batch_qids = list(range(qid, qid + batch_size))
        qid += batch_size
        attempted_batches += 1

        t = asyncio.create_task(one_batch(batch_qids))
        tasks.add(t)
        t.add_done_callback(tasks.discard)

        next_t += interval
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    # Wait for all tasks to complete
    if tasks:
        await asyncio.gather(*list(tasks), return_exceptions=True)

    await ch.close()

    ok = [s for s in samples if s.ok]
    fail = len(samples) - len(ok)

    summary = {
        "mode": "direct-batch",
        "config": {
            "faiss_addr": faiss_addr,
            "rps_items_target": items_per_s,
            "duration_s": duration_s,
            "concurrency": concurrency,
            "batch_size": batch_size,
            "batches_per_s_target": batches_per_s,
        },
        "attempted": {"batches": attempted_batches, "items": attempted_batches * batch_size},
        "completed": {"batches_total": len(samples), "ok_batches": len(ok), "fail_batches": fail},
        "achieved": {
            "batches_per_s": (len(samples) / duration_s) if duration_s else 0.0,
            "items_per_s": ((len(samples) * batch_size) / duration_s) if duration_s else 0.0,
        },
        "stats_ms": {
            "rpc_per_batch": _stat([s.rpc_ms for s in ok]),
            "faiss_per_item": _stat([s.faiss_ms for s in ok]),
        },
        "top_errors": err_counts.most_common(10),
    }

    print("\n=== FAISS benchmark summary ===")
    print(summary)

# Sanity RPC rationale:
# This 1-item request validates end-to-end wiring (auth metadata, service availability, stub compatibility)
# before the timed run begins, so subsequent throughput/latency statistics cannot be attributed to a
# misconfiguration that prevented RPC execution.
async def bench_aggregator(agg_addr: str, queries_per_s: float, duration_s: int, concurrency: int):
    """
    Send Search requests to FaissAggregator (one embedding per query).
    """
    md = (("x-internal-auth", INTERNAL_AUTH_TOKEN),)

    ch = grpc.aio.insecure_channel(
        agg_addr,
        options=[
            ("grpc.max_receive_message_length", GRPC_MAX_MB * 1024 * 1024),
            ("grpc.max_send_message_length", GRPC_MAX_MB * 1024 * 1024),
        ],
    )
    await _channel_ready(ch, 5.0)
    stub = dedup_pb2_grpc.FaissAggregatorStub(ch)

    # Sanity RPC
    try:
        _ = await stub.Search(dedup_pb2.SearchRequest(query_id=1, embedding_f32=_rand_emb_bytes()), timeout=3.0, metadata=md)
    except grpc.aio.AioRpcError as e:
        await ch.close()
        raise RuntimeError(f"Sanity Search failed: {e.code().name} {e.details()}") from e

    sem = asyncio.Semaphore(concurrency)
    samples: List[Sample] = []
    err_counts: Counter = Counter()

    interval = 1.0 / queries_per_s if queries_per_s > 0 else 0.0
    t0 = time.perf_counter()
    end_t = t0 + duration_s
    tasks = set()
    next_t = time.perf_counter()
    qid = 1

    async def one_call(my_qid: int) -> None:
        async with sem:
            t1 = time.perf_counter()
            try:
                req = dedup_pb2.SearchRequest(query_id=my_qid, embedding_f32=_rand_emb_bytes())
                resp = await stub.Search(req, timeout=3.0, metadata=md)
                rpc_ms = (time.perf_counter() - t1) * 1000.0
                note = getattr(resp, "note", "ok")
                ok = (note == "ok" or note == "")
                samples.append(Sample(ok, rpc_ms, float(getattr(resp, "faiss_ms", 0.0)), float(getattr(resp, "aggregator_wait_ms", 0.0)), note))
                if not ok:
                    err_counts[note] += 1
            except grpc.aio.AioRpcError as e:
                rpc_ms = (time.perf_counter() - t1) * 1000.0
                note = f"rpc_error:{e.code().name}"
                samples.append(Sample(False, rpc_ms, 0.0, 0.0, note))
                err_counts[note] += 1
            except Exception as e:
                rpc_ms = (time.perf_counter() - t1) * 1000.0
                note = f"client_exception:{type(e).__name__}"
                samples.append(Sample(False, rpc_ms, 0.0, 0.0, note))
                err_counts[note] += 1

    while time.perf_counter() < end_t:
        t = asyncio.create_task(one_call(qid))
        tasks.add(t)
        t.add_done_callback(tasks.discard)
        qid += 1

        next_t += interval
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    if tasks:
        await asyncio.gather(*list(tasks), return_exceptions=True)

    await ch.close()

    ok = [s for s in samples if s.ok]
    fail = len(samples) - len(ok)

    summary = {
        "mode": "aggregator",
        "config": {"agg_addr": agg_addr, "rps_queries_target": queries_per_s, "duration_s": duration_s, "concurrency": concurrency},
        "completed": {"total": len(samples), "ok": len(ok), "fail": fail},
        "achieved": {"queries_per_s": (len(samples) / duration_s) if duration_s else 0.0},
        "stats_ms": {
            "rpc": _stat([s.rpc_ms for s in ok]),
            "agg_wait": _stat([s.agg_wait_ms for s in ok]),
            "faiss": _stat([s.faiss_ms for s in ok]),
        },
        "top_errors": err_counts.most_common(10),
    }

    print("\n=== FAISS benchmark summary ===")
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["aggregator", "direct-batch"], required=True)
    ap.add_argument("--agg", default="127.0.0.1:50052")
    ap.add_argument("--faiss", default="127.0.0.1:50051")
    ap.add_argument("--rps", type=float, required=True)
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    if args.rps <= 0:
        raise ValueError("--rps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    if args.mode == "aggregator":
        asyncio.run(bench_aggregator(args.agg, args.rps, args.duration, args.concurrency))
    else:
        asyncio.run(bench_direct_batch(args.faiss, args.rps, args.duration, args.concurrency, args.batch_size))


if __name__ == "__main__":
    main()
