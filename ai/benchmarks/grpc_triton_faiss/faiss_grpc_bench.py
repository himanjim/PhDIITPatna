#!/usr/bin/env python3
"""
This script benchmarks the gRPC serving tier of a FAISS-based similarity
search system under synthetic embedding load. It supports two execution
modes: direct batch calls to the FAISS search service and single-query
calls to an upstream aggregator that may microbatch requests before
forwarding them downstream. The benchmark is designed to measure service-
level behaviour such as RPC latency, queueing effects, batching overhead,
and overload responses, rather than biometric recognition accuracy.

Synthetic embeddings are appropriate here because the purpose of the test
is capacity and latency characterisation of the serving path. If the
research question instead concerns embedding distribution, threshold
behaviour, or biometric decision quality, the workload should be generated
from the same face model used in the full verification pipeline.
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

# Locate the generated protobuf client stubs from a small set of project-
# local paths so that the benchmark can run from a checked-out directory
# without requiring manual PYTHONPATH configuration. This improves
# reproducibility across development and experimental environments.
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
# Compute a simple empirical percentile from a list of observed values.
# This helper is used for benchmark summaries where consistent reporting
# across runs is more important than interpolation sophistication.
def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    k = max(0, min(len(ys) - 1, k))
    return ys[k]

# Summarise a list of latency measurements using a compact set of
# descriptive statistics commonly reported in systems benchmarking,
# including p50, p95, p99, mean, and maximum.
def _stat(vals: List[float]) -> dict:
    return {
        "count": len(vals),
        "p50": _percentile(vals, 50),
        "p95": _percentile(vals, 95),
        "p99": _percentile(vals, 99),
        "mean": statistics.mean(vals) if vals else float("nan"),
        "max": max(vals) if vals else float("nan"),
    }

# Generate one synthetic embedding in the wire format expected by the gRPC
# services. The vector is sampled from a standard normal distribution,
# stored as float32, and serialized as raw bytes so that payload size and
# parsing overhead remain close to the intended service contract.
def _rand_emb_bytes() -> bytes:
    v = _RNG.standard_normal((EMB_DIM,), dtype=np.float32)
    return v.tobytes()


# ------------------------- Benchmark modes ------------------------- #
# Lightweight container for the outcome of one completed benchmarked RPC or
# batch RPC. It stores whether the call is considered successful together
# with client-observed latency and selected timing fields reported by the
# server.
class Sample:
    __slots__ = ("ok", "rpc_ms", "faiss_ms", "agg_wait_ms", "note")
    def __init__(self, ok: bool, rpc_ms: float, faiss_ms: float, agg_wait_ms: float, note: str):
        self.ok = ok
        self.rpc_ms = rpc_ms
        self.faiss_ms = faiss_ms
        self.agg_wait_ms = agg_wait_ms
        self.note = note

# Verify that the gRPC channel becomes ready before the timed portion of
# the benchmark begins. This prevents runs from appearing artificially
# successful when no effective RPC traffic is possible because of address,
# connectivity, or service-availability problems.
async def _channel_ready(ch: grpc.aio.Channel, timeout_s: float = 3.0) -> None:
    """Fail fast if channel cannot become READY."""
    await asyncio.wait_for(ch.channel_ready(), timeout=timeout_s)

# Benchmark the direct batch-search interface exposed by the FAISS service.
# The workload generator issues SearchBatch RPCs at a configured offered
# rate, using synthetic embeddings and bounded client-side concurrency. A
# short sanity request is performed before timing begins so that measured
# results are not confounded by missing authentication, stub mismatch, or
# an unreachable service. The reported summary distinguishes successful and
# failed batches and presents both configured and achieved throughput.
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

    # Execute one batch RPC against the FAISS service and record the
    # resulting client latency together with any per-item FAISS timing
    # returned by the server. Errors are captured explicitly so that batch
    # failures contribute to the final overload and stability picture
    # rather than disappearing from the summary.
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

# Benchmark the single-query interface exposed by the aggregator service.
# Each RPC carries one synthetic embedding, while the aggregator is free to
# microbatch requests internally before forwarding them to FAISS. As in the
# direct-batch mode, a short sanity request is issued before the measured
# run so that the benchmark reflects service behaviour rather than setup
# failure. The summary includes client-observed RPC latency together with
# any aggregator wait time and FAISS processing time reported in the
# response.
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

    # Execute one single-query RPC against the aggregator and record the
    # resulting outcome. The collected sample includes overall client-side
    # latency and, when available, the service-reported FAISS time and
    # aggregator waiting time associated with that request.
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

# Parse command-line arguments, validate the requested workload settings,
# and dispatch the benchmark in either aggregator mode or direct-batch
# mode. This function provides the main reproducible entry point for the
# experiment.
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

# Standard command-line entry point.
if __name__ == "__main__":
    main()
