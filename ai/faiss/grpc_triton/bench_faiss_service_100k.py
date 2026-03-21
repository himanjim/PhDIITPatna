# This script benchmarks a gRPC-based embedding search pipeline centred on
# the FAISS aggregator service. Its primary purpose is to generate a
# controlled stream of search requests at a specified offered rate and to
# measure application-level latency as observed by the client. In addition
# to end-to-end RPC timing, the benchmark records server-reported internal
# timings such as FAISS processing time and aggregator queueing delay when
# those values are returned in the response. The script is therefore
# intended for systems evaluation and capacity analysis rather than for
# functional correctness testing alone.

import argparse
import asyncio
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple
import grpc
import numpy as np

# Locate and import the generated gRPC client stubs required for the
# benchmark. The function searches a small set of plausible project-local
# locations so that the script remains usable across different directory
# layouts without forcing manual PYTHONPATH changes. This makes the client
# easier to run in development, testing, and experimental environments.
def _import_stubs():
    """
    Locate dedup_pb2/dedup_pb2_grpc either in:
      - current PYTHONPATH, or
      - ./gen next to this script, or
      - ../gen relative to this script.
    """
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

# Return the gRPC channel options used by the benchmark client. These
# settings define message-size limits and transport keepalive behaviour so
# that large embedding payloads can be transmitted safely and long-running
# experiments remain stable. The values are operational rather than
# algorithmic, but they materially affect benchmark reliability.
def _grpc_options(max_mb: int = 64):
    return [
        ("grpc.max_receive_message_length", max_mb * 1024 * 1024),
        ("grpc.max_send_message_length", max_mb * 1024 * 1024),
        # Conservative keepalive settings to reduce the risk of transport-
        # level disconnects in deployments that limit idle or frequent pings.
        ("grpc.keepalive_time_ms", 120_000),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.keepalive_permit_without_calls", 0),
        ("grpc.http2.max_pings_without_data", 2),
        ("grpc.http2.min_time_between_pings_ms", 10_000),
        ("grpc.http2.min_ping_interval_without_data_ms", 10_000),
    ]

# Generate a deterministic pool of synthetic embeddings for use during the
# benchmark. Each vector is sampled from a standard normal distribution,
# converted to float32, and serialized as raw bytes so that request payload
# construction during the run is inexpensive. Using a fixed seed keeps the
# offered workload reproducible across repeated experiments.
def _make_pool(dim: int, pool: int, seed: int) -> List[bytes]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((pool, dim)).astype(np.float32, copy=False)
    return [x[i].tobytes() for i in range(pool)]

# Compute a simple empirical percentile from a list of observed latency
# values. The implementation is intentionally lightweight and sufficient
# for reporting benchmark summaries such as p50, p95, and p99, where exact
# interpolation is less important than consistent comparison across runs.
def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = int(p * (len(xs) - 1))
    return xs[i]

# Execute the main benchmark against the FAISS aggregator service. The
# function opens an asynchronous gRPC channel, generates requests from a
# reusable pool of synthetic embeddings, and schedules them according to an
# open-loop offered-rate model. For each successful response, it records
# client-observed RPC latency together with internal timings reported by
# the service, including FAISS processing time and aggregator wait time.
# The final summary is intended to characterise both throughput stress and
# queueing behaviour under the selected load parameters.
async def bench_aggregator(addr: str, token: str, rps: float, duration_s: int, concurrency: int, dim: int, pool: int, seed: int):
    md = (("x-internal-auth", token),)
    ch = grpc.aio.insecure_channel(addr, options=_grpc_options())
    await asyncio.wait_for(ch.channel_ready(), timeout=10.0)
    stub = dedup_pb2_grpc.FaissAggregatorStub(ch)

    emb_pool = _make_pool(dim, pool, seed)

    lat_ms: List[float] = []
    faiss_ms: List[float] = []
    agg_wait_ms: List[float] = []
    ok = 0
    fail = 0

    sem = asyncio.Semaphore(concurrency)
    stop_at = time.perf_counter() + duration_s
    qid = 0

    # Send one asynchronous Search RPC to the aggregator and record the
    # outcome. The request carries a synthetic embedding and a query
    # identifier, while the response is used to collect both end-to-end
    # client latency and server-side timing fields. Failures are counted
    # explicitly so that the summary reflects not only latency but also
    # service stability under load.
    async def one_call(my_qid: int):
        nonlocal ok, fail
        emb = random.choice(emb_pool)
        req = dedup_pb2.SearchRequest(query_id=my_qid, embedding_f32=emb)
        t0 = time.perf_counter()
        try:
            resp = await stub.Search(req, timeout=5.0, metadata=md)
            dt = (time.perf_counter() - t0) * 1000.0
            lat_ms.append(dt)
            faiss_ms.append(float(resp.faiss_ms))
            agg_wait_ms.append(float(resp.aggregator_wait_ms))
            ok += 1
        except grpc.aio.AioRpcError:
            fail += 1
        finally:
            sem.release()

    # Open-loop request scheduler designed to keep the offered arrival rate
    # close to the configured target, independent of observed response time.
    next_due = time.perf_counter()
    interval = 1.0 / max(1e-9, rps)

    while True:
        now = time.perf_counter()
        if now >= stop_at:
            break

        # If scheduling falls behind, release additional requests without
        # delay so that the offered load remains close to the target rate.
        while next_due <= now and now < stop_at:
            await sem.acquire()
            asyncio.create_task(one_call(qid))
            qid += 1
            next_due += interval

        # Sleep briefly to avoid unnecessary busy-waiting between releases.
        await asyncio.sleep(min(0.001, max(0.0, next_due - time.perf_counter())))

    # Wait for all in-flight RPCs to complete before closing the channel.
    while sem._value != concurrency:
        await asyncio.sleep(0.01)

    await ch.close()

    total = ok + fail
    print("\n=== Aggregator benchmark summary ===")
    print({"addr": addr, "rps_target": rps, "duration_s": duration_s, "concurrency": concurrency, "sent": total, "ok": ok, "fail": fail})
    if ok:
        print({
            "rpc_ms": {
                "p50": _pct(lat_ms, 0.50),
                "p95": _pct(lat_ms, 0.95),
                "p99": _pct(lat_ms, 0.99),
                "mean": statistics.mean(lat_ms),
                "max": max(lat_ms),
            },
            "agg_wait_ms": {
                "p50": _pct(agg_wait_ms, 0.50),
                "p95": _pct(agg_wait_ms, 0.95),
                "p99": _pct(agg_wait_ms, 0.99),
                "mean": statistics.mean(agg_wait_ms),
                "max": max(agg_wait_ms),
            },
            "faiss_ms": {
                "p50": _pct(faiss_ms, 0.50),
                "p95": _pct(faiss_ms, 0.95),
                "p99": _pct(faiss_ms, 0.99),
                "mean": statistics.mean(faiss_ms),
                "max": max(faiss_ms),
            },
        })

# Parse command-line parameters and launch the benchmark with the selected
# workload characteristics. The exposed arguments control the target
# offered rate, test duration, maximum client-side concurrency, embedding
# dimensionality, synthetic workload size, and random seed so that the
# experiment can be reproduced or scaled systematically.
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True, help="Aggregator address host:port, e.g. 127.0.0.1:50052")
    ap.add_argument("--token", required=True, help="INTERNAL_AUTH_TOKEN")
    ap.add_argument("--rps", type=float, default=1000.0)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=1000)
    ap.add_argument("--dim", type=int, default=int(os.getenv("EMB_DIM", "512")))
    ap.add_argument("--pool", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    await bench_aggregator(args.agg, args.token, args.rps, args.duration, args.concurrency, args.dim, args.pool, args.seed)

# Standard command-line entry point for running the asynchronous benchmark.
if __name__ == "__main__":
    asyncio.run(main())
