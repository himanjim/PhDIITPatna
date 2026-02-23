#!/usr/bin/env python3
"""
faiss_aggregator_grpc.py (v3.1)
-------------------------------
Microbatching aggregator for FAISS gRPC.

Fixes:
- grpc.aio context.abort() must be awaited (no RuntimeWarning).
- conservative keepalive to avoid GOAWAY "too_many_pings".

Env vars
--------
INTERNAL_AUTH_TOKEN (required)
AGG_MICROBATCH_MS=1
AGG_MICROBATCH_MAX=256
AGG_MAX_INFLIGHT=20000
AGG_MAX_DOWNSTREAM_INFLIGHT=50
AGG_DOWNSTREAM_TIMEOUT_S=5
GRPC_MAX_MB=64
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

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



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# This process is a microbatching front-end:
#   Client -> (Search, single embedding) -> Aggregator -> (SearchBatch) -> FAISS server
#
# Rationale:
# - Microbatching amortizes Python scheduling + RPC overhead at high request rates.
# - Backpressure is explicit: when the queue is full, requests fail fast with RESOURCE_EXHAUSTED.
# - Downstream concurrency is bounded (AGG_MAX_DOWNSTREAM_INFLIGHT) to protect FAISS/GPU stability.
# -----------------------------------------------------------------------------
INTERNAL_AUTH_TOKEN = os.getenv("INTERNAL_AUTH_TOKEN", "").strip()
if not INTERNAL_AUTH_TOKEN:
    raise RuntimeError("INTERNAL_AUTH_TOKEN must be set (fail-closed).")

AGG_MICROBATCH_MS = int(os.getenv("AGG_MICROBATCH_MS", "1"))
AGG_MICROBATCH_MAX = int(os.getenv("AGG_MICROBATCH_MAX", "256"))
AGG_MAX_INFLIGHT = int(os.getenv("AGG_MAX_INFLIGHT", "20000"))
AGG_MAX_DOWNSTREAM_INFLIGHT = int(os.getenv("AGG_MAX_DOWNSTREAM_INFLIGHT", "50"))
AGG_DOWNSTREAM_TIMEOUT_S = float(os.getenv("AGG_DOWNSTREAM_TIMEOUT_S", "5"))
GRPC_MAX_MB = int(os.getenv("GRPC_MAX_MB", "64"))


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


@dataclass
class _Pending:
    req: dedup_pb2.SearchRequest
    fut: asyncio.Future
    t_enq: float



# -----------------------------------------------------------------------------
# gRPC channel/server tuning
# -----------------------------------------------------------------------------
# Keepalive settings are deliberately conservative. Under high-rate benchmarking, overly aggressive
# keepalive pings can trigger HTTP/2 GOAWAY (too_many_pings) on some stacks.
# Message sizes are sized for batch RPCs (many embeddings in one request/response).
# -----------------------------------------------------------------------------
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


class FaissAggregatorServicer(dedup_pb2_grpc.FaissAggregatorServicer):
    def __init__(self, faiss_stub: dedup_pb2_grpc.FaissDedupStub):
        self.faiss = faiss_stub
        self.q: "asyncio.Queue[_Pending]" = asyncio.Queue(maxsize=AGG_MAX_INFLIGHT)
        self.downstream_sem = asyncio.Semaphore(AGG_MAX_DOWNSTREAM_INFLIGHT)

    async def Search(self, request, context):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        try:
            self.q.put_nowait(_Pending(req=request, fut=fut, t_enq=time.perf_counter()))
        except asyncio.QueueFull:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "overloaded")
        return await fut

    async def microbatch_loop(self):
        md = (("x-internal-auth", INTERNAL_AUTH_TOKEN),)

        while True:
            p0 = await self.q.get()
            batch: List[_Pending] = [p0]
            deadline = time.perf_counter() + (AGG_MICROBATCH_MS / 1000.0)

            while len(batch) < AGG_MICROBATCH_MAX:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.q.get(), timeout=timeout))
                except asyncio.TimeoutError:
                    break

            # Prepare a single SearchBatch RPC to downstream FAISS. This is where microbatching
            # translates many client calls into one backend call.
            qids = [p.req.query_id for p in batch]
            embs = [p.req.embedding_f32 for p in batch]
            breq = dedup_pb2.SearchBatchRequest(query_ids=qids, embeddings_f32=embs)

            async with self.downstream_sem:
                try:
                    bresp = await self.faiss.SearchBatch(breq, timeout=AGG_DOWNSTREAM_TIMEOUT_S, metadata=md)
                except grpc.aio.AioRpcError as e:
                    for p in batch:
                        if p.fut.done():
                            continue
                        p.fut.set_result(dedup_pb2.SearchResponse(
                            query_id=p.req.query_id,
                            nearest_id=-1,
                            distance_sq=float("inf"),
                            distance_l2=float("inf"),
                            is_match=False,
                            faiss_ms=0.0,
                            faiss_batch_ms=0.0,
                            batch_size=len(batch),
                            aggregator_wait_ms=(time.perf_counter() - p.t_enq) * 1000.0,
                            note=f"downstream_error:{e.code().name}",
                        ))
                    continue


            # Defensive mapping: responses are expected to align with request order, but mapping by
            # query_id prevents incorrect routing if ordering changes or partial results occur.
            by_id = {r.query_id: r for r in bresp.results}
            n = len(batch)

            for p in batch:
                if p.fut.done():
                    continue
                r = by_id.get(p.req.query_id)
                if r is None:
                    p.fut.set_result(dedup_pb2.SearchResponse(
                        query_id=p.req.query_id,
                        nearest_id=-1,
                        distance_sq=float("inf"),
                        distance_l2=float("inf"),
                        is_match=False,
                        faiss_ms=0.0,
                        faiss_batch_ms=float(bresp.faiss_batch_ms),
                        batch_size=n,
                        aggregator_wait_ms=(time.perf_counter() - p.t_enq) * 1000.0,
                        note="missing_result",
                    ))
                else:
                    p.fut.set_result(dedup_pb2.SearchResponse(
                        query_id=r.query_id,
                        nearest_id=r.nearest_id,
                        distance_sq=r.distance_sq,
                        distance_l2=r.distance_l2,
                        is_match=r.is_match,
                        faiss_ms=r.faiss_ms,
                        faiss_batch_ms=float(bresp.faiss_batch_ms),
                        batch_size=n,
                        aggregator_wait_ms=(time.perf_counter() - p.t_enq) * 1000.0,
                        note=r.note,
                    ))


async def serve(listen: str, faiss_addr: str):
    ch = grpc.aio.insecure_channel(faiss_addr, options=_grpc_options())
    await asyncio.wait_for(ch.channel_ready(), timeout=10.0)

    stub = dedup_pb2_grpc.FaissDedupStub(ch)
    agg = FaissAggregatorServicer(stub)

    server = grpc.aio.server(
        interceptors=[InternalAuthInterceptor(INTERNAL_AUTH_TOKEN)],
        options=_grpc_options(),
    )
    dedup_pb2_grpc.add_FaissAggregatorServicer_to_server(agg, server)
    server.add_insecure_port(listen)
    await server.start()

    print(f"[AGG gRPC] listen={listen} -> faiss={faiss_addr} microbatch={AGG_MICROBATCH_MS}ms max={AGG_MICROBATCH_MAX} inflight={AGG_MAX_INFLIGHT} downstream_inflight={AGG_MAX_DOWNSTREAM_INFLIGHT}")
    asyncio.create_task(agg.microbatch_loop())
    await server.wait_for_termination()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:50052")
    ap.add_argument("--faiss", default="127.0.0.1:50051")
    args = ap.parse_args()
    asyncio.run(serve(args.listen, args.faiss))


if __name__ == "__main__":
    main()
