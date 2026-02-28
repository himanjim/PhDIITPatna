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

import hashlib
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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

ADMIN_HTTP_LISTEN = os.getenv("ADMIN_HTTP_LISTEN", "").strip()  # e.g. "127.0.0.1:9200"
METRICS_WINDOW = int(os.getenv("METRICS_WINDOW", "2048"))


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
    t_call: float
    
    
# ------------------------- Policy snapshot + metrics -------------------------

def _canon_json_bytes(obj: dict) -> bytes:
    """
    Deterministic JSON for stable hashing.
    (Sorted keys + no whitespace so the hash is reproducible across runs.)
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _LatencyStats:
    """Tiny fixed-window latency stats (enough for /v1/metrics)."""
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

# ---- Admin HTTP endpoints (module scope) ----
_ADMIN_STATE = {"agg": None, "faiss_addr": None, "started_ts": time.time()}

class _AdminHandler(BaseHTTPRequestHandler):
    server_version = "faiss-agg-admin/1.0"

    def _send_json(self, code: int, obj: dict) -> None:
        b = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _auth_ok(self) -> bool:
        return self.headers.get("X-Internal-Auth", "") == INTERNAL_AUTH_TOKEN

    def do_GET(self):  # noqa: N802
        if self.path in ("/v1/health", "/ping"):
            agg = _ADMIN_STATE["agg"]
            qd = int(agg.q.qsize()) if agg else 0
            return self._send_json(200, {"status": "ok", "queue_depth": qd})

        if not self._auth_ok():
            return self._send_json(403, {"detail": "forbidden"})

        agg = _ADMIN_STATE["agg"]
        faiss_addr = _ADMIN_STATE["faiss_addr"]
        if agg is None or faiss_addr is None:
            return self._send_json(503, {"detail": "not_ready"})

        if self.path == "/v1/policy":
            pol = _policy_obj_agg(faiss_addr)
            canon = _canon_json_bytes(pol)
            return self._send_json(200, {"policy": pol, "policy_hash_sha256": _sha256_hex(canon)})

        if self.path == "/v1/metrics":
            return self._send_json(
                200,
                {
                    "queue_depth": int(agg.q.qsize()),
                    "rpc_ms": agg.lat_rpc.summary(),
                    "agg_wait_ms": agg.lat_wait.summary(),
                    "faiss_batch_ms": agg.lat_faiss_batch.summary(),
                    "started_ts": _ADMIN_STATE["started_ts"],
                },
            )

        return self._send_json(404, {"detail": "not_found"})

def _start_admin_http(listen: str) -> None:
    host, port_s = listen.rsplit(":", 1)
    httpd = ThreadingHTTPServer((host, int(port_s)), _AdminHandler)
    t = threading.Thread(target=httpd.serve_forever, name="faiss-agg-admin-http", daemon=True)
    t.start()

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
    
def _policy_obj_agg(faiss_addr: str) -> dict:
    return {
        "service": "faiss_grpc_aggregator",
        "version": "v3.1",
        "microbatch_ms": AGG_MICROBATCH_MS,
        "microbatch_max": AGG_MICROBATCH_MAX,
        "max_inflight": AGG_MAX_INFLIGHT,
        "downstream_faiss": faiss_addr,
    }


class FaissAggregatorServicer(dedup_pb2_grpc.FaissAggregatorServicer):
    def __init__(self, faiss_stub: dedup_pb2_grpc.FaissDedupStub):
        self.faiss = faiss_stub
        self.q: "asyncio.Queue[_Pending]" = asyncio.Queue(maxsize=AGG_MAX_INFLIGHT)
        self.downstream_sem = asyncio.Semaphore(AGG_MAX_DOWNSTREAM_INFLIGHT)
        
        self.lat_rpc = _LatencyStats(METRICS_WINDOW)
        self.lat_wait = _LatencyStats(METRICS_WINDOW)
        self.lat_faiss_batch = _LatencyStats(METRICS_WINDOW)

    async def Search(self, request, context):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        try:
            self.q.put_nowait(_Pending(req=request, fut=fut, t_enq=time.perf_counter(), t_call=time.perf_counter()))
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
                    t0 = time.perf_counter()
                    bresp = await self.faiss.SearchBatch(breq, timeout=AGG_DOWNSTREAM_TIMEOUT_S, metadata=md)
                    self.lat_faiss_batch.add_ms((time.perf_counter() - t0) * 1000.0)
                except grpc.aio.AioRpcError as e:
                    for p in batch:
                        if p.fut.done():
                            continue
                            
                        now = time.perf_counter()
                        wait_ms = (now - p.t_enq) * 1000.0
                        rpc_ms = (now - p.t_call) * 1000.0

                        # Aggregator-side latency accounting (cheap O(1) append)
                        self.lat_wait.add_ms(wait_ms)
                        self.lat_rpc.add_ms(rpc_ms)

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
                    
                now = time.perf_counter()
                wait_ms = (now - p.t_enq) * 1000.0
                rpc_ms = (now - p.t_call) * 1000.0

                self.lat_wait.add_ms(wait_ms)
                self.lat_rpc.add_ms(rpc_ms)    
                
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
    
    if ADMIN_HTTP_LISTEN:
        _ADMIN_STATE["agg"] = agg
        _ADMIN_STATE["faiss_addr"] = faiss_addr
        _start_admin_http(ADMIN_HTTP_LISTEN)
        print(f"[AGG ADMIN] http://{ADMIN_HTTP_LISTEN} (auth via X-Internal-Auth)")

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
