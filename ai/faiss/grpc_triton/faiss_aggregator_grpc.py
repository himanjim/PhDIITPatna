#!/usr/bin/env python3
"""
This module implements a gRPC-based microbatching front end for a FAISS
search service. It accepts single-query search requests from clients,
queues them briefly, combines them into bounded downstream batch calls,
and then returns one result per original request. The design is intended
to reduce per-request scheduling and transport overhead at high offered
load while preserving a simple single-query interface for callers.

In addition to request forwarding, the service enforces internal-token
authentication, applies explicit backpressure through bounded queues and
downstream concurrency limits, and exposes lightweight administrative
endpoints for health, policy, and recent latency summaries. The code is
therefore best understood as an operational middleware layer between
benchmark clients and the underlying FAISS search tier.
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

# Enforce a simple internal authentication boundary for all gRPC methods
# exposed by the aggregator. Requests must present the expected shared
# token in metadata; otherwise the call is rejected before it reaches the
# service implementation. This keeps the aggregator aligned with the
# assumption that it is reachable only within a controlled internal
# network and should not accept anonymous traffic.
class InternalAuthInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, expected_token: str):
        self.expected = expected_token

    # Intercept each incoming RPC and validate the supplied internal-auth
    # metadata before dispatch. On authentication failure, the method
    # returns an aborting handler so that unauthorised calls are rejected
    # consistently and early in the request path.
    async def intercept_service(self, continuation, handler_call_details):
        md = dict(handler_call_details.invocation_metadata or [])
        tok = md.get("x-internal-auth", "")
        if tok != self.expected:
            async def abort_behavior(request, context):
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "forbidden")
            return grpc.aio.unary_unary_rpc_method_handler(abort_behavior)
        return await continuation(handler_call_details)

# Represent one queued client request awaiting inclusion in a downstream
# FAISS batch. The structure stores the original request, a future used to
# deliver the eventual response, and enqueue/call timestamps needed for
# aggregator-side latency accounting.
@dataclass
class _Pending:
    req: dedup_pb2.SearchRequest
    fut: asyncio.Future
    t_enq: float
    t_call: float
    
    
# ------------------------- Policy snapshot + metrics -------------------------
# Serialize a policy object into a deterministic JSON byte sequence so
# that hashes remain stable across runs and across dictionary insertion
# order. This is used for administrative reporting rather than for the
# search path itself.
def _canon_json_bytes(obj: dict) -> bytes:
    """
    Deterministic JSON for stable hashing.
    (Sorted keys + no whitespace so the hash is reproducible across runs.)
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

# Compute a SHA-256 digest for the supplied byte sequence and return the
# hexadecimal representation. The function is used to expose a stable
# policy hash through the administrative API.
def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# Maintain a bounded rolling window of recent latency observations and
# provide compact percentile summaries for operational inspection. The
# class is intentionally lightweight because it supports an administrative
# endpoint and is not meant to replace a full telemetry system.
class _LatencyStats:
    """Fixed-window latency summary used for lightweight operational metrics."""
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

# Expose minimal administrative HTTP endpoints for health reporting,
# policy inspection, and recent latency summaries. These endpoints are
# operational aids for debugging and benchmarking and are deliberately
# separated from the gRPC search path.
class _AdminHandler(BaseHTTPRequestHandler):
    server_version = "faiss-agg-admin/1.0"

    # Return a compact JSON response with the supplied HTTP status code.
    # A stable JSON encoding is sufficient here because the endpoint is
    # intended for machine-readable administrative inspection.
    def _send_json(self, code: int, obj: dict) -> None:
        b = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    # Validate the internal administrative token for endpoints that expose
    # policy or metrics data. Health-style endpoints may remain available
    # without authentication, but policy and metrics views are protected.
    def _auth_ok(self) -> bool:
        return self.headers.get("X-Internal-Auth", "") == INTERNAL_AUTH_TOKEN

    # Serve the supported administrative GET endpoints. The method returns
    # lightweight health information for liveness checks and, when
    # authorised, exposes policy and recent latency summaries derived from
    # the in-process aggregator state.
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

# Start the auxiliary administrative HTTP server in a background thread so
# that operational inspection does not block or complicate the asyncio-
# based gRPC serving loop.
def _start_admin_http(listen: str) -> None:
    host, port_s = listen.rsplit(":", 1)
    httpd = ThreadingHTTPServer((host, int(port_s)), _AdminHandler)
    t = threading.Thread(target=httpd.serve_forever, name="faiss-agg-admin-http", daemon=True)
    t.start()

# Return the gRPC channel and server options used by the aggregator. These
# parameters bound message sizes and apply conservative keepalive settings
# so that large batched requests can be carried safely without creating
# unnecessary transport churn in long-running benchmark sessions.
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

# Build the small policy description exposed by the administrative API.
# This captures the active microbatching and backpressure configuration so
# that an external observer can verify the running service parameters.
def _policy_obj_agg(faiss_addr: str) -> dict:
    return {
        "service": "faiss_grpc_aggregator",
        "version": "v3.1",
        "microbatch_ms": AGG_MICROBATCH_MS,
        "microbatch_max": AGG_MICROBATCH_MAX,
        "max_inflight": AGG_MAX_INFLIGHT,
        "downstream_faiss": faiss_addr,
    }

# Implement the single-query gRPC interface presented to clients while
# internally forwarding work to the FAISS service in batches. The class
# maintains the request queue, downstream concurrency guard, and rolling
# latency summaries required for both service operation and benchmark
# interpretation.
class FaissAggregatorServicer(dedup_pb2_grpc.FaissAggregatorServicer):

    # Initialise the aggregator state, including the bounded incoming
    # request queue, the semaphore protecting downstream concurrency, and
    # fixed-window latency trackers used by the administrative metrics
    # endpoint.
    def __init__(self, faiss_stub: dedup_pb2_grpc.FaissDedupStub):
        self.faiss = faiss_stub
        self.q: "asyncio.Queue[_Pending]" = asyncio.Queue(maxsize=AGG_MAX_INFLIGHT)
        self.downstream_sem = asyncio.Semaphore(AGG_MAX_DOWNSTREAM_INFLIGHT)
        
        self.lat_rpc = _LatencyStats(METRICS_WINDOW)
        self.lat_wait = _LatencyStats(METRICS_WINDOW)
        self.lat_faiss_batch = _LatencyStats(METRICS_WINDOW)
        
    # Accept one client search request and enqueue it for inclusion in a
    # downstream microbatch. If the queue is full, the call is rejected
    # immediately with RESOURCE_EXHAUSTED so that overload is signalled
    # explicitly rather than converted into unbounded latency growth.
    async def Search(self, request, context):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        try:
            self.q.put_nowait(_Pending(req=request, fut=fut, t_enq=time.perf_counter(), t_call=time.perf_counter()))
        except asyncio.QueueFull:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "overloaded")
        return await fut


    # Continuously drain queued single-query requests into bounded batches
    # and forward them to the downstream FAISS service. The loop waits only
    # for a short microbatching interval or until the maximum batch size is
    # reached, whichever comes first. Responses are then mapped back to the
    # originating client futures, with aggregator-observed waiting time and
    # downstream batch timing incorporated into the returned metadata.
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

            # Form one downstream batch request from the currently collected
            # single-query calls so that backend FAISS processing is amortised
            # across multiple client requests.
            
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

                        # Record aggregator-observed waiting and RPC time even
                        # when the downstream FAISS batch call fails.
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


            # Map results by query_id rather than assuming positional alignment.
            # This avoids incorrect response routing if ordering changes or if
            # the downstream service returns partial results.
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

# Create the downstream FAISS channel, instantiate the aggregator
# servicer, start the gRPC server, and optionally enable the auxiliary
# administrative HTTP interface. This function is the operational entry
# point for running the aggregator as a network service.
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

# Parse command-line options and launch the asynchronous aggregator
# service with the selected listening address and downstream FAISS target.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:50052")
    ap.add_argument("--faiss", default="127.0.0.1:50051")
    args = ap.parse_args()
    asyncio.run(serve(args.listen, args.faiss))

# Standard command-line entry point.
if __name__ == "__main__":
    main()
