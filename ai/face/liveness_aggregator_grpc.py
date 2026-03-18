#!/usr/bin/env python3
"""
This module implements a lightweight gRPC aggregator for the liveness
service. It accepts the same external RPC interface as an individual
replica and forwards requests to one of several backend replicas using a
round-robin selection policy. The purpose is to provide a simple load-
distribution layer for benchmarking and operational testing without
embedding liveness logic in the aggregator itself.

The implementation is intentionally minimal: policy and clip-list queries
are forwarded to a representative backend, while liveness requests are
distributed across the configured replica set.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import List, Tuple

import grpc

import liveness_pb2
import liveness_pb2_grpc

# Normalise the bind string so that bare port notation is expanded into a
# full host:port form suitable for gRPC server startup.
def _parse_bind(bind: str) -> str:
    bind = bind.strip()
    return ("0.0.0.0" + bind) if bind.startswith(":") else bind

# Parse the comma-separated backend list supplied on the command line and
# return the individual replica addresses in a validated form.
def _split_backends(s: str) -> List[str]:
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            out.append(part)
    if not out:
        raise SystemExit("No backends specified (use --backends host1:port1,host2:port2,...)")
    return out

# Build the metadata tuple used to propagate the internal authentication
# token to downstream replicas.
def _md(auth_token: str) -> Tuple[Tuple[str, str], ...]:
    tok = (auth_token or "").strip()
    return (("x-internal-auth", tok),) if tok else tuple()

# Provide a simple coroutine-safe round-robin selector over the configured
# backend replica indices.
class _RoundRobin:
    def __init__(self, n: int):
        self.n = max(1, int(n))
        self.i = 0
        self.lock = asyncio.Lock()

    # Return the next backend index in round-robin order while ensuring
    # that concurrent callers do not corrupt the selection state.
    async def next(self) -> int:
        async with self.lock:
            j = self.i
            self.i = (self.i + 1) % self.n
            return j

# Implement the liveness gRPC interface as a forwarding layer over a set
# of backend replicas. The aggregator authenticates inbound requests,
# maintains client channels to replicas, and dispatches liveness calls
# according to the round-robin policy.
class LivenessAggregator(liveness_pb2_grpc.LivenessBenchServicer):
    def __init__(self, backends: List[str], auth_token: str):
        self.backends = backends
        self.auth_token = (auth_token or "").strip()
        self._rr = _RoundRobin(len(backends))
        self._channels: List[grpc.aio.Channel] = []
        self._stubs: List[liveness_pb2_grpc.LivenessBenchStub] = []

    # Create and retain gRPC channels and stubs for all configured backend
    # replicas before the server begins handling traffic.
    async def start(self):
        for b in self.backends:
            ch = grpc.aio.insecure_channel(b)
            self._channels.append(ch)
            self._stubs.append(liveness_pb2_grpc.LivenessBenchStub(ch))

    # Close all downstream channels during orderly shutdown.
    async def close(self):
        for ch in self._channels:
            await ch.close()

    # Enforce the optional internal authentication requirement on inbound
    # RPCs before forwarding them to any replica.
    def _auth_or_abort(self, context: grpc.aio.ServicerContext) -> None:
        if not self.auth_token:
            return
        md = dict(context.invocation_metadata())
        if md.get("x-internal-auth", "") != self.auth_token:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "forbidden")

    async def Health(self, request, context):
        return liveness_pb2.HealthResponse(status="ok", replica_id="aggregator")

    # Forward the policy query to one backend replica and return its
    # response unchanged.
    async def GetPolicy(self, request, context):
        self._auth_or_abort(context)
        return await self._stubs[0].GetPolicy(request, metadata=_md(self.auth_token))

    # Forward the clip-list query to one backend replica and return its
    # response unchanged.
    async def ListClips(self, request, context):
        self._auth_or_abort(context)
        return await self._stubs[0].ListClips(request, metadata=_md(self.auth_token))

    # Forward one clip-based liveness request to the next replica chosen by
    # the round-robin selector.
    async def CheckByClipId(self, request, context):
        self._auth_or_abort(context)
        idx = await self._rr.next()
        return await self._stubs[idx].CheckByClipId(request, metadata=_md(self.auth_token))

# Parse command-line configuration, initialise the backend connections,
# start the aggregator gRPC server, and keep it running until termination.
async def _main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0:9200")
    ap.add_argument("--backends", required=True)
    ap.add_argument("--auth-token", default=os.environ.get("INTERNAL_AUTH_TOKEN", ""))
    args = ap.parse_args()

    bind = _parse_bind(args.bind)
    backends = _split_backends(args.backends)

    agg = LivenessAggregator(backends=backends, auth_token=args.auth_token)
    await agg.start()

    server = grpc.aio.server()
    liveness_pb2_grpc.add_LivenessBenchServicer_to_server(agg, server)
    server.add_insecure_port(bind)

    await server.start()
    print(f"[OK] aggregator started: {bind} backends={len(backends)}")
    try:
        await server.wait_for_termination()
    finally:
        await agg.close()


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
