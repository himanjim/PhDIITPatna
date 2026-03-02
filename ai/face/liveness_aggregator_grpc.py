#!/usr/bin/env python3
"""
liveness_aggregator_grpc.py

Thin gRPC aggregator that load-balances CheckByClipId across replicas (round-robin).

Why aio here?
-------------
The aggregator should not become the bottleneck in high-concurrency benchmarks.
Using grpc.aio keeps the forwarder lightweight while replicas do the real work.

Auth:
- Optional token via --auth-token or env INTERNAL_AUTH_TOKEN.
- Enforced on inbound calls and forwarded to replicas via metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import List, Tuple

import grpc

import liveness_pb2
import liveness_pb2_grpc


def _parse_bind(bind: str) -> str:
    bind = bind.strip()
    return ("0.0.0.0" + bind) if bind.startswith(":") else bind


def _split_backends(s: str) -> List[str]:
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            out.append(part)
    if not out:
        raise SystemExit("No backends specified (use --backends host1:port1,host2:port2,...)")
    return out


def _md(auth_token: str) -> Tuple[Tuple[str, str], ...]:
    tok = (auth_token or "").strip()
    return (("x-internal-auth", tok),) if tok else tuple()


class _RoundRobin:
    def __init__(self, n: int):
        self.n = max(1, int(n))
        self.i = 0
        self.lock = asyncio.Lock()

    async def next(self) -> int:
        async with self.lock:
            j = self.i
            self.i = (self.i + 1) % self.n
            return j


class LivenessAggregator(liveness_pb2_grpc.LivenessBenchServicer):
    def __init__(self, backends: List[str], auth_token: str):
        self.backends = backends
        self.auth_token = (auth_token or "").strip()
        self._rr = _RoundRobin(len(backends))
        self._channels: List[grpc.aio.Channel] = []
        self._stubs: List[liveness_pb2_grpc.LivenessBenchStub] = []

    async def start(self):
        for b in self.backends:
            ch = grpc.aio.insecure_channel(b)
            self._channels.append(ch)
            self._stubs.append(liveness_pb2_grpc.LivenessBenchStub(ch))

    async def close(self):
        for ch in self._channels:
            await ch.close()

    def _auth_or_abort(self, context: grpc.aio.ServicerContext) -> None:
        if not self.auth_token:
            return
        md = dict(context.invocation_metadata())
        if md.get("x-internal-auth", "") != self.auth_token:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "forbidden")

    async def Health(self, request, context):
        return liveness_pb2.HealthResponse(status="ok", replica_id="aggregator")

    async def GetPolicy(self, request, context):
        self._auth_or_abort(context)
        return await self._stubs[0].GetPolicy(request, metadata=_md(self.auth_token))

    async def ListClips(self, request, context):
        self._auth_or_abort(context)
        return await self._stubs[0].ListClips(request, metadata=_md(self.auth_token))

    async def CheckByClipId(self, request, context):
        self._auth_or_abort(context)
        idx = await self._rr.next()
        return await self._stubs[idx].CheckByClipId(request, metadata=_md(self.auth_token))


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