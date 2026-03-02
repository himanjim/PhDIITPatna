#!/usr/bin/env python3
"""
bench_liveness_grpc.py

High-concurrency gRPC benchmark driver for liveness (via aggregator or direct replica).

Outputs:
- <out_prefix>_runs.csv    : one row per request
- <out_prefix>_summary.csv : one-row summary with p50/p95/p99 + throughput + error_rate

Semantics:
- rpc_wall_ms       : client-observed RPC duration (includes queuing/transport)
- server_compute_ms : replica compute-only time around liveness call
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import grpc

import liveness_pb2
import liveness_pb2_grpc


def _pctl(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = int(round((len(s) - 1) * q))
    i = max(0, min(i, len(s) - 1))
    return float(s[i])


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


@dataclass(frozen=True)
class Job:
    clip_id: str
    prompt: str


def _workload_jobs(preset: str) -> List[Tuple[Job, float]]:
    preset = (preset or "").strip().lower()
    if preset == "idle_mix":
        return [(Job("IDLE", "none"), 0.70), (Job("HEAD_SHAKE", "none"), 0.30)]
    if preset == "prompt_mix":
        return [(Job("POSE", "pose"), 0.50), (Job("BLINK_TRY", "blink"), 0.50)]
    if preset == "full_mix":
        return [
            (Job("IDLE", "none"), 0.25),
            (Job("HEAD_SHAKE", "none"), 0.15),
            (Job("POSE", "pose"), 0.20),
            (Job("BLINK_TRY", "blink"), 0.20),
            (Job("MULTIFACE", "none"), 0.10),
            (Job("NO_MOTION", "none"), 0.10),
        ]
    raise SystemExit("workload must be: idle_mix|prompt_mix|full_mix")


def _pick_job(rng: random.Random, jobs: List[Tuple[Job, float]]) -> Job:
    total = sum(w for _, w in jobs)
    x = rng.random() * total
    acc = 0.0
    for j, w in jobs:
        acc += w
        if x <= acc:
            return j
    return jobs[-1][0]


def _md(auth_token: str) -> Tuple[Tuple[str, str], ...]:
    tok = (auth_token or "").strip()
    return (("x-internal-auth", tok),) if tok else tuple()


def _ensure_parent(path: str) -> None:
    p = Path(path).expanduser().resolve()
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


async def _warmup(stub, auth_token: str, jobs, n: int, seed: int, max_frames: int, timeout_s: float):
    rng = random.Random(seed)
    for i in range(n):
        j = _pick_job(rng, jobs)
        req = liveness_pb2.CheckByClipIdRequest(
            request_id=f"warmup-{i}",
            clip_id=j.clip_id,
            prompt=j.prompt,
            max_frames=max_frames,
        )
        await stub.CheckByClipId(req, metadata=_md(auth_token), timeout=timeout_s)


async def _worker(wid: int, stub, auth_token: str, jobs, n: int, seed: int, max_frames: int, timeout_s: float) -> List[Dict]:
    rng = random.Random(seed + wid * 10007)
    rows: List[Dict] = []

    for k in range(n):
        j = _pick_job(rng, jobs)
        req = liveness_pb2.CheckByClipIdRequest(
            request_id=f"w{wid}-{k}",
            clip_id=j.clip_id,
            prompt=j.prompt,
            max_frames=max_frames,
        )
        t0 = time.perf_counter()
        try:
            resp = await stub.CheckByClipId(req, metadata=_md(auth_token), timeout=timeout_s)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            row = {
                "worker": wid,
                "seq": k,
                "clip_id": j.clip_id,
                "prompt": j.prompt,
                "ok": int(resp.ok),
                "reason_code": resp.reason_code,
                "replica_id": resp.replica_id,
                "rpc_wall_ms": float(wall_ms),
                "server_compute_ms": float(resp.server_compute_ms),
            }
            if "server_total_ms" in resp.metrics:
                row["server_total_ms"] = float(resp.metrics["server_total_ms"])
        except grpc.aio.AioRpcError as e:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            row = {
                "worker": wid,
                "seq": k,
                "clip_id": j.clip_id,
                "prompt": j.prompt,
                "ok": 0,
                "reason_code": f"RPC_ERROR_{e.code().name}",
                "replica_id": "",
                "rpc_wall_ms": float(wall_ms),
                "server_compute_ms": 0.0,
            }
        rows.append(row)

    return rows


def _summarise(rows: List[Dict]) -> Dict[str, float]:
    wall = [float(r["rpc_wall_ms"]) for r in rows]
    comp = [float(r["server_compute_ms"]) for r in rows]
    err = [1 for r in rows if str(r.get("reason_code", "")).startswith("RPC_ERROR_")]

    return {
        "n": float(len(rows)),
        "error_rate": (float(sum(err)) / float(len(rows))) if rows else 0.0,

        "rpc_wall_mean": _mean(wall),
        "rpc_wall_p50": _pctl(wall, 0.50),
        "rpc_wall_p95": _pctl(wall, 0.95),
        "rpc_wall_p99": _pctl(wall, 0.99),
        "rpc_wall_max": max(wall) if wall else 0.0,

        "server_compute_mean": _mean(comp),
        "server_compute_p50": _pctl(comp, 0.50),
        "server_compute_p95": _pctl(comp, 0.95),
        "server_compute_p99": _pctl(comp, 0.99),
        "server_compute_max": max(comp) if comp else 0.0,
    }


async def _main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--auth-token", default=os.environ.get("INTERNAL_AUTH_TOKEN", ""))
    ap.add_argument("--workload", default="idle_mix")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--out-prefix", default="bench_liveness")
    args = ap.parse_args()

    jobs = _workload_jobs(args.workload)

    channel = grpc.aio.insecure_channel(args.target)
    stub = liveness_pb2_grpc.LivenessBenchStub(channel)

    h = await stub.Health(liveness_pb2.HealthRequest(), metadata=_md(args.auth_token), timeout=args.timeout_s)
    print(f"[OK] target health: status={h.status} replica_id={h.replica_id}")

    try:
        cl = await stub.ListClips(liveness_pb2.ListClipsRequest(), metadata=_md(args.auth_token), timeout=args.timeout_s)
        print(f"[OK] server clip cache: {len(cl.clip_ids)} clips")
    except Exception as e:
        print(f"[WARN] ListClips failed (continuing): {e}")

    if args.warmup > 0:
        print(f"[STEP] warmup: {args.warmup} requests")
        await _warmup(stub, args.auth_token, jobs, args.warmup, args.seed, args.max_frames, args.timeout_s)

    conc = max(1, int(args.concurrency))

    # exact distribution of n requests
    per = int(args.n) // conc
    rem = int(args.n) % conc
    per_worker = [per + (1 if i < rem else 0) for i in range(conc)]
    planned = sum(per_worker)

    print(f"[STEP] run: workload={args.workload} concurrency={conc} planned_n={planned} max_frames={args.max_frames} timeout_s={args.timeout_s}")

    t0 = time.perf_counter()
    worker_rows = await asyncio.gather(*[
        _worker(w, stub, args.auth_token, jobs, per_worker[w], args.seed, args.max_frames, args.timeout_s)
        for w in range(conc)
    ])
    wall_s = time.perf_counter() - t0

    rows: List[Dict] = []
    for rr in worker_rows:
        rows.extend(rr)

    await channel.close()

    runs_path = f"{args.out_prefix}_runs.csv"
    summ_path = f"{args.out_prefix}_summary.csv"
    _ensure_parent(runs_path)
    _ensure_parent(summ_path)

    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        cols = sorted(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[OUT] wrote {runs_path} rows={len(rows)}")

    summ = _summarise(rows)
    throughput = (len(rows) / wall_s) if wall_s > 0 else 0.0

    summ_row = {
        "target": args.target,
        "workload": args.workload,
        "concurrency": conc,
        "planned_n": planned,
        "wall_s": wall_s,
        "throughput_rps": throughput,
        **summ,
    }
    with open(summ_path, "w", newline="", encoding="utf-8") as f:
        cols = list(summ_row.keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(summ_row)
    print(f"[OUT] wrote {summ_path}")

    print("\n--- Summary ---")
    print(f"Throughput: {throughput:.1f} req/s  Wall: {wall_s:.3f}s  N: {len(rows)}  ErrorRate: {summ['error_rate']*100.0:.2f}%")
    print(f"RPC wall ms: mean={summ['rpc_wall_mean']:.2f}  p50={summ['rpc_wall_p50']:.2f}  p95={summ['rpc_wall_p95']:.2f}  p99={summ['rpc_wall_p99']:.2f}  max={summ['rpc_wall_max']:.2f}")
    print(f"Server compute ms: mean={summ['server_compute_mean']:.2f}  p50={summ['server_compute_p50']:.2f}  p95={summ['server_compute_p95']:.2f}  p99={summ['server_compute_p99']:.2f}  max={summ['server_compute_max']:.2f}")


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()