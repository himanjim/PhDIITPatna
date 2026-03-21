#!/usr/bin/env python3
"""
This script benchmarks Triton-based embedding inference under two client
submission models: single-image RPCs and client-side microbatched RPCs.
Images are loaded and converted into model-ready tensors before timing
begins, so the reported results reflect Triton inference plus client-side
RPC behaviour rather than full end-to-end image processing. The benchmark
is intended for service-capacity analysis, including latency, throughput,
concurrency effects, batching behaviour, and overload response under an
open-loop offered load.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from collections import Counter
from typing import Dict, List, Set

import cv2
import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput


TRITON_GRPC = os.getenv("TRITON_GRPC", "localhost:8001")
TRITON_MODEL = os.getenv("TRITON_MODEL", "buffalo_l")
EMB_LAYER = os.getenv("EMB_LAYER", "683")
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "face_benchmark_gdrive/voter_images")

IMG_SIZE = (112, 112)  # H,W
EMB_DIM = int(os.getenv("EMB_DIM", "512"))

# Preload and preprocess all usable images from the configured directory
# before the benchmark starts. Each image is decoded, converted to float32,
# resized to the model input shape, and transformed into CHW layout so
# that the timed benchmark path excludes file-system and preprocessing
# overhead.
def load_images(folder: str) -> List[np.ndarray]:
    """Preload + preprocess images (CHW float32 in [0,1])."""
    imgs: List[np.ndarray] = []
    for fname in sorted(os.listdir(folder)):
        p = os.path.join(folder, fname)
        if not os.path.isfile(p):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        img = img.astype("float32") / 255.0
        img = cv2.resize(img, IMG_SIZE).transpose(2, 0, 1)  # CHW
        imgs.append(img)
    if not imgs:
        raise RuntimeError(f"No images found in IMAGE_FOLDER={folder!r}")
    return imgs

# Summarise a list of latency values in milliseconds using standard
# descriptive statistics commonly reported in systems benchmarking,
# including p50, p95, p99, mean, and maximum.
def np_stats_ms(vals: List[float]) -> Dict[str, float]:
    """Compute p50/p95/p99/mean/max for millisecond values."""
    if not vals:
        return {"count": 0, "p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "mean": float("nan"), "max": float("nan")}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "max": float(a.max()),
    }

# Validate that the deployed Triton model is configured for batching and
# that its declared maximum batch size is compatible with the requested
# client-side microbatch size. This prevents the benchmark from running in
# a configuration that cannot realise the intended batching behaviour.
def check_model_batching(client: InferenceServerClient, batch_size: int) -> None:
    """Fail early if Triton model batching is disabled or max_batch_size < requested batch_size."""

    cfg = client.get_model_config(TRITON_MODEL)
    max_bs = None
    if hasattr(cfg, "config") and hasattr(cfg.config, "max_batch_size"):
        max_bs = int(cfg.config.max_batch_size)
    elif hasattr(cfg, "max_batch_size"):
        max_bs = int(cfg.max_batch_size)

    if max_bs is None:
        return
    if max_bs == 0:
        raise RuntimeError("Model batching disabled (max_batch_size=0).")
    if max_bs < batch_size:
        raise RuntimeError(f"Model max_batch_size={max_bs} < requested batch_size={batch_size}.")

# Execute one Triton inference call for the supplied batch and return the
# embedding tensor. The underlying client call is blocking, so it is moved
# into a worker thread to preserve compatibility with the surrounding
# asyncio-based benchmark harness.
async def triton_infer(client: InferenceServerClient, batch: np.ndarray, out: InferRequestedOutput) -> np.ndarray:
    """Run one blocking Triton infer call in a worker thread (asyncio.to_thread)."""

    ii = InferInput("input.1", batch.shape, "FP32")
    ii.set_data_from_numpy(batch)
    resp = await asyncio.to_thread(client.infer, TRITON_MODEL, inputs=[ii], outputs=[out])
    return resp.as_numpy(EMB_LAYER)

# Track currently outstanding asyncio tasks without retaining an unbounded
# history of completed tasks. This keeps the benchmark harness lightweight
# while still allowing a final drain step after request scheduling stops.
class TaskTracker:
    """Tracks outstanding tasks without storing a huge list."""

    def __init__(self) -> None:
        self._tasks: Set[asyncio.Task] = set()

    # Register a newly created task and arrange for it to be removed from
    # the tracked set automatically when it completes.
    def track(self, t: asyncio.Task) -> None:
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    # Wait until all currently tracked tasks have completed. This is used
    # after the scheduling window closes so that in-flight RPCs are not
    # abandoned before summary statistics are computed.
    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

# Run the benchmark in single-image mode using an open-loop constant-
# arrival scheduler. Each scheduled request sends exactly one image to
# Triton, and the function records per-call and per-image latency together
# with basic success and failure counts after the warmup interval.
async def run_single(
    client: InferenceServerClient,
    imgs: List[np.ndarray],
    out: InferRequestedOutput,
    rps_images: float,
    t_end: float,
    t_warm_end: float,
    sem: asyncio.Semaphore,
    tracker: TaskTracker,
    lat_ms: List[float],
    per_image_ms: List[float],
    counters: Dict[str, int],
) -> None:
    """Open-loop constant arrival: schedule one request each 1/rps seconds, without catch-up bursts."""

    interval = 1.0 / rps_images
    next_deadline = time.perf_counter()

    # Execute one single-image inference request under the concurrency
    # semaphore and record the resulting latency and outcome if the call
    # falls within the measured portion of the run.
    async def one_call() -> None:
        async with sem:
            img = random.choice(imgs)
            batch = img[None, ...]  # [1,3,112,112] view
            t0 = time.perf_counter()
            ok = True
            try:
                emb = await triton_infer(client, batch, out)
                if emb.shape[0] != 1 or emb.shape[1] != EMB_DIM:
                    raise RuntimeError(f"Unexpected embedding shape {emb.shape}")
            except Exception:
                ok = False
            dt = (time.perf_counter() - t0) * 1000.0

            now = time.perf_counter()
            if now >= t_warm_end:
                counters["calls_total"] += 1
                if ok:
                    counters["calls_ok"] += 1
                    counters["images_ok"] += 1
                    lat_ms.append(dt)
                    per_image_ms.append(dt)
                else:
                    counters["calls_fail"] += 1

    while time.perf_counter() < t_end:
        now = time.perf_counter()
        if now < next_deadline:
            await asyncio.sleep(next_deadline - now)
            continue

        tracker.track(asyncio.create_task(one_call()))
        next_deadline += interval

        # If we're late, skip missed slots (no burst catch-up).
        if now - next_deadline > interval:
            next_deadline = now

# Run the benchmark in client-side microbatching mode. Images are offered
# to a bounded local queue at the configured open-loop arrival rate, and a
# consumer task flushes batches according to the configured batch-size and
# batch-window limits. This mode is intended to study the interaction
# between arrival rate, batching opportunity, concurrency, and overload
# behaviour.
async def run_client_batch(
    client: InferenceServerClient,
    imgs: List[np.ndarray],
    out: InferRequestedOutput,
    rps_images: float,
    t_end: float,
    t_warm_end: float,
    sem: asyncio.Semaphore,
    tracker: TaskTracker,
    lat_ms: List[float],
    per_image_ms: List[float],
    batch_hist: Counter,
    counters: Dict[str, int],
    batch_size: int,
    batch_window_ms: int,
) -> None:
    """Client microbatching: producer enqueues images at rps; consumer flushes batches."""
    
    # Use a bounded producer queue so that the harness itself does not
    # create unbounded memory growth under overload. When this queue fills,
    # offered inputs are dropped and counted explicitly.

    q: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=max(1, batch_size) * max(1, sem._value) * 2)

    # Offer images to the local batching queue at the configured open-loop
    # rate. If the queue is full, the input is dropped rather than delayed,
    # preserving the intended arrival model of the benchmark.
    async def producer() -> None:
        interval = 1.0 / rps_images
        next_deadline = time.perf_counter()
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            if now < next_deadline:
                await asyncio.sleep(next_deadline - now)
                continue

            try:
                q.put_nowait(random.choice(imgs))
                if now >= t_warm_end:
                    counters["images_enqueued"] += 1
            # When the local queue is full, the harness drops the offered
            # input rather than allowing unbounded backlog to accumulate.
            # The resulting drop count is therefore part of benchmark
            # interpretation, not merely a client-side anomaly.
            except asyncio.QueueFull:
                if now >= t_warm_end:
                    counters["images_dropped"] += 1

            next_deadline += interval
            if now - next_deadline > interval:
                next_deadline = now

    # Execute one batched Triton inference request and record both per-call
    # latency and a simple per-image latency proxy obtained by dividing the
    # batch latency by the realised batch size.
    async def one_batch_call(batch_imgs: List[np.ndarray]) -> None:
        bs = len(batch_imgs)
        async with sem:
            batch = np.stack(batch_imgs, axis=0).astype(np.float32, copy=False)
            t0 = time.perf_counter()
            ok = True
            try:
                emb = await triton_infer(client, batch, out)
                if emb.shape[0] != bs or emb.shape[1] != EMB_DIM:
                    raise RuntimeError(f"Unexpected embedding shape {emb.shape}")
            except Exception:
                ok = False
            dt = (time.perf_counter() - t0) * 1000.0

            now = time.perf_counter()
            if now >= t_warm_end:
                counters["calls_total"] += 1
                counters["batches_sent"] += 1
                batch_hist[bs] += 1
                if ok:
                    counters["calls_ok"] += 1
                    counters["images_ok"] += bs
                    lat_ms.append(dt)
                    per_image_ms.append(dt / max(1, bs))
                else:
                    counters["calls_fail"] += 1

    # Collect images from the local queue and flush them as a batch once
    # either the target batch size is reached or the batch window expires.
    # This approximates a simple client-side microbatcher with bounded
    # queueing delay.
    async def consumer() -> None:
        while time.perf_counter() < t_end or (not q.empty()):
            batch_imgs: List[np.ndarray] = []
            deadline = time.perf_counter() + (batch_window_ms / 1000.0)

            try:
                img = await asyncio.wait_for(q.get(), timeout=max(0.0, deadline - time.perf_counter()))
                batch_imgs.append(img)
            except asyncio.TimeoutError:
                continue

            while len(batch_imgs) < batch_size and time.perf_counter() < deadline:
                try:
                    batch_imgs.append(q.get_nowait())
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)
                    break

            tracker.track(asyncio.create_task(one_batch_call(batch_imgs)))

    tracker.track(asyncio.create_task(producer()))
    tracker.track(asyncio.create_task(consumer()))

# Orchestrate the full benchmark run, including input loading, Triton
# client setup, warmup handling, execution in the selected benchmark mode,
# draining of outstanding tasks, and final summary generation. This is the
# main experimental control function of the script.
async def run_bench(mode: str, rps_images: float, duration_s: int, warmup_s: int,
                    concurrency: int, batch_size: int, batch_window_ms: int, seed: int) -> None:
    if rps_images <= 0:
        raise ValueError("--rps must be > 0")
    if warmup_s < 0 or warmup_s >= duration_s:
        raise ValueError("--warmup must be >=0 and < duration")

    random.seed(seed)
    np.random.seed(seed)

    imgs = load_images(IMAGE_FOLDER)
    client = InferenceServerClient(TRITON_GRPC)
    if mode == "client-batch":
        check_model_batching(client, batch_size)

    out = InferRequestedOutput(EMB_LAYER)
    sem = asyncio.Semaphore(concurrency)
    tracker = TaskTracker()

    lat_ms: List[float] = []
    per_image_ms: List[float] = []
    batch_hist: Counter = Counter()

    counters = {
        "calls_total": 0,
        "calls_ok": 0,
        "calls_fail": 0,
        "images_ok": 0,
        "images_enqueued": 0,
        "images_dropped": 0,
        "batches_sent": 0,
    }

    t_start = time.perf_counter()
    t_warm_end = t_start + warmup_s
    t_end = t_start + duration_s

    if mode == "single":
        await run_single(client, imgs, out, rps_images, t_end, t_warm_end, sem, tracker, lat_ms, per_image_ms, counters)
    else:
        await run_client_batch(client, imgs, out, rps_images, t_end, t_warm_end, sem, tracker, lat_ms, per_image_ms, batch_hist, counters, batch_size, batch_window_ms)

    await tracker.drain()
    
    # NOTE: 'measured_s' uses the nominal (duration - warmup) window, and intentionally does not
    # include the post-window drain time spent waiting for in-flight RPCs to complete (tracker.drain()).
    # Under heavy load, drain time can be non-trivial; if you require strict accounting, measure
    # wall-clock time between warmup end and drain completion and use that as the denominator.

    measured_s = max(1e-9, duration_s - warmup_s)

    summary = {
        "config": {
            "mode": mode,
            "rps_images_target": float(rps_images),
            "duration_s": int(duration_s),
            "warmup_s": int(warmup_s),
            "concurrency": int(concurrency),
            "batch_size": int(batch_size if mode == "client-batch" else 1),
            "batch_window_ms": int(batch_window_ms if mode == "client-batch" else 0),
            "triton_grpc": TRITON_GRPC,
            "model": TRITON_MODEL,
            "output": EMB_LAYER,
            "seed": int(seed),
        },
        "counts": counters,
        "achieved": {
            "calls_per_s": counters["calls_ok"] / measured_s,
            "images_per_s": counters["images_ok"] / measured_s,
            "drop_rate": (counters["images_dropped"] / max(1, counters["images_enqueued"] + counters["images_dropped"])) if mode == "client-batch" else 0.0,
        },
        "stats_ms": {
            "per_call": np_stats_ms(lat_ms),
            "per_image_proxy": np_stats_ms(per_image_ms),
        },
    }

    if mode == "client-batch":
        # Batch-size diagnostics:
        # - batch_hist_top10 shows what batch sizes were *actually executed* by the client microbatcher.
        # - batch_mean is the empirical average batch size. If this remains near 1–3, then effective batching
        #   is limited by arrival rate and microbatch window, and throughput improvements should be attributed
        #   to hardware/replication rather than batching.
        top10 = batch_hist.most_common(10)
        summary["batch_hist_top10"] = [{"batch_size": int(bs), "count": int(c)} for bs, c in top10]
        if batch_hist:
            total_batches = sum(batch_hist.values())
            total_items = sum(bs * c for bs, c in batch_hist.items())
            summary["batch_mean"] = float(total_items / max(1, total_batches))

    print("\n=== Triton benchmark summary ===")
    print(summary)

# Parse command-line arguments and launch the asynchronous benchmark with
# the requested workload parameters.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "client-batch"], default="single")
    ap.add_argument("--rps", type=float, required=True, help="Target images/sec (open-loop).")

    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=50)

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--batch-window-ms", type=int, default=2)

    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    asyncio.run(run_bench(
        mode=args.mode,
        rps_images=args.rps,
        duration_s=args.duration,
        warmup_s=args.warmup,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        batch_window_ms=args.batch_window_ms,
        seed=args.seed,
    ))

# Standard command-line entry point.
if __name__ == "__main__":
    main()
