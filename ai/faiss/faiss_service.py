"""
faiss_service.py
----------------
Election-scoped FAISS similarity service (INTERNAL ONLY).

This is the deployable successor of your old benchmark-only `faiss_ms (1).py`.
It preserves the old harness endpoints (/search, /search_batch, /ping) but adds:
- Explicit internal authentication (fail-closed)
- Stable "policy snapshot" hashing (RFC8785 JCS + SHA-256) for audit/public dashboard
- Optional Ed25519 signing of the policy snapshot (for publication)
- Optional encrypted snapshots of the FAISS index (Fernet) for restore/restart
- Optional batched background updates (for benchmark realism), without forcing it in election mode

Security boundary (per UI + audit docs)
---------------------------------------
- This service MUST NOT be Internet-exposed.
- Only trusted backend services (e.g., LVS / gateway) may call it.
- Do not log embeddings/vectors.
- Do not return neighbor IDs or distances to client devices; those are internal only.

Run:
  uvicorn faiss_service:app --host 0.0.0.0 --port 9010
"""

# Implementation note
# -------------------
# The service is intentionally conservative and audit-friendly:
#   • The FAISS index is treated as an internal component; vectors must never be logged.
#   • Runtime behaviour is controlled via environment variables to support reproducible
#     deployments (and verifiable policy snapshots).
#   • Election deployments are expected to run in 'static' mode (ENABLE_UPDATES=0).
#     Update paths remain available for controlled benchmark experiments only.


from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import faiss
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# Optional deps for strict RFC8785 canonical JSON
try:  # pragma: no cover
    import jcs  # python-jcs
except Exception:  # pragma: no cover
    jcs = None

# Optional deps for snapshot encryption and policy signing
try:  # pragma: no cover
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except Exception:  # pragma: no cover
    Fernet = None
    serialization = None
    Ed25519PrivateKey = None


# ------------------------ Configuration ------------------------ #
EMB_DIM = int(os.getenv("EMB_DIM", "512"))

# Calibrated L2 threshold (your baseline)
TAU_L2 = float(os.getenv("TAU_L2", "1.150"))
TAU_SQ = TAU_L2 * TAU_L2

# App-level internal auth (fail-closed)
INTERNAL_AUTH_TOKEN = os.getenv("INTERNAL_AUTH_TOKEN", "").strip()
# Fail-closed security posture: the process must refuse to start without an internal token.
if not INTERNAL_AUTH_TOKEN:
    raise RuntimeError("INTERNAL_AUTH_TOKEN must be set (fail-closed).")

# Index build/serve

# Index serving strategy
# ----------------------
# The service can optionally clone a CPU index to GPU for low-latency *exact* search.
# GPU use is restricted to FlatL2 in this implementation; HNSW serves on CPU.
USE_GPU = os.getenv("USE_GPU", "1") == "1"
GPU_ID = int(os.getenv("GPU_ID", "0"))
INDEX_TYPE = os.getenv("INDEX_TYPE", "flat").lower()  # flat|hnsw (hnsw optional)

# HNSW knobs (only if INDEX_TYPE=hnsw)
HNSW_M = int(os.getenv("HNSW_M", "32"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "64"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))

# Bootstrap (bench realism; keep off for election deployments)

# Bootstrap mode is intended for benchmark realism. For elections, keep BOOTSTRAP_MODE=empty.
BOOTSTRAP_MODE = os.getenv("BOOTSTRAP_MODE", "empty").lower()  # empty|random
RANDOM_N = int(os.getenv("RANDOM_N", "1000000"))
RANDOM_CHUNK = int(os.getenv("RANDOM_CHUNK", "100000"))  # chunked bootstrap to avoid huge peak RAM

# Updates:
# - In election mode: keep ENABLE_UPDATES=0 (static registry).
# - In benchmark mode: you may enable updates and (optionally) enqueue-on-search to mimic growth.

# Update policy
# -------------
# In the paper's threat model, election operation should be side-effect free; therefore
# updates are disabled by default and must be explicitly enabled for benchmarks.
ENABLE_UPDATES = os.getenv("ENABLE_UPDATES", "0") == "1"
BATCH_UPDATE_SIZE = int(os.getenv("BATCH_UPDATE_SIZE", "4096"))
BATCH_UPDATE_MS = int(os.getenv("BATCH_UPDATE_MS", "100"))
LEGACY_ENQUEUE_ON_SEARCH = os.getenv("LEGACY_ENQUEUE_ON_SEARCH", "0") == "1"

# Snapshot persistence (optional)
SNAPSHOT_PATH = os.getenv("SNAPSHOT_PATH", "").strip()
SNAPSHOT_FERNET_KEY = os.getenv("SNAPSHOT_FERNET_KEY", "").strip()  # optional; base64 urlsafe key for Fernet

# Optional: sign policy snapshot for publication
POLICY_SIGNING_ED25519_PRIV_B64 = os.getenv("POLICY_SIGNING_ED25519_PRIV_B64", "").strip()  # base64(32 bytes)

# Lightweight latency metrics
METRICS_WINDOW = int(os.getenv("METRICS_WINDOW", "2048"))


# ------------------------ Canonical JSON + hashing ------------------------ #
def _jcs_bytes(obj) -> bytes:
    """
    RFC8785 JCS canonicalization (preferred). If python-jcs is unavailable, use a strict
    deterministic JSON fallback (sufficient for controlled objects).
    """
    if jcs is not None:
        return jcs.canonicalize(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _maybe_sign_policy(canon: bytes) -> dict:
    """
    Optionally sign canonical policy bytes using Ed25519.

    Returns {} if signing is not configured or cryptography is unavailable.
    """
    if not POLICY_SIGNING_ED25519_PRIV_B64 or Ed25519PrivateKey is None or serialization is None:
        return {}
    try:
        seed = base64.b64decode(POLICY_SIGNING_ED25519_PRIV_B64)
        if len(seed) != 32:
            return {}
        sk = Ed25519PrivateKey.from_private_bytes(seed)
        sig = sk.sign(canon)
        pk = sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return {
            "policy_sig_alg": "Ed25519",
            "policy_sig_b64": base64.b64encode(sig).decode("ascii"),
            "policy_sig_pub_b64": base64.b64encode(pk).decode("ascii"),
        }
    except Exception:
        return {}


# ------------------------ Snapshot encryption (optional) ------------------------ #
def _fernet() -> Optional["Fernet"]:
    if not SNAPSHOT_FERNET_KEY:
        return None
    if Fernet is None:
        raise RuntimeError("cryptography is required when SNAPSHOT_FERNET_KEY is set.")
    # Fernet expects a urlsafe base64-encoded 32-byte key.
    return Fernet(SNAPSHOT_FERNET_KEY.encode("utf-8"))


def save_snapshot(index_cpu: faiss.Index, path: str) -> None:
    """
    Serialize the CPU FAISS index to bytes, optionally encrypt, and atomically write to disk.
    """
    arr = faiss.serialize_index(index_cpu)
    raw = bytes(arr)
    f = _fernet()
    blob = f.encrypt(raw) if f else raw
    tmp = path + ".tmp"
    with open(tmp, "wb") as w:
        w.write(blob)
    os.replace(tmp, path)


def load_snapshot(path: str) -> Optional[faiss.Index]:
    """
    Load (and optionally decrypt) a serialized FAISS index from disk.
    Returns None if file does not exist.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as r:
        blob = r.read()
    f = _fernet()
    raw = f.decrypt(blob) if f else blob
    return faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8))


# ------------------------ Pydantic models ------------------------ #
class QueryVector(BaseModel):
    """
    Internal query object.

    IMPORTANT: do not pass real voter identifiers.
    Use an election-scoped pseudonym (e.g., HMAC(EID||roll_id)).
    """
    subject_ref: str = Field(..., description="Pseudonymous reference (no real voter ID).")
    vector_b64_f32: Optional[str] = Field(
        None,
        description="Embedding as base64-encoded float32 bytes (len=EMB_DIM*4). Preferred for speed.",
    )
    vector: Optional[List[float]] = Field(
        None,
        description="Embedding as float list (slower; JSON-heavy).",
    )


class QueryBatch(BaseModel):
    subject_refs: List[str]
    vectors_b64_f32: Optional[List[str]] = None
    vectors: Optional[List[List[float]]] = None


class UpsertBatch(BaseModel):
    """
    Add vectors with explicit IDs.
    Recommended for initial preload (registry build) rather than enqueue-on-search.
    """
    ids: List[int]
    vectors_b64_f32: Optional[List[str]] = None
    vectors: Optional[List[List[float]]] = None


# ------------------------ Index manager ------------------------ #
@dataclass
class FaissManager:
    """
    Keep CPU index as the durable source-of-truth (snapshots, deterministic restore).
    Optionally keep a GPU clone for low-latency search.
    """
    index_cpu: faiss.Index
    index_srv: faiss.Index
    use_gpu: bool

    def ntotal(self) -> int:
        return int(self.index_srv.ntotal)


def _make_cpu_index() -> faiss.Index:
    """
    Create the baseline CPU index.

    Baseline (paper/UI): Flat L2 with explicit IDs (IndexIDMap2(IndexFlatL2)).
    Optional: HNSW (IndexIDMap2(IndexHNSWFlat)) for speed/scale experiments.
    """
    if INDEX_TYPE == "hnsw":
        base = faiss.IndexHNSWFlat(EMB_DIM, HNSW_M)
        base.hnsw.efSearch = HNSW_EF_SEARCH
        base.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        return faiss.IndexIDMap2(base)

    base = faiss.IndexFlatL2(EMB_DIM)
    return faiss.IndexIDMap2(base)


def _clone_to_gpu(cpu_index: faiss.Index) -> Tuple[faiss.Index, bool]:
    """
    Attempt to clone a CPU index to GPU. Falls back to CPU if GPU is unavailable or unsupported.
    """
    if not USE_GPU:
        return cpu_index, False

    # HNSW GPU cloning is not generally supported; serve on CPU for HNSW.
    if INDEX_TYPE != "flat":
        return cpu_index, False

    try:
        res = faiss.StandardGpuResources()

        # Prefer float16 storage on GPU for memory + bandwidth wins.
        # The exact API varies by faiss build; handle both signatures.
        try:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, GPU_ID, cpu_index, co)
        except Exception:
            gpu_index = faiss.index_cpu_to_gpu(res, GPU_ID, cpu_index)

        # Warm-up to avoid first-request latency spike.
        _ = gpu_index.search(np.zeros((1, EMB_DIM), dtype=np.float32), 1)
        return gpu_index, True
    except Exception:
        return cpu_index, False


def _bootstrap_random(index_cpu: faiss.Index) -> None:
    """
    Optional: populate the index with random vectors (benchmark realism).

    Uses chunked generation to avoid very large peak memory.
    """
    if BOOTSTRAP_MODE != "random" or RANDOM_N <= 0:
        return

    rng = np.random.default_rng(seed=42)
    start = 0
    while start < RANDOM_N:
        n = min(RANDOM_CHUNK, RANDOM_N - start)
        vecs = rng.standard_normal((n, EMB_DIM), dtype=np.float32)
        ids = np.arange(start, start + n, dtype=np.int64)
        index_cpu.add_with_ids(vecs, ids)
        start += n


def build_manager() -> FaissManager:
    # 1) Load snapshot (if provided)
    cpu_idx = None
    if SNAPSHOT_PATH:
        cpu_idx = load_snapshot(SNAPSHOT_PATH)
        if cpu_idx is not None and not isinstance(cpu_idx, faiss.IndexIDMap2):
            raise RuntimeError("Snapshot must be an IndexIDMap2.")

    # 2) Else build empty CPU index
    if cpu_idx is None:
        cpu_idx = _make_cpu_index()

    # 3) Optional bootstrap (bench-only)
    if cpu_idx.ntotal == 0:
        _bootstrap_random(cpu_idx)

    # 4) Clone to GPU (if configured/supported)
    srv_idx, use_gpu = _clone_to_gpu(cpu_idx)
    return FaissManager(index_cpu=cpu_idx, index_srv=srv_idx, use_gpu=use_gpu)


# ------------------------ Lightweight latency stats ------------------------ #
class LatencyStats:
    def __init__(self, window: int):
        self._q: Deque[float] = deque(maxlen=window)

    def add_ms(self, ms: float) -> None:
        self._q.append(float(ms))

    def summary(self) -> dict:
        if not self._q:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        xs = sorted(self._q)
        n = len(xs)
        return {
            "count": n,
            "p50_ms": xs[int(0.50 * (n - 1))],
            "p95_ms": xs[int(0.95 * (n - 1))],
            "max_ms": xs[-1],
        }


# ------------------------ FastAPI app ------------------------ #
app = FastAPI(title="FAISS Similarity Service (Internal)", version="1.1")
mgr = build_manager()

lat_search = LatencyStats(METRICS_WINDOW)
lat_upsert = LatencyStats(METRICS_WINDOW)

# When updates are enabled, we apply a single lock around all FAISS mutations/searches
# to avoid undefined behavior in some FAISS builds under concurrent add+search.
# Concurrency control: FAISS add/search is not uniformly thread-safe across builds; we
# serialise operations when updates are enabled to avoid undefined behaviour.
_faiss_lock = asyncio.Lock()

# Optional background update queue (bench realism)
# Background updates are queued to decouple ingestion from FAISS mutation in benchmark mode.
_update_q: "asyncio.Queue[Tuple[np.ndarray, np.ndarray]]" = asyncio.Queue()  # (vecs_f32, ids_i64)


async def require_internal_auth(req: Request):
    tok = req.headers.get("x-internal-auth", "")
    # Starlette normalises header keys; lookup is case-insensitive for incoming requests.
    if tok != INTERNAL_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")


@app.exception_handler(Exception)
async def _err_handler(request: Request, exc: Exception):
    # Avoid leaking internal details.
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": "internal_error"})


@app.on_event("startup")
async def _startup():
    # Background batch flusher (only when ENABLE_UPDATES=1)
    if ENABLE_UPDATES:
        asyncio.create_task(_flush_updates())


async def _flush_updates():
    """
    Flush enqueued updates in batches. Mirrors the old faiss_ms behavior, but keeps
    it explicitly optional for election deployments.

    IMPORTANT:
    - Updates are off by default.
    - If you enable updates, keep search+add serialized via _faiss_lock.
    """
    while True:
        await asyncio.sleep(BATCH_UPDATE_MS / 1000.0)

        # Drain up to BATCH_UPDATE_SIZE vectors per flush.
        vecs_list = []
        ids_list = []
        while (not _update_q.empty()) and (len(ids_list) < BATCH_UPDATE_SIZE):
            v, ids = _update_q.get_nowait()
            vecs_list.append(v)
            ids_list.append(ids)

        if not ids_list:
            continue

        vecs = np.vstack(vecs_list).astype(np.float32, copy=False)
        ids = np.concatenate(ids_list).astype(np.int64, copy=False)

        t0 = time.perf_counter()
        async with _faiss_lock:
        # CPU index is the durable source-of-truth; GPU clone is updated in lockstep when enabled.
            mgr.index_cpu.add_with_ids(vecs, ids)
            if mgr.use_gpu:
                mgr.index_srv.add_with_ids(vecs, ids)
        lat_upsert.add_ms((time.perf_counter() - t0) * 1000.0)

        # Optional snapshot after flush (bench convenience)
        if SNAPSHOT_PATH:
            try:
                save_snapshot(mgr.index_cpu, SNAPSHOT_PATH)
            except Exception:
                # Snapshot is operational; never fail service for it.
                pass


# ------------------------ Vector parsing ------------------------ #
def _vec_from_payload(q: QueryVector) -> np.ndarray:
    if q.vector_b64_f32:
        raw = base64.b64decode(q.vector_b64_f32)
        if len(raw) != EMB_DIM * 4:
            raise HTTPException(400, f"vector_b64_f32 must decode to {EMB_DIM * 4} bytes")
        return np.frombuffer(raw, dtype=np.float32).reshape(1, -1)

    if q.vector is None:
        raise HTTPException(400, "Provide either vector or vector_b64_f32")
    if len(q.vector) != EMB_DIM:
        raise HTTPException(400, f"vector must be length {EMB_DIM}")
    return np.asarray(q.vector, dtype=np.float32).reshape(1, -1)


def _vecs_from_batch(b: QueryBatch) -> np.ndarray:
    if b.vectors_b64_f32 is not None:
        if len(b.vectors_b64_f32) != len(b.subject_refs):
            raise HTTPException(400, "length mismatch")
        arrs = []
        for s in b.vectors_b64_f32:
            raw = base64.b64decode(s)
            if len(raw) != EMB_DIM * 4:
                raise HTTPException(400, f"each vectors_b64_f32 must decode to {EMB_DIM * 4} bytes")
            arrs.append(np.frombuffer(raw, dtype=np.float32))
        return np.stack(arrs, axis=0).astype(np.float32, copy=False)

    if b.vectors is None:
        raise HTTPException(400, "Provide either vectors or vectors_b64_f32")
    if len(b.vectors) != len(b.subject_refs):
        raise HTTPException(400, "length mismatch")

    arr = np.asarray(b.vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != EMB_DIM:
        raise HTTPException(400, f"each vector must be length {EMB_DIM}")
    return arr


# ------------------------ Policy snapshot ------------------------ #
def _policy_obj() -> dict:
    return {
        "service": "faiss_similarity",
        "version": "1.1",
        "canonicalization": ("RFC8785-JCS" if jcs is not None else "deterministic-json-fallback"),
        "hash": "SHA-256",
        "emb_dim": EMB_DIM,
        "metric": "L2",
        "tau_l2": TAU_L2,
        "tau_sq": TAU_SQ,
        "index_type": INDEX_TYPE,
        "use_gpu": bool(mgr.use_gpu),
        "updates_enabled": bool(ENABLE_UPDATES),
    }


# ------------------------ Endpoints (v1) ------------------------ #
@app.get("/v1/health")
async def v1_health():
    return {"status": "ok", "ntotal": mgr.ntotal(), "index_type": INDEX_TYPE, "gpu": mgr.use_gpu}


@app.get("/v1/policy", dependencies=[Depends(require_internal_auth)])
async def v1_policy():
    obj = _policy_obj()
    canon = _jcs_bytes(obj)
    out = {"policy": obj, "policy_hash_sha256": _sha256_hex(canon)}
    out.update(_maybe_sign_policy(canon))
    return out


@app.get("/v1/metrics", dependencies=[Depends(require_internal_auth)])
async def v1_metrics():
    return {
        "ntotal": mgr.ntotal(),
        "search_latency": lat_search.summary(),
        "upsert_latency": lat_upsert.summary(),
        "update_queue_depth": int(_update_q.qsize()) if ENABLE_UPDATES else 0,
    }


@app.post("/v1/search", dependencies=[Depends(require_internal_auth)])
async def v1_search(q: QueryVector):
    vec = _vec_from_payload(q)

    # Measure FAISS kernel time only (exclude any lock wait time).
    if ENABLE_UPDATES:
        async with _faiss_lock:
            t0 = time.perf_counter()
            D, I = await run_in_threadpool(mgr.index_srv.search, vec, 1)
            dt_ms = (time.perf_counter() - t0) * 1000.0
    else:
        t0 = time.perf_counter()
        D, I = await run_in_threadpool(mgr.index_srv.search, vec, 1)
        dt_ms = (time.perf_counter() - t0) * 1000.0
    lat_search.add_ms(dt_ms)

    dist_sq = float(D[0][0])
    nn_id = int(I[0][0])

    if nn_id < 0 or (not math.isfinite(dist_sq)):
        dist_sq = float("inf")
        dist_l2 = float("inf")
        is_match = False
    else:
        dist_l2 = float(math.sqrt(dist_sq))
        is_match = bool(dist_l2 <= TAU_L2)

    # IMPORTANT: election deployments should keep search side-effect free.
    # For benchmarks only, you may opt-in to enqueue the query vector as an update.
    if ENABLE_UPDATES and LEGACY_ENQUEUE_ON_SEARCH:
        await _update_q.put((vec.astype(np.float32, copy=False), np.array([0], dtype=np.int64)))  # placeholder

    return {
        "nearest_id": nn_id,
        "distance_sq": dist_sq,
        "distance_l2": dist_l2,
        "is_match": is_match,
        "faiss_search_time_ms": round(dt_ms, 3),
    }


@app.post("/v1/search_batch", dependencies=[Depends(require_internal_auth)])
async def v1_search_batch(b: QueryBatch):
    vecs = _vecs_from_batch(b)

    # Measure FAISS kernel time only (exclude any lock wait time).
    if ENABLE_UPDATES:
        async with _faiss_lock:
            t0 = time.perf_counter()
            # Note: this call is synchronous. In-process tests use this directly; in production
            # deployments, uvicorn workers or a threadpool can be used to scale CPU concurrency.
            D, I = mgr.index_srv.search(vecs, 1)
            dt_ms = (time.perf_counter() - t0) * 1000.0
    else:
        t0 = time.perf_counter()
            # Note: this call is synchronous. In-process tests use this directly; in production
            # deployments, uvicorn workers or a threadpool can be used to scale CPU concurrency.
        D, I = mgr.index_srv.search(vecs, 1)
        dt_ms = (time.perf_counter() - t0) * 1000.0
    lat_search.add_ms(dt_ms)

    items = []
    for subj, d, i in zip(b.subject_refs, D, I):
        dist_sq = float(d[0])
        nn_id = int(i[0])
        if nn_id < 0 or (not math.isfinite(dist_sq)):
            dist_sq = float("inf")
            dist_l2 = float("inf")
            is_match = False
        else:
            dist_l2 = float(math.sqrt(dist_sq))
            is_match = bool(dist_l2 <= TAU_L2)
        items.append(
            {
                "subject_ref": subj,
                "nearest_id": nn_id,
                "distance_sq": dist_sq,
                "distance_l2": dist_l2,
                "is_match": is_match,
            }
        )

    return {"items": items, "faiss_search_time_ms": round(dt_ms, 3)}


@app.post("/v1/upsert_batch", dependencies=[Depends(require_internal_auth)])
async def v1_upsert_batch(u: UpsertBatch):
    if not ENABLE_UPDATES:
        raise HTTPException(403, "updates_disabled")
    if not u.ids:
        return {"added": 0, "ntotal": mgr.ntotal()}

    # Parse vectors (b64 preferred)
    if u.vectors_b64_f32 is not None:
        if len(u.vectors_b64_f32) != len(u.ids):
            raise HTTPException(400, "length mismatch")
        arrs = []
        for s in u.vectors_b64_f32:
            raw = base64.b64decode(s)
            if len(raw) != EMB_DIM * 4:
                raise HTTPException(400, f"each vectors_b64_f32 must decode to {EMB_DIM * 4} bytes")
            arrs.append(np.frombuffer(raw, dtype=np.float32))
        vecs = np.stack(arrs, axis=0).astype(np.float32, copy=False)
    elif u.vectors is not None:
        if len(u.vectors) != len(u.ids):
            raise HTTPException(400, "length mismatch")
        vecs = np.asarray(u.vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != EMB_DIM:
            raise HTTPException(400, f"each vector must be length {EMB_DIM}")
    else:
        raise HTTPException(400, "Provide either vectors or vectors_b64_f32")

    ids = np.asarray(u.ids, dtype=np.int64)

    # Add (serialized for safety when updates enabled)
    t0 = time.perf_counter()
    async with _faiss_lock:
        # CPU index is the durable source-of-truth; GPU clone is updated in lockstep when enabled.
        mgr.index_cpu.add_with_ids(vecs, ids)
        if mgr.use_gpu:
            mgr.index_srv.add_with_ids(vecs, ids)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    lat_upsert.add_ms(dt_ms)

    if SNAPSHOT_PATH:
        try:
            save_snapshot(mgr.index_cpu, SNAPSHOT_PATH)
        except Exception:
            pass

    return {"added": int(len(ids)), "ntotal": mgr.ntotal(), "upsert_time_ms": round(dt_ms, 3)}


# ------------------------ Backward-compatible endpoints (legacy harness) ------------------------ #
# These preserve your old benchmark interface from faiss_ms (1).py.
class LegacyQueryVector(BaseModel):
    voter_id: int
    vector: List[float]


class LegacyQueryBatch(BaseModel):
    voter_ids: List[int]
    vectors: List[List[float]]


@app.post("/search", dependencies=[Depends(require_internal_auth)])
async def legacy_search(q: LegacyQueryVector):
    # Preserve old behavior: squared L2 distance reported as "distance"
    qq = QueryVector(subject_ref=str(q.voter_id), vector=q.vector)
    r = await v1_search(qq)

    # Optional bench behavior: enqueue an "insert" mirroring the old service.
    # Not recommended for election deployments.
    if ENABLE_UPDATES and LEGACY_ENQUEUE_ON_SEARCH:
        vec = np.asarray(q.vector, dtype=np.float32).reshape(1, -1)
        t2 = time.perf_counter()
        await _update_q.put((vec, np.array([q.voter_id], dtype=np.int64)))
        update_time_ms = (time.perf_counter() - t2) * 1000.0
    else:
        update_time_ms = 0.0

    return {
        "nearest_id": r["nearest_id"],
        "distance": r["distance_sq"],
        "distance_sq": r["distance_sq"],
        "distance_l2": r["distance_l2"],
        "faiss_search_time_ms": r["faiss_search_time_ms"],
        "faiss_index_update_time_ms": round(update_time_ms, 3),  # legacy field (queue put only)
    }


@app.post("/search_batch", dependencies=[Depends(require_internal_auth)])
async def legacy_search_batch(b: LegacyQueryBatch):
    bb = QueryBatch(subject_refs=[str(x) for x in b.voter_ids], vectors=b.vectors)
    r = await v1_search_batch(bb)

    # Optional bench behavior: enqueue inserts in batch
    if ENABLE_UPDATES and LEGACY_ENQUEUE_ON_SEARCH:
        vecs = np.asarray(b.vectors, dtype=np.float32)
        ids = np.asarray(b.voter_ids, dtype=np.int64)
        await _update_q.put((vecs, ids))

    out = []
    for vid, it in zip(b.voter_ids, r["items"]):
        out.append(
            {
                "voter_id": vid,
                "nearest_id": it["nearest_id"],
                "distance": it["distance_sq"],  # legacy: squared L2
                "distance_sq": it["distance_sq"],
                "distance_l2": it["distance_l2"],
            }
        )
    return out


@app.get("/ping")
async def ping():
    # No auth, for local container probes only.
    return {"status": "ok", "ntotal": mgr.ntotal()}
