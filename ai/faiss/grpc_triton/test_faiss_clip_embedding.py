# This module defines an optional integration test for the FAISS gRPC
# service using a realistic end-to-end path: a face embedding is extracted
# from a video clip with InsightFace, ingested into the FAISS service
# through IngestBatch, and then queried again through Search to confirm
# that the stored vector can be retrieved as a self-match. The test is
# intended for functional integration checking rather than performance
# benchmarking. It can run either against an already running FAISS server
# or against a temporary local subprocess started specifically for the
# test.

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import grpc

import asyncio


RUN = os.getenv("RUN_FAISS_CLIP_TEST", "0") == "1"
CLIP = os.getenv("FAISS_TEST_CLIP", "").strip()
FAISS_ADDR = os.getenv("FAISS_ADDR", "").strip()     # e.g. "127.0.0.1:50051"
TEST_TOKEN = os.getenv("TEST_TOKEN", "").strip()     # must match INTERNAL_AUTH_TOKEN

# Locate and import the generated protobuf stubs from a small set of
# project-local paths so that the test can run from a checked-out source
# tree without requiring manual PYTHONPATH changes. This keeps the test
# easier to execute in development and continuous-integration settings.
def _import_stubs():
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

# Obtain an ephemeral local TCP port that can be used for a temporary FAISS
# subprocess started by the test when no external service address has been
# supplied.
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p

# Wait until the specified TCP port becomes reachable or raise an error
# after the timeout. This is used when the test starts a local subprocess
# and needs a simple readiness check before opening the gRPC channel.
def _wait_port(host: str, port: int, timeout_s: float = 10.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"port not ready: {host}:{port}")

# Execute an end-to-end integration test of the FAISS gRPC service using a
# face embedding extracted from a real video clip. The test supports two
# modes: it can connect to an already running FAISS server, or it can start
# a temporary local server subprocess with a minimal configuration. After
# extracting and normalising one embedding from the clip, the test ingests
# the vector, waits briefly for asynchronous index flushing, and then polls
# until the same vector is returned as a nearest-neighbour self-match.
@pytest.mark.asyncio
async def test_faiss_with_clip_embedding():
    if not RUN:
        pytest.fail("RUN_FAISS_CLIP_TEST is not set to 1. Export RUN_FAISS_CLIP_TEST=1 to run this test.")
    if not CLIP:
        pytest.fail("FAISS_TEST_CLIP is empty. Export FAISS_TEST_CLIP=/abs/path/to/clip.mp4")
    # Skip if your current proto/stubs do not include IngestBatch.
    

    import cv2
    from insightface.app import FaceAnalysis

    token = TEST_TOKEN or "test-token"

    # Decide whether to use an externally supplied FAISS service or to
    # launch a temporary local subprocess for the duration of the test. The
    # subprocess path keeps the test self-contained when no external target
    # is available.
    p = None
    if FAISS_ADDR:
        if not TEST_TOKEN:
            pytest.fail("FAISS_ADDR is set but TEST_TOKEN is empty. Set TEST_TOKEN to INTERNAL_AUTH_TOKEN of the running FAISS server.")
        target = FAISS_ADDR
    else:
        port = _free_port()
        target = f"127.0.0.1:{port}"

        faiss_srv = str(Path(__file__).resolve().parent / "faiss_grpc_server.py")
        env = os.environ.copy()
        env.update(
            {
                "INTERNAL_AUTH_TOKEN": token,
                "EMB_DIM": "512",
                "TAU_L2": "1.150",
                "USE_GPU": "0",
                "INDEX_TYPE": "flat",
                "BOOTSTRAP_MODE": "empty",
                "ALLOW_EMPTY_INDEX": "1",
                "ENABLE_UPDATES": "1",
                "UPDATE_BATCH_MS": "10",
                "UPDATE_BATCH_SIZE": "64",
                "UPDATE_MAX_QUEUE": "10000",
            }
        )

        p = subprocess.Popen(
            [sys.executable, faiss_srv, "--listen", target],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for the *actual* local port.
        _wait_port("127.0.0.1", port)

    try:
        # For an externally supplied service, readiness is checked through
        # the gRPC channel. For a locally spawned subprocess, the TCP port
        # has already been awaited before entering the test body.

        # Extract a single normalised face embedding from the supplied video
        # clip. The test accepts the first frame that contains exactly one
        # detected face with a valid embedding, which keeps the integration
        # path realistic while avoiding ambiguity from multi-face frames.
        fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(640, 640))

        cap = cv2.VideoCapture(str(CLIP))
        assert cap.isOpened(), f"Cannot open clip: {CLIP}"

        emb = None
        for _ in range(120):
            ok, frame = cap.read()
            if not ok:
                break
            faces = fa.get(frame)
            if len(faces) == 1 and getattr(faces[0], "embedding", None) is not None:
                e = faces[0].embedding.astype(np.float32, copy=False)
                emb = e / (np.linalg.norm(e) + 1e-12)
                emb = emb.astype(np.float32, copy=False)
                break
        cap.release()
        assert emb is not None, "Could not extract a single-face embedding from the clip."
        
        # Create the authenticated gRPC client used for ingestion and
        # search. The same internal-auth metadata is required whether the
        # target service is externally supplied or locally spawned.
        md = (("x-internal-auth", token),)
        ch = grpc.aio.insecure_channel(target)
        await ch.channel_ready()
        stub = dedup_pb2_grpc.FaissDedupStub(ch)
        
        # Python gRPC callables are attached to the stub instance, so the
        # capability check must be performed on the constructed object.
        if not hasattr(stub, "IngestBatch"):
            pytest.fail(
                "IngestBatch callable is missing on the stub instance. "
                "This usually means the wrong 'gen/' was imported or stubs were not regenerated."
            )

        # Submit the extracted embedding through the append-only ingest path
        # so that it becomes available for later nearest-neighbour search.
        # The test uses a fixed application-level identifier to make the
        # expected self-match explicit.
        req_ing = dedup_pb2.IngestBatchRequest(
            ids=[123],
            embeddings_f32=[emb.tobytes()],
        )
        try:
            _ = await stub.IngestBatch(req_ing, timeout=3.0, metadata=md)
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                pytest.fail("Server does not implement IngestBatch (UNIMPLEMENTED). Deploy a server build that includes IngestBatch.")
            raise
            
        await asyncio.sleep(0.2)

        # Poll for a self-match rather than issuing a single immediate
        # search, because the service may flush queued updates to the index
        # asynchronously. The test therefore checks eventual visibility of
        # the ingested vector rather than assuming synchronous durability.
        req = dedup_pb2.SearchRequest(query_id=1, embedding_f32=emb.tobytes())

        r = None
        for _ in range(25):  # ~5s total
            r = await stub.Search(req, timeout=3.0, metadata=md)
            if r.nearest_id == 123 and r.is_match:
                break
            await asyncio.sleep(0.2)

        assert r is not None
        assert r.nearest_id == 123
        assert r.is_match is True
        assert float(r.distance_l2) <= 2e-2

        await ch.close()
    # Ensure that any locally spawned FAISS subprocess is terminated even
    # if the test fails partway through, so that repeated test runs do not
    # leave orphaned background processes.
    finally:
        if p is not None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
