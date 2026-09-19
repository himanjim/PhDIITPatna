#!/usr/bin/env python3
"""
face_embedding_backend.py

Shared embedding backend for the new InsightFace + FAISS accuracy experiments.

The implementation intentionally follows the geometry already used in the PhD
repository:

    detect -> align -> raccoon_l / w600k_r50
    -> 512-D embedding -> L2 normalise -> FAISS IndexFlatL2

Two execution paths are retained:

1. local
   Uses InsightFace FaceAnalysis directly. This is the easiest mode for Google
   Colab or a workstation and is appropriate for dry runs and accuracy checks.

2. triton
   Uses InsightFace only for SCRFD detection/alignment, then sends the aligned
   crop to the same Triton model interface used by calibrate_threshold.py:
       model       = raccoon_l (only if you explicitly deploy it in Triton)
       input name  = input.1
       output name = 683
   For the present Windows accuracy experiments, use the local InsightFace 2.0 path.
   The Triton path is retained only for compatibility with earlier repository work.

FAISS returns SQUARED Euclidean distance. If the operational true-L2 threshold
is tau_L2=1.15, compare the FAISS distance against tau_sq = 1.15**2 = 1.3225.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Windows/CUDA robustness:
# ONNX Runtime >=1.21 can preload CUDA/cuDNN DLLs installed by the
# onnxruntime-gpu[cuda,cudnn] package.  Calling this before InsightFace creates
# any sessions avoids common Windows 'CUDAExecutionProvider not available'
# / missing-DLL fallbacks.  On systems where the function or NVIDIA site
# packages are absent, we simply continue and let ONNX Runtime use its normal
# DLL search path.
try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
        except Exception:
            try:
                ort.preload_dlls()
            except Exception:
                pass
except Exception:
    ort = None

from insightface.app import FaceAnalysis
from insightface.utils import face_align

try:
    from tritonclient.grpc import (
        InferenceServerClient,
        InferInput,
        InferRequestedOutput,
    )
except Exception:
    # Triton is optional when --backend local is used.
    InferenceServerClient = None
    InferInput = None
    InferRequestedOutput = None


IMAGE_SIZE = (112, 112)
DET_SIZE = (640, 640)
EMBEDDING_DIM = 512


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a float32 unit-length embedding."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= eps:
        raise ValueError("Embedding norm is zero or non-finite.")
    return (v / n).astype(np.float32, copy=False)


def largest_face(faces):
    """
    Match calibrate_threshold.py: if more than one face is detected, use the
    largest face rather than silently taking an arbitrary detection.
    """
    if not faces:
        return None
    return max(
        faces,
        key=lambda x: float((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])),
    )


class FaceEmbedder:
    """Small wrapper that keeps local and Triton experiments behaviorally aligned."""

    def __init__(
        self,
        backend: str = "local",
        providers: Optional[Sequence[str]] = None,
        det_size: Tuple[int, int] = DET_SIZE,
        triton_url: str = "localhost:8001",
        model_name: str = "raccoon_l",
        input_name: str = "input.1",
        output_name: str = "683",
        use_rgb_before_triton: bool = False,
        strict_one_face: bool = False,
    ):
        self.backend = backend.lower().strip()
        if self.backend not in {"local", "triton"}:
            raise ValueError("backend must be 'local' or 'triton'")

        self.det_size = tuple(det_size)
        self.triton_url = triton_url
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.use_rgb_before_triton = bool(use_rgb_before_triton)
        self.strict_one_face = bool(strict_one_face)

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # When CUDA is explicitly requested, fail before loading the model if the
        # provider is unavailable.  This prevents an unnoticed CPU fallback from
        # being reported as a GTX 1650 experiment.
        if (
            self.backend == "local"
            and "CUDAExecutionProvider" in providers
            and ort is not None
            and "CUDAExecutionProvider" not in ort.get_available_providers()
        ):
            raise RuntimeError(
                "CUDAExecutionProvider is not available. Run check_environment.py "
                "or rerun the experiment with --cpu."
            )

        if self.backend == "local":
            # FaceAnalysis 2.0 performs the raccoon_l detector/alignment and w600k_r50
            # recognition internally and exposes both embedding and
            # normed_embedding.
            self.app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection", "recognition"],
                providers=list(providers),
            )
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            self.triton = None
        else:
            # This exactly follows calibrate_threshold.py: detection/alignment
            # is local on CPU; recognition is delegated to Triton.
            self.app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=-1, det_size=self.det_size)

            if InferenceServerClient is None:
                raise RuntimeError(
                    "tritonclient is not installed. "
                    "Install with: pip install 'tritonclient[grpc]'"
                )
            self.triton = InferenceServerClient(self.triton_url)

            # Fail early if the service/model is not ready.
            if not self.triton.is_server_live():
                raise RuntimeError(f"Triton server is not live at {self.triton_url}")
            if not self.triton.is_server_ready():
                raise RuntimeError(f"Triton server is not ready at {self.triton_url}")
            if not self.triton.is_model_ready(self.model_name):
                raise RuntimeError(
                    f"Triton model '{self.model_name}' is not ready."
                )

    def _detect(self, img_bgr: np.ndarray):
        """Detect faces and enforce the chosen face-selection policy."""
        faces = self.app.get(img_bgr)

        if self.strict_one_face:
            if len(faces) != 1:
                return None, len(faces)
            return faces[0], 1

        return largest_face(faces), len(faces)

    def _triton_preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Mirror calibrate_threshold.py:
          resize 112x112 -> optional BGR->RGB -> float32
          -> (pixel - 127.5)/128.0 -> CHW
        """
        img = cv2.resize(aligned_bgr, IMAGE_SIZE)

        if self.use_rgb_before_triton:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        return img.transpose(2, 0, 1).astype(np.float32, copy=False)

    def _triton_embed(self, chw: np.ndarray) -> np.ndarray:
        """Send one aligned face to the explicitly configured Triton recognition model."""
        batch = np.expand_dims(chw, axis=0).astype(np.float32, copy=False)

        inp = InferInput(self.input_name, batch.shape, "FP32")
        inp.set_data_from_numpy(batch)
        out = InferRequestedOutput(self.output_name)

        result = self.triton.infer(
            self.model_name,
            inputs=[inp],
            outputs=[out],
        )
        vec = result.as_numpy(self.output_name)
        if vec is None or vec.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected Triton output shape for '{self.output_name}': "
                f"{None if vec is None else vec.shape}"
            )
        return l2_normalize(vec[0])

    def embed_path(self, image_path: str | Path):
        """
        Return:
            (embedding_or_None, metadata_dict)

        A skipped image is represented as embedding=None rather than being
        assigned a fabricated vector. This keeps detection failures visible in
        the final experiment report.
        """
        image_path = str(image_path)
        img = cv2.imread(image_path)
        if img is None:
            return None, {
                "path": image_path,
                "status": "read_failed",
                "faces_detected": 0,
            }

        face, n_faces = self._detect(img)
        if face is None:
            return None, {
                "path": image_path,
                "status": "face_policy_failed",
                "faces_detected": int(n_faces),
            }

        try:
            if self.backend == "local":
                # normed_embedding is preferred. If unavailable, explicitly
                # normalise the raw embedding as done in test_faiss_clip_embedding.py.
                v = getattr(face, "normed_embedding", None)
                if v is None:
                    v = getattr(face, "embedding", None)
                if v is None:
                    raise RuntimeError("InsightFace face object has no embedding.")
                emb = l2_normalize(v)
            else:
                # Use exactly the 5-point alignment method in
                # calibrate_threshold.py before Triton recognition.
                aligned = face_align.norm_crop(
                    img,
                    landmark=face.kps,
                    image_size=IMAGE_SIZE[0],
                )
                chw = self._triton_preprocess(aligned)
                emb = self._triton_embed(chw)

            if emb.shape[0] != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIM}-D embedding, got {emb.shape}"
                )

            return emb, {
                "path": image_path,
                "status": "ok",
                "faces_detected": int(n_faces),
            }
        except Exception as exc:
            return None, {
                "path": image_path,
                "status": f"embedding_failed:{type(exc).__name__}",
                "faces_detected": int(n_faces),
            }


def save_embedding_cache(
    cache_path: str | Path,
    embeddings: Dict[str, np.ndarray],
) -> None:
    """
    Save a path->embedding cache as a compressed NPZ file.

    Paths are stored exactly as strings. Re-running the experiment against the
    same extracted dataset can therefore skip the expensive face inference step.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(embeddings.keys())
    X = np.vstack([embeddings[p] for p in paths]).astype(np.float32)
    np.savez_compressed(
        cache_path,
        paths=np.asarray(paths, dtype=object),
        embeddings=X,
    )


def load_embedding_cache(cache_path: str | Path) -> Dict[str, np.ndarray]:
    """Load a cache produced by save_embedding_cache()."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}

    d = np.load(cache_path, allow_pickle=True)
    paths = d["paths"].tolist()
    X = d["embeddings"].astype(np.float32)

    return {
        str(p): l2_normalize(v)
        for p, v in zip(paths, X)
    }


def embed_paths_with_cache(
    paths: Iterable[str | Path],
    embedder: FaceEmbedder,
    cache_path: str | Path,
    recompute: bool = False,
    progress_every: int = 100,
):
    """
    Embed all requested images, reusing any cached vectors.

    Returns:
        embeddings: dict[path] -> 512-D unit vector
        failures:   list of metadata dictionaries for skipped images
    """
    requested = [str(Path(p)) for p in paths]

    cache = {} if recompute else load_embedding_cache(cache_path)
    embeddings: Dict[str, np.ndarray] = {}
    failures: List[dict] = []

    for i, p in enumerate(requested, 1):
        if p in cache:
            embeddings[p] = cache[p]
        else:
            emb, meta = embedder.embed_path(p)
            if emb is None:
                failures.append(meta)
            else:
                embeddings[p] = emb
                cache[p] = emb

        if i % progress_every == 0 or i == len(requested):
            print(
                f"[EMBED] processed={i}/{len(requested)} "
                f"ok={len(embeddings)} failures={len(failures)}"
            )

    # Save the complete cache, including vectors from earlier runs.
    if cache:
        save_embedding_cache(cache_path, cache)

    return embeddings, failures


def wilson_interval(k: int, n: int, z: float = 1.959963984540054):
    """95% Wilson interval for a binomial proportion, without scipy."""
    if n <= 0:
        return float("nan"), float("nan")

    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * np.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
        / denom
    )
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    """Utility for recording model/dataset artifact hashes in reproducibility notes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
