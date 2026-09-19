#!/usr/bin/env python3
"""
experiment_common.py

Shared utilities for the InsightFace 2.0 / raccoon_l experiments.

This module is intentionally small.  It provides only the functions needed by
the PGU-Face and CelebA evaluation scripts:

    * InsightFace 2.0 model initialisation
    * CUDA provider validation
    * 512-D L2-normalised embeddings
    * embedding cache handling
    * Wilson confidence intervals
    * reproducibility metadata

The GPU is used for InsightFace inference.  FAISS remains on the CPU.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# On Windows, ONNX Runtime can install CUDA/cuDNN libraries as Python
# site-packages.  Preloading them before InsightFace opens its ONNX sessions
# avoids a common case where CUDAExecutionProvider is present but dependent DLLs
# are not yet visible to Windows.
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls(directory="")
    except Exception as exc:
        print(f"[WARN] ONNX Runtime DLL preload reported: {exc}")

from insightface.app import FaceAnalysis


EMBEDDING_DIM = 512
DEFAULT_MODEL_PACK = "raccoon_l"
DEFAULT_DET_SIZE = (640, 640)


def package_version(name: str) -> str:
    """Return the installed distribution version without importing it."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without loading a whole ONNX file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def model_file_hashes(model_pack: str = DEFAULT_MODEL_PACK) -> Dict[str, str]:
    """
    Hash the locally installed ONNX files.

    These hashes are useful in the thesis because they record the exact model
    artefacts used, not only the package/model names.
    """
    root = Path.home() / ".insightface" / "models" / model_pack
    hashes: Dict[str, str] = {}

    if not root.exists():
        return hashes

    for path in sorted(root.rglob("*.onnx")):
        hashes[str(path.relative_to(root))] = sha256_file(path)

    return hashes


def runtime_report(model_pack: str = DEFAULT_MODEL_PACK) -> dict:
    """Collect the software metadata required to reproduce an experiment."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "insightface": package_version("insightface"),
        "onnxruntime_gpu": package_version("onnxruntime-gpu"),
        "faiss_cpu": package_version("faiss-cpu"),
        "numpy": package_version("numpy"),
        "opencv_python_headless": package_version("opencv-python-headless"),
        "onnxruntime_available_providers": ort.get_available_providers(),
        "model_pack": model_pack,
        "model_file_sha256": model_file_hashes(model_pack),
    }


def write_runtime_report(path: Path, model_pack: str = DEFAULT_MODEL_PACK) -> None:
    """Write reproducibility metadata beside the experiment results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(runtime_report(model_pack), indent=2),
        encoding="utf-8",
    )


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a float32 unit-length embedding."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("Embedding has an invalid norm.")

    return (vector / norm).astype(np.float32, copy=False)


class FaceEmbedder:
    """
    InsightFace 2.0 wrapper used by the experiments.

    Only detection and recognition modules are loaded.  This reduces memory use
    on the 4 GB GTX 1650 and avoids unrelated age/gender models.
    """

    def __init__(
        self,
        model_pack: str = DEFAULT_MODEL_PACK,
        device: str = "cuda",
        det_size: Tuple[int, int] = DEFAULT_DET_SIZE,
        strict_one_face: bool = False,
    ):
        self.model_pack = model_pack
        self.device = device.lower()
        self.det_size = tuple(det_size)
        self.strict_one_face = bool(strict_one_face)

        available = ort.get_available_providers()

        if self.device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider is not available. "
                    "Run the environment check or use --cpu."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        elif self.device == "cpu":
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        else:
            raise ValueError("device must be 'cuda' or 'cpu'")

        print(f"[MODEL] InsightFace model pack: {self.model_pack}")
        print(f"[MODEL] Requested providers: {providers}")

        self.app = FaceAnalysis(
            name=self.model_pack,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )

        # A fixed detector resolution makes the experiment reproducible across
        # PGU-Face and CelebA.
        self.app.prepare(
            ctx_id=ctx_id,
            det_size=self.det_size,
        )

        # Print the actual providers used by each loaded ONNX session when
        # InsightFace exposes them.  This helps detect an unintended CPU
        # fallback on Windows.
        for key, model in getattr(self.app, "models", {}).items():
            session = getattr(model, "session", None)
            if session is not None and hasattr(session, "get_providers"):
                print(
                    f"[MODEL] {key}: providers={session.get_providers()}"
                )

    def embed_path(self, image_path: str | Path):
        """
        Extract one L2-normalised face embedding.

        By default the largest detected face is used.  This mirrors the policy
        in the earlier calibration code.  With --strict-one-face, an image is
        rejected unless exactly one face is present.
        """
        image_path = str(Path(image_path))
        image = cv2.imread(image_path)

        if image is None:
            return None, {
                "path": image_path,
                "status": "read_failed",
                "faces_detected": 0,
            }

        faces = self.app.get(image)

        if self.strict_one_face:
            if len(faces) != 1:
                return None, {
                    "path": image_path,
                    "status": "face_policy_failed",
                    "faces_detected": len(faces),
                }
            face = faces[0]

        else:
            if not faces:
                return None, {
                    "path": image_path,
                    "status": "no_face_detected",
                    "faces_detected": 0,
                }

            # PGU-Face should contain a single subject, but using the largest
            # face makes the behaviour deterministic if a stray face is found.
            face = max(
                faces,
                key=lambda item: float(
                    (item.bbox[2] - item.bbox[0])
                    * (item.bbox[3] - item.bbox[1])
                ),
            )

        try:
            vector = getattr(face, "normed_embedding", None)

            if vector is None:
                raw = getattr(face, "embedding", None)
                if raw is None:
                    raise RuntimeError(
                        "Recognition model returned no embedding."
                    )
                vector = l2_normalize(raw)
            else:
                vector = l2_normalize(vector)

            if vector.shape != (EMBEDDING_DIM,):
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIM}-D embedding; "
                    f"received {vector.shape}."
                )

            return vector, {
                "path": image_path,
                "status": "ok",
                "faces_detected": len(faces),
            }

        except Exception as exc:
            return None, {
                "path": image_path,
                "status": f"embedding_failed:{type(exc).__name__}",
                "faces_detected": len(faces),
            }


def save_embedding_cache(
    path: Path,
    embeddings: Dict[str, np.ndarray],
    model_pack: str,
) -> None:
    """
    Save embeddings with model metadata.

    Model metadata prevents a cache made with buffalo_l from being silently
    reused with raccoon_l.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(embeddings)
    matrix = np.vstack(
        [embeddings[key] for key in keys]
    ).astype(np.float32)

    metadata = {
        "cache_format": 2,
        "model_pack": model_pack,
        "insightface_version": package_version("insightface"),
        "embedding_dim": EMBEDDING_DIM,
    }

    np.savez_compressed(
        path,
        paths=np.asarray(keys, dtype=object),
        embeddings=matrix,
        metadata_json=json.dumps(metadata),
    )


def load_embedding_cache(
    path: Path,
    model_pack: str,
) -> Dict[str, np.ndarray]:
    """Load only a cache created with the same model pack and library version."""
    if not path.exists():
        return {}

    data = np.load(path, allow_pickle=True)

    if "metadata_json" not in data:
        print(
            f"[CACHE] Ignoring old cache without metadata: {path}"
        )
        return {}

    metadata = json.loads(str(data["metadata_json"]))

    expected = {
        "model_pack": model_pack,
        "insightface_version": package_version("insightface"),
        "embedding_dim": EMBEDDING_DIM,
    }

    for key, expected_value in expected.items():
        actual_value = metadata.get(key)

        if actual_value != expected_value:
            print(
                f"[CACHE] Ignoring {path}: {key}={actual_value!r}; "
                f"expected {expected_value!r}"
            )
            return {}

    paths = data["paths"].tolist()
    matrix = data["embeddings"].astype(np.float32)

    return {
        str(image_path): l2_normalize(vector)
        for image_path, vector in zip(paths, matrix)
    }


def embed_paths_with_cache(
    image_paths: Iterable[str | Path],
    embedder: FaceEmbedder,
    cache_path: Path,
    recompute: bool = False,
    progress_every: int = 100,
):
    """Embed image paths while reusing a compatible on-disk cache."""
    paths = [str(Path(path)) for path in image_paths]

    cache = (
        {}
        if recompute
        else load_embedding_cache(
            cache_path,
            embedder.model_pack,
        )
    )

    embeddings: Dict[str, np.ndarray] = {}
    failures: List[dict] = []

    for index, path in enumerate(paths, start=1):
        if path in cache:
            embeddings[path] = cache[path]

        else:
            vector, metadata = embedder.embed_path(path)

            if vector is None:
                failures.append(metadata)
            else:
                embeddings[path] = vector
                cache[path] = vector

        if index % progress_every == 0 or index == len(paths):
            print(
                f"[EMBED] {index:,}/{len(paths):,} processed; "
                f"{len(embeddings):,} usable; "
                f"{len(failures):,} failed"
            )

    if cache:
        save_embedding_cache(
            cache_path,
            cache,
            model_pack=embedder.model_pack,
        )

    return embeddings, failures


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
):
    """Calculate a 95% Wilson confidence interval for a binomial rate."""
    if total <= 0:
        return float("nan"), float("nan")

    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (
        p + z * z / (2.0 * total)
    ) / denominator

    half = (
        z
        * np.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )

    return (
        float(max(0.0, centre - half)),
        float(min(1.0, centre + half)),
    )
