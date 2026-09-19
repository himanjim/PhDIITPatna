#!/usr/bin/env python3
"""Verify Python, ONNX Runtime CUDA, FAISS and raccoon_l before long runs."""

from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

import onnxruntime as ort

# The [cuda,cudnn] extras place NVIDIA runtime DLLs in Python site-packages.
# Loading them explicitly is the most reliable path on Windows.
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")

from insightface.app import FaceAnalysis


def version(name: str) -> str:
    """Read the installed distribution version without importing it."""
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not installed"


def main() -> None:
    print("=== Core versions ===")
    print("InsightFace       :", version("insightface"))
    print("ONNX Runtime GPU  :", version("onnxruntime-gpu"))
    print("FAISS CPU         :", version("faiss-cpu"))
    print("NumPy             :", version("numpy"))
    print("ORT providers     :", ort.get_available_providers())

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise SystemExit(
            "CUDAExecutionProvider is missing. Do not start the dataset run yet."
        )

    model_root = Path.home() / ".insightface" / "models" / "raccoon_l"
    if not model_root.exists():
        raise SystemExit(
            f"raccoon_l is not installed at {model_root}. "
            "Run download_raccoon_l.py first."
        )

    print("\nLoading only the detector and recogniser...")
    app = FaceAnalysis(
        name="raccoon_l",
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Print the provider assigned to each model session.  This catches a silent
    # CUDA-to-CPU fallback before it contaminates the experimental record.
    for name, model in getattr(app, "models", {}).items():
        session = getattr(model, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            print(f"{name:20s}: {session.get_providers()}")

    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
