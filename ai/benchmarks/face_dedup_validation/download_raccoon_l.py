#!/usr/bin/env python3
"""Download and verify the official InsightFace 2.0 raccoon_l model pack."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# The URL and digest are from the official InsightFace model-zoo release.
MODEL_URL = (
    "https://github.com/deepinsight/insightface/releases/download/"
    "model-zoo/raccoon_l.zip"
)
EXPECTED_SHA256 = (
    "70cd4f2f1de0a89dd0983bdac55a066a6178543f86e9ec87154f6f259bdded7e"
)


def sha256_file(path: Path) -> str:
    """Hash a large file without loading the full archive into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Download with a compact progress indicator."""
    def progress(blocks: int, block_size: int, total_size: int) -> None:
        received = blocks * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * received / total_size)
            print(
                f"\r[DOWNLOAD] {pct:6.2f}%  "
                f"{received / 1024 / 1024:,.1f} MB",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(url, destination, reporthook=progress)
    print()


def install_archive(archive: Path, models_root: Path) -> Path:
    """Extract the verified archive into the normal InsightFace model root."""
    target = models_root / "raccoon_l"
    if target.exists():
        shutil.rmtree(target)

    models_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as zf:
        files = [name for name in zf.namelist() if not name.endswith("/")]
        already_wrapped = files and all(
            name.replace("\\", "/").startswith("raccoon_l/")
            for name in files
        )

        if already_wrapped:
            zf.extractall(models_root)
        else:
            target.mkdir(parents=True, exist_ok=True)
            zf.extractall(target)

    # These are the two model files documented for raccoon_l.  Their presence
    # is checked after extraction so a malformed archive fails immediately.
    detector = list(target.rglob("det_10g_wo.onnx"))
    recognizer = list(target.rglob("w600k_r50.onnx"))
    if len(detector) != 1 or len(recognizer) != 1:
        raise RuntimeError(
            "Installed archive does not contain exactly one detector and "
            "one w600k_r50 recogniser."
        )

    print(f"[MODEL] Installed: {target}")
    print(f"[MODEL] Detector:   {detector[0]}")
    print(f"[MODEL] Recogniser: {recognizer[0]}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path.home() / ".insightface" / "models",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="raccoon_l_") as tmp:
        archive = Path(tmp) / "raccoon_l.zip"
        print("[MODEL] Downloading raccoon_l from the official model zoo...")
        download(MODEL_URL, archive)

        actual = sha256_file(archive)
        print(f"[MODEL] SHA-256: {actual}")
        if actual.lower() != EXPECTED_SHA256.lower():
            raise SystemExit("Model archive SHA-256 verification failed.")

        print("[MODEL] Archive digest verified.")
        install_archive(archive, args.models_root)


if __name__ == "__main__":
    main()
