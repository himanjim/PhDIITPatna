#!/usr/bin/env python3
"""Download and verify the three CelebA files needed by this experiment."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import gdown

# IDs and MD5 values are the same ones used by torchvision's CelebA loader.
FILES = [
    (
        "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
        "img_align_celeba.zip",
        "00d2c5bc6d35e252742224ab0c1e8fcb",
    ),
    (
        "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
        "identity_CelebA.txt",
        "32bd1bd63d3c78cd57e08160ec5ed1e2",
    ),
    (
        "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
        "list_eval_partition.txt",
        "d32c9cbf5e040fd4025c592c306e6668",
    ),
]


def md5_file(path: Path) -> str:
    """Compute the published CelebA MD5 digest."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data") / "celeba")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for file_id, filename, expected_md5 in FILES:
        destination = args.out / filename

        if destination.exists() and md5_file(destination) == expected_md5:
            print(f"[DATA] Already verified: {destination}")
            continue

        print(f"[DATA] Downloading {filename}...")
        result = gdown.download(id=file_id, output=str(destination), quiet=False)
        if result is None:
            raise SystemExit(f"Download failed: {filename}")

        actual = md5_file(destination)
        if actual != expected_md5:
            raise SystemExit(
                f"MD5 mismatch for {filename}: expected {expected_md5}, got {actual}"
            )
        print(f"[DATA] Verified: {filename}")

    image_dir = args.out / "img_align_celeba"
    if not image_dir.exists() or not any(image_dir.glob("*.jpg")):
        print("[DATA] Extracting img_align_celeba.zip...")
        with zipfile.ZipFile(args.out / "img_align_celeba.zip", "r") as zf:
            zf.extractall(args.out)

    print(f"[DATA] Ready: {image_dir}")


if __name__ == "__main__":
    main()
