#!/usr/bin/env python3
"""
Calibrate a fresh face-matching threshold for InsightFace 2.0 / raccoon_l.

The earlier paper's tau_L2=1.15 came from a different software/model package.
For the new experiments we reserve 1,000 CelebA identities for calibration,
select a threshold at the requested pairwise FMR, and keep the remaining
identities completely separate for the later 1:N test.

This is deliberately a calibration script, not a training script.  No network
weights are changed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from face_embedding_backend import FaceEmbedder, embed_paths_with_cache


def read_identity_file(identity_file: Path, image_root: Path):
    """Group CelebA image paths by identity label."""
    grouped = defaultdict(list)
    with identity_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"Unexpected identity line {line_number}: {line!r}")
            filename, identity = parts
            path = image_root / filename
            if path.is_file():
                grouped[str(identity)].append(str(path))

    for identity in grouped:
        grouped[identity] = sorted(grouped[identity])
    return dict(grouped)


def make_identity_split(grouped, split_file: Path, seed: int, calibration_ids: int, recreate: bool):
    """Create one deterministic identity-disjoint calibration/evaluation split."""
    eligible = sorted(
        identity for identity, images in grouped.items() if len(images) >= 2
    )

    if split_file.exists() and not recreate:
        split = json.loads(split_file.read_text(encoding="utf-8"))
        if split.get("seed") != seed or split.get("calibration_identity_count") != calibration_ids:
            raise ValueError(
                "Existing split was created with different settings. "
                "Use --recreate-split if you intentionally want a new split."
            )
        return split

    if len(eligible) <= calibration_ids:
        raise ValueError("Too few identities for a separate evaluation set.")

    rng = random.Random(seed)
    rng.shuffle(eligible)

    split = {
        "seed": seed,
        "eligible_identity_count": len(eligible),
        "calibration_identity_count": calibration_ids,
        "calibration_ids": eligible[:calibration_ids],
        "evaluation_ids": eligible[calibration_ids:],
    }

    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def deterministic_images(identity: str, paths, seed: int):
    """Shuffle one person's images reproducibly rather than using filename order."""
    result = list(paths)
    random.Random(f"{seed}:{identity}").shuffle(result)
    return result


def build_positive_pairs(labels):
    """Create all same-identity pairs among the selected calibration images."""
    by_identity = defaultdict(list)
    for index, label in enumerate(labels):
        by_identity[label].append(index)

    pairs = []
    for indices in by_identity.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pairs.append((indices[a], indices[b]))
    return pairs


def build_negative_pairs(labels, count: int, seed: int):
    """Sample a fixed number of different-identity pairs."""
    by_identity = defaultdict(list)
    for index, label in enumerate(labels):
        by_identity[label].append(index)

    identities = list(by_identity)
    rng = random.Random(seed)
    pairs = []

    while len(pairs) < count:
        first_id, second_id = rng.sample(identities, 2)
        pairs.append(
            (
                rng.choice(by_identity[first_id]),
                rng.choice(by_identity[second_id]),
            )
        )
    return pairs


def pair_l2(matrix: np.ndarray, pairs):
    """Calculate true L2 distances for many pairs in one vectorised operation."""
    left = np.fromiter((a for a, _ in pairs), dtype=np.int64)
    right = np.fromiter((b for _, b in pairs), dtype=np.int64)
    return np.linalg.norm(matrix[left] - matrix[right], axis=1).astype(np.float32)


def metrics(pos_d, neg_d, tau: float):
    """Return verification metrics at one true-L2 threshold."""
    tp = int(np.sum(pos_d <= tau))
    fn = len(pos_d) - tp
    fp = int(np.sum(neg_d <= tau))
    tn = len(neg_d) - fp

    fmr = fp / len(neg_d)
    fnmr = fn / len(pos_d)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / len(pos_d)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tau_l2": float(tau),
        "tau_sq_l2": float(tau * tau),
        "fmr": float(fmr),
        "fnmr": float(fnmr),
        "accuracy": float((tp + tn) / (len(pos_d) + len(neg_d))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_matches": fp,
        "false_non_matches": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--identity-split", type=Path, default=Path("celeba_identity_split.json"))
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--calibration-identities", type=int, default=1000)
    parser.add_argument("--recreate-split", action="store_true")
    parser.add_argument("--max-images-per-id", type=int, default=4)
    parser.add_argument("--negative-pairs", type=int, default=200000)
    parser.add_argument("--target-fmr", type=float, default=0.001)
    parser.add_argument("--grid-start", type=float, default=0.70)
    parser.add_argument("--grid-end", type=float, default=1.50)
    parser.add_argument("--grid-step", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--strict-one-face", action="store_true")
    parser.add_argument("--cache", default="celeba_raccoon_l_calibration_cache.npz")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "calibration_raccoon_l")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grouped = read_identity_file(args.identity_file, args.images)

    split = make_identity_split(
        grouped,
        args.identity_split,
        args.seed,
        args.calibration_identities,
        args.recreate_split,
    )
    print(
        f"[SPLIT] calibration={len(split['calibration_ids']):,}; "
        f"evaluation={len(split['evaluation_ids']):,}"
    )

    selected_paths = []
    selected_labels = []
    for identity in split["calibration_ids"]:
        ordered = deterministic_images(identity, grouped[identity], args.seed)
        for path in ordered[: args.max_images_per_id]:
            selected_paths.append(path)
            selected_labels.append(identity)

    print(f"[DATA] Calibration images requested: {len(selected_paths):,}")

    providers = (
        ["CPUExecutionProvider"]
        if args.cpu
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    embedder = FaceEmbedder(
        backend="local",
        providers=providers,
        model_name="raccoon_l",
        strict_one_face=args.strict_one_face,
    )

    embeddings, failures = embed_paths_with_cache(
        selected_paths,
        embedder,
        args.cache,
        recompute=args.recompute,
        progress_every=100,
    )
    pd.DataFrame(failures).to_csv(args.out_dir / "embedding_failures.csv", index=False)

    kept_paths, kept_labels = [], []
    for path, label in zip(selected_paths, selected_labels):
        if path in embeddings:
            kept_paths.append(path)
            kept_labels.append(label)

    matrix = np.vstack([embeddings[path] for path in kept_paths]).astype(np.float32)
    positive_pairs = build_positive_pairs(kept_labels)
    negative_pairs = build_negative_pairs(kept_labels, args.negative_pairs, args.seed + 1)

    pos_d = pair_l2(matrix, positive_pairs)
    neg_d = pair_l2(matrix, negative_pairs)

    thresholds = np.arange(
        args.grid_start,
        args.grid_end + args.grid_step / 2,
        args.grid_step,
        dtype=np.float64,
    )
    rows = [metrics(pos_d, neg_d, float(tau)) for tau in thresholds]

    valid = [row for row in rows if row["fmr"] <= args.target_fmr]
    if not valid:
        raise SystemExit("No threshold in the requested grid meets target FMR.")

    # Choose the largest threshold that still satisfies the false-match target;
    # among admissible thresholds this normally gives the lowest false non-match rate.
    selected = max(valid, key=lambda row: row["tau_l2"])
    eer_like = min(rows, key=lambda row: abs(row["fmr"] - row["fnmr"]))
    legacy = metrics(pos_d, neg_d, 1.15)

    pd.DataFrame(rows).to_csv(args.out_dir / "threshold_sweep.csv", index=False)

    summary = {
        "software": "InsightFace 2.0",
        "model_pack": "raccoon_l",
        "recognition_model": "w600k_r50.onnx",
        "seed": args.seed,
        "identity_split": str(args.identity_split),
        "calibration_identities": len(split["calibration_ids"]),
        "evaluation_identities": len(split["evaluation_ids"]),
        "calibration_images_embedded": len(kept_paths),
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "target_fmr": args.target_fmr,
        "selected": selected,
        "eer_like": eer_like,
        "legacy_tau_1_15": legacy,
        "positive_l2_mean": float(np.mean(pos_d)),
        "negative_l2_mean": float(np.mean(neg_d)),
    }
    (args.out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== New raccoon_l operating point ===")
    print(f"tau_L2          : {selected['tau_l2']:.6f}")
    print(f"tau_squared     : {selected['tau_sq_l2']:.6f}")
    print(f"FMR             : {selected['fmr']:.6f}")
    print(f"FNMR            : {selected['fnmr']:.6f}")
    print("\n=== Earlier tau=1.15 on the same new embeddings ===")
    print(f"FMR             : {legacy['fmr']:.6f}")
    print(f"FNMR            : {legacy['fnmr']:.6f}")
    print(f"\n[OUT] {args.out_dir / 'calibration_summary.json'}")


if __name__ == "__main__":
    main()
