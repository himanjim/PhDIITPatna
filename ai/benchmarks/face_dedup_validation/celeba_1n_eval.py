#!/usr/bin/env python3
"""
celeba_1n_eval.py

True 1:N InsightFace + FAISS evaluation on CelebA.

Purpose
-------
The existing PhD paper already reports:
  * pairwise threshold calibration, and
  * synthetic-vector FAISS scalability.

This script fills the missing accuracy experiment:
  "When one real face is searched against an N-person election-day gallery,
   does the correct prior voter remain the nearest accepted identity?"

Metrics
-------
For each gallery size N, the script reports:

1. Rank-1 identification rate on MATED queries.
   A query is mated when the same identity exists in the gallery.

2. FNIR (False Negative Identification Rate).
   A mated query is counted as a failure when:
       - the nearest identity is wrong, OR
       - the nearest identity is correct but its squared-L2 distance exceeds
         the operational threshold.

3. FPIR (False Positive Identification Rate).
   A NON-MATED query comes from an identity absent from the gallery.
   It is a false positive identification if any nearest neighbour is accepted
   by the threshold.

FAISS IndexFlatL2 returns SQUARED L2 distance. The script therefore squares
the freshly calibrated true-L2 threshold before making a FAISS decision.

Dataset layout expected
-----------------------
data/celeba/
    img_align_celeba/
        000001.jpg
        ...
    identity_CelebA.txt

identity_CelebA.txt format:
    000001.jpg 2880
    000002.jpg 2937
    ...

The script deliberately embeds only the subset needed for the experiment,
rather than all 202,599 CelebA images.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd

from face_embedding_backend import (
    FaceEmbedder,
    embed_paths_with_cache,
    wilson_interval,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_identity_file(identity_file: Path, image_root: Path):
    """
    Read CelebA identity annotations and retain only image files that actually
    exist in the extracted image directory.
    """
    by_id: Dict[str, List[str]] = defaultdict(list)

    with identity_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Bad identity line {line_no}: expected '<filename> <identity>'"
                )

            filename, identity = parts[0], parts[1]
            p = image_root / filename
            if p.exists():
                by_id[str(identity)].append(str(p))

    for identity in by_id:
        by_id[identity] = sorted(by_id[identity])

    return dict(by_id)


def choose_identity_images(
    by_id: Dict[str, List[str]],
    seed: int,
    max_gallery_size: int,
    mated_probes_per_id: int,
    nonmated_probes_per_id: int,
):
    """
    Build deterministic per-identity image choices.

    Gallery identities need at least:
        1 gallery image + mated_probes_per_id probe images.

    The gallery identity order is shuffled once and then reused so that
    N=1000 is a strict subset of N=2500, etc. This makes scale trends easier
    to interpret.
    """
    rng = random.Random(seed)

    eligible_gallery_ids = [
        identity
        for identity, imgs in by_id.items()
        if len(imgs) >= 1 + mated_probes_per_id
    ]

    rng.shuffle(eligible_gallery_ids)

    if len(eligible_gallery_ids) < max_gallery_size:
        print(
            f"[WARN] Requested maximum gallery N={max_gallery_size}, but only "
            f"{len(eligible_gallery_ids)} identities have enough images. "
            f"The experiment will cap N at {len(eligible_gallery_ids)}."
        )
        max_gallery_size = len(eligible_gallery_ids)

    ordered_gallery_ids = eligible_gallery_ids[:max_gallery_size]

    selections = {}

    for identity, imgs in by_id.items():
        local = list(imgs)

        # Per-identity deterministic shuffle avoids always choosing the lowest
        # filename as enrollment/probe.
        local_rng = random.Random(f"{seed}:{identity}")
        local_rng.shuffle(local)

        if identity in ordered_gallery_ids:
            selections[identity] = {
                "gallery": local[0],
                "mated": local[1 : 1 + mated_probes_per_id],
                "nonmated": [],
            }
        else:
            selections[identity] = {
                "gallery": None,
                "mated": [],
                "nonmated": local[:nonmated_probes_per_id],
            }

    return ordered_gallery_ids, selections


def build_required_path_set(
    by_id: Dict[str, List[str]],
    ordered_gallery_ids: List[str],
    selections: dict,
    nonmated_probes_per_id: int,
):
    """
    Embed only images that can participate in at least one tested gallery size.

    For identities not in the maximum gallery, take up to
    nonmated_probes_per_id images as open-set probes.
    """
    required = set()

    for identity in ordered_gallery_ids:
        s = selections[identity]
        required.add(s["gallery"])
        required.update(s["mated"])

    max_gallery_set = set(ordered_gallery_ids)
    for identity, imgs in by_id.items():
        if identity in max_gallery_set:
            continue

        local_rng = random.Random(f"nonmated:{identity}")
        local = list(imgs)
        local_rng.shuffle(local)
        required.update(local[:nonmated_probes_per_id])

    return sorted(required)


def faiss_search(
    gallery_matrix: np.ndarray,
    query_matrix: np.ndarray,
):
    """Exact 1-nearest-neighbour search using the same FAISS geometry."""
    index = faiss.IndexFlatL2(gallery_matrix.shape[1])
    index.add(gallery_matrix.astype(np.float32, copy=False))
    D, I = index.search(query_matrix.astype(np.float32, copy=False), 1)
    return D[:, 0], I[:, 0]


def evaluate_one_gallery(
    N: int,
    ordered_gallery_ids: List[str],
    by_id: Dict[str, List[str]],
    selections: dict,
    embeddings: Dict[str, np.ndarray],
    tau_sq: float,
    nonmated_probes_per_id: int,
    nonmated_limit: int,
    seed: int,
):
    """
    Evaluate one gallery size and return:
        summary row
        detailed per-query rows
    """
    gallery_ids = ordered_gallery_ids[:N]

    valid_gallery_ids = []
    gallery_vectors = []
    gallery_paths = []

    for identity in gallery_ids:
        p = selections[identity]["gallery"]
        if p in embeddings:
            valid_gallery_ids.append(identity)
            gallery_vectors.append(embeddings[p])
            gallery_paths.append(p)

    if len(valid_gallery_ids) < 2:
        raise RuntimeError(f"Too few valid gallery embeddings for N={N}")

    G = np.vstack(gallery_vectors).astype(np.float32)

    details = []

    # ---------------- MATED probes ---------------- #
    mated_vectors = []
    mated_truth = []
    mated_paths = []

    for identity in valid_gallery_ids:
        for p in selections[identity]["mated"]:
            if p in embeddings:
                mated_vectors.append(embeddings[p])
                mated_truth.append(identity)
                mated_paths.append(p)

    if not mated_vectors:
        raise RuntimeError(f"No mated probes available for N={N}")

    Qm = np.vstack(mated_vectors).astype(np.float32)
    Dm, Im = faiss_search(G, Qm)

    mated_rank1_correct = 0
    mated_success = 0

    for truth, qpath, d_sq, idx in zip(mated_truth, mated_paths, Dm, Im):
        nearest_id = valid_gallery_ids[int(idx)]
        rank1_correct = nearest_id == truth
        accepted = float(d_sq) <= tau_sq
        success = rank1_correct and accepted

        mated_rank1_correct += int(rank1_correct)
        mated_success += int(success)

        details.append(
            {
                "gallery_size": N,
                "query_type": "mated",
                "true_identity": truth,
                "query_path": qpath,
                "nearest_identity": nearest_id,
                "nearest_gallery_path": gallery_paths[int(idx)],
                "distance_sq_l2": float(d_sq),
                "distance_l2": float(np.sqrt(max(0.0, float(d_sq)))),
                "threshold_sq_l2": float(tau_sq),
                "accepted_as_prior_voter": bool(accepted),
                "rank1_correct": bool(rank1_correct),
                "decision_correct": bool(success),
            }
        )

    # ---------------- NON-MATED probes ---------------- #
    gallery_set = set(valid_gallery_ids)
    nonmated_candidates = []

    for identity, imgs in by_id.items():
        if identity in gallery_set:
            continue

        # Use multiple images per absent identity. This is important at
        # N~=9,000 because relatively few evaluation identities remain outside the
        # largest gallery; one query per identity would be too small
        # for useful FPIR estimation.
        # Use only images that were actually selected for embedding.  For
        # identities that will enter a larger nested gallery, this normally
        # means their preselected gallery/mated images.  For identities outside
        # the maximum gallery, it includes the larger open-set probe sample.
        # This keeps the total embedding workload modest while still providing
        # thousands of non-mated probes at the smaller N values.
        local = [p for p in imgs if p in embeddings]
        local_rng = random.Random(f"{seed}:N{N}:nonmated:{identity}")
        local_rng.shuffle(local)

        for p in local[:nonmated_probes_per_id]:
            nonmated_candidates.append((identity, p))

    # Deterministic global cap for runtime / CSV size.
    rng = random.Random(f"{seed}:N{N}:nonmated_global")
    rng.shuffle(nonmated_candidates)
    if nonmated_limit > 0:
        nonmated_candidates = nonmated_candidates[:nonmated_limit]

    false_positive_identifications = 0
    nonmated_distances = []

    if nonmated_candidates:
        Qn = np.vstack(
            [embeddings[p] for _, p in nonmated_candidates]
        ).astype(np.float32)

        Dn, In = faiss_search(G, Qn)

        for (truth, qpath), d_sq, idx in zip(nonmated_candidates, Dn, In):
            nearest_id = valid_gallery_ids[int(idx)]
            accepted = float(d_sq) <= tau_sq
            false_positive_identifications += int(accepted)
            nonmated_distances.append(float(d_sq))

            details.append(
                {
                    "gallery_size": N,
                    "query_type": "nonmated",
                    "true_identity": truth,
                    "query_path": qpath,
                    "nearest_identity": nearest_id,
                    "nearest_gallery_path": gallery_paths[int(idx)],
                    "distance_sq_l2": float(d_sq),
                    "distance_l2": float(np.sqrt(max(0.0, float(d_sq)))),
                    "threshold_sq_l2": float(tau_sq),
                    "accepted_as_prior_voter": bool(accepted),
                    "rank1_correct": False,
                    "decision_correct": bool(not accepted),
                }
            )

    mated_n = len(mated_truth)
    nonmated_n = len(nonmated_candidates)

    rank1_rate = mated_rank1_correct / mated_n
    fnir_count = mated_n - mated_success
    fnir = fnir_count / mated_n

    fpir = (
        false_positive_identifications / nonmated_n
        if nonmated_n
        else float("nan")
    )

    fnir_lo, fnir_hi = wilson_interval(fnir_count, mated_n)
    fpir_lo, fpir_hi = wilson_interval(
        false_positive_identifications,
        nonmated_n,
    )

    mated_d = [
        r["distance_sq_l2"]
        for r in details
        if r["query_type"] == "mated"
    ]

    summary = {
        "requested_gallery_size": int(N),
        "actual_gallery_size": int(len(valid_gallery_ids)),
        "tau_l2": float(np.sqrt(tau_sq)),
        "tau_sq_l2": float(tau_sq),
        "mated_queries": int(mated_n),
        "nonmated_queries": int(nonmated_n),
        "rank1_identification_rate": float(rank1_rate),
        "fnir": float(fnir),
        "fnir_95ci_low": float(fnir_lo),
        "fnir_95ci_high": float(fnir_hi),
        "fpir": float(fpir),
        "fpir_95ci_low": float(fpir_lo),
        "fpir_95ci_high": float(fpir_hi),
        "mated_distance_sq_median": float(np.median(mated_d)),
        "mated_distance_sq_p95": float(np.percentile(mated_d, 95)),
        "nonmated_distance_sq_median": (
            float(np.median(nonmated_distances))
            if nonmated_distances
            else float("nan")
        ),
        "nonmated_distance_sq_p05": (
            float(np.percentile(nonmated_distances, 5))
            if nonmated_distances
            else float("nan")
        ),
        "mated_failures": int(fnir_count),
        "nonmated_false_positive_identifications": int(
            false_positive_identifications
        ),
    }

    return summary, details


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--images",
        required=True,
        help="Path to extracted CelebA img_align_celeba directory.",
    )
    ap.add_argument(
        "--identity-file",
        required=True,
        help="Path to identity_CelebA.txt.",
    )
    ap.add_argument(
        "--gallery-sizes",
        default="1000,2500,5000,7500,9000",
        help="Comma-separated N values.",
    )
    ap.add_argument("--tau-l2", type=float, default=None, help="Optional true-L2 threshold. Prefer --calibration-json for the new raccoon_l study.")
    ap.add_argument("--calibration-json", default=None, help="calibration_summary.json produced by calibrate_threshold_raccoon.py")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument(
        "--identity-split",
        default=None,
        help="Optional identity_split.json. When supplied, only evaluation_ids are used for 1:N testing.",
    )

    ap.add_argument(
        "--mated-probes-per-id",
        type=int,
        default=1,
        help="Number of genuine duplicate probes per enrolled identity.",
    )
    ap.add_argument(
        "--nonmated-probes-per-id",
        type=int,
        default=10,
        help=(
            "Maximum images per identity absent from the current gallery. "
            "Use >1 because a 10k CelebA gallery leaves few absent identities."
        ),
    )
    ap.add_argument(
        "--nonmated-limit",
        type=int,
        default=10000,
        help="Global cap per gallery size; 0 means no cap.",
    )

    ap.add_argument("--backend", choices=["local", "triton"], default="local")
    ap.add_argument("--triton-url", default="localhost:8001")
    ap.add_argument("--model-pack", "--triton-model", dest="triton_model", default="raccoon_l", help="InsightFace model pack. --triton-model is retained as a backward-compatible alias.")
    ap.add_argument("--triton-input", default="input.1")
    ap.add_argument("--triton-output", default="683")
    ap.add_argument(
        "--rgb",
        action="store_true",
        help=(
            "Convert aligned BGR crop to RGB before Triton. "
            "Leave OFF to reproduce current calibrate_threshold.py defaults."
        ),
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Force local InsightFace to CPUExecutionProvider.",
    )
    ap.add_argument(
        "--strict-one-face",
        action="store_true",
        help=(
            "Reject images unless exactly one face is detected. "
            "Default reproduces calibrate_threshold.py by taking the largest face."
        ),
    )
    ap.add_argument("--cache", default="celeba_raccoon_l_embeddings_cache.npz")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out-dir", default="celeba_1n_raccoon_l_results")

    args = ap.parse_args()

    # The new raccoon_l experiments use a freshly calibrated threshold.  A
    # numeric --tau-l2 remains available for controlled comparisons, but it is
    # not silently defaulted to the earlier buffalo_l value.
    if args.calibration_json:
        calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
        tau_l2 = float(calibration["selected"]["tau_l2"])
        print(f"[THRESHOLD] Loaded calibrated tau_L2={tau_l2:.6f} from {args.calibration_json}")
    elif args.tau_l2 is not None:
        tau_l2 = float(args.tau_l2)
        print(f"[THRESHOLD] Using explicitly supplied tau_L2={tau_l2:.6f}")
    else:
        raise SystemExit(
            "Provide --calibration-json from calibrate_threshold_raccoon.py "
            "or explicitly provide --tau-l2."
        )

    image_root = Path(args.images)
    identity_file = Path(args.identity_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_root.is_dir():
        raise SystemExit(f"CelebA image directory not found: {image_root}")
    if not identity_file.is_file():
        raise SystemExit(f"CelebA identity file not found: {identity_file}")

    gallery_sizes = sorted(
        {int(x.strip()) for x in args.gallery_sizes.split(",") if x.strip()}
    )
    max_gallery = max(gallery_sizes)

    print("[DATA] Reading CelebA identities...")
    by_id = read_identity_file(identity_file, image_root)

    # If calibration created an identity-disjoint split, remove the calibration
    # identities before constructing any gallery or query.  This keeps threshold
    # selection and final 1:N evaluation statistically separate.
    if args.identity_split:
        split = json.loads(Path(args.identity_split).read_text(encoding="utf-8"))
        evaluation_ids = set(str(x) for x in split["evaluation_ids"])
        by_id = {identity: images for identity, images in by_id.items() if identity in evaluation_ids}
        print(f"[SPLIT] Using {len(by_id):,} evaluation identities from {args.identity_split}")

    image_count = sum(len(v) for v in by_id.values())
    print(
        f"[DATA] usable images={image_count:,} "
        f"identities={len(by_id):,}"
    )

    ordered_gallery_ids, selections = choose_identity_images(
        by_id=by_id,
        seed=args.seed,
        max_gallery_size=max_gallery,
        mated_probes_per_id=args.mated_probes_per_id,
        nonmated_probes_per_id=args.nonmated_probes_per_id,
    )

    # Remove requested N values that exceed the actually eligible maximum.
    gallery_sizes = [n for n in gallery_sizes if n <= len(ordered_gallery_ids)]
    if not gallery_sizes:
        raise SystemExit("No requested gallery size can be constructed.")

    required_paths = build_required_path_set(
        by_id,
        ordered_gallery_ids,
        selections,
        args.nonmated_probes_per_id,
    )

    print(
        f"[DATA] Only {len(required_paths):,} selected images need embeddings "
        f"(not all {image_count:,} CelebA images)."
    )

    providers = (
        ["CPUExecutionProvider"]
        if args.cpu
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    embedder = FaceEmbedder(
        backend=args.backend,
        providers=providers,
        triton_url=args.triton_url,
        model_name=args.triton_model,
        input_name=args.triton_input,
        output_name=args.triton_output,
        use_rgb_before_triton=args.rgb,
        strict_one_face=args.strict_one_face,
    )

    embeddings, failures = embed_paths_with_cache(
        required_paths,
        embedder=embedder,
        cache_path=args.cache,
        recompute=args.recompute,
        progress_every=100,
    )

    pd.DataFrame(failures).to_csv(
        out_dir / "embedding_failures.csv",
        index=False,
    )

    tau_sq = float(tau_l2 ** 2)
    summaries = []
    all_details = []

    for N in gallery_sizes:
        print(f"\n[EVAL] ===== Gallery N={N:,} =====")

        summary, details = evaluate_one_gallery(
            N=N,
            ordered_gallery_ids=ordered_gallery_ids,
            by_id=by_id,
            selections=selections,
            embeddings=embeddings,
            tau_sq=tau_sq,
            nonmated_probes_per_id=args.nonmated_probes_per_id,
            nonmated_limit=args.nonmated_limit,
            seed=args.seed,
        )

        summaries.append(summary)
        all_details.extend(details)

        print(
            f"Rank-1={summary['rank1_identification_rate']:.6f}  "
            f"FNIR={summary['fnir']:.6f}  "
            f"FPIR={summary['fpir']:.6f}"
        )
        print(
            f"mated={summary['mated_queries']:,}  "
            f"non-mated={summary['nonmated_queries']:,}"
        )

        # A practical warning for rare-event reporting.
        if summary["nonmated_queries"] < 1000:
            print(
                "[WARN] <1000 non-mated probes at this N. "
                "Do not make a precise 0.1% FPIR claim from this row alone; "
                "report the Wilson confidence interval and probe count."
            )

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.DataFrame(all_details)

    summary_path = out_dir / "celeba_1n_summary.csv"
    detail_path = out_dir / "celeba_1n_query_details.csv"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    run_config = {
        "dataset": "CelebA",
        "seed": args.seed,
        "backend": args.backend,
        "tau_l2": tau_l2,
        "tau_sq_l2": tau_sq,
        "gallery_sizes": gallery_sizes,
        "mated_probes_per_id": args.mated_probes_per_id,
        "nonmated_probes_per_id": args.nonmated_probes_per_id,
        "nonmated_limit": args.nonmated_limit,
        "strict_one_face": args.strict_one_face,
        "rgb_before_triton": args.rgb,
        "usable_identities": len(by_id),
        "selected_images_for_embedding": len(required_paths),
        "successful_embeddings": len(embeddings),
        "embedding_failures": len(failures),
    }

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\n[OUT] Wrote:")
    print(f"  {summary_path}")
    print(f"  {detail_path}")
    print(f"  {out_dir / 'embedding_failures.csv'}")
    print(f"  {out_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
