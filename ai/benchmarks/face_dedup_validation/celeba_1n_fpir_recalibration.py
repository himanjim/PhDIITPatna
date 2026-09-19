#!/usr/bin/env python3
"""
celeba_1n_fpir_recalibration.py

Recalibrate the CelebA de-duplication threshold using 1:N search-level FPIR
(False Positive Identification Rate) rather than pairwise FMR (False Match
Rate), while reusing the existing cached embeddings.

Why this script exists
----------------------
The earlier calibration selected a threshold using pairwise verification
metrics:

    unrelated pair -> distance <= tau ? -> false match

That approach is appropriate for 1:1 face verification, but the voting use
case is a 1:N search:

    query -> FAISS search against the entire gallery
          -> nearest neighbour distance <= tau ? -> false positive identification

A threshold that yields an acceptably small pairwise FMR can still produce an
unacceptably large 1:N FPIR when the gallery contains many identities.
Therefore this script recalibrates tau using search-level FPIR, not pairwise
FMR.

What the script does
--------------------
1. Load the existing CelebA identity file and the saved identity-disjoint split.
2. Reuse the previously computed embedding cache; no InsightFace inference is
   performed here.
3. Construct repeated 1:N search trials using ONLY the calibration identities
   from the split file.
4. Aggregate:
      * nearest-neighbour distances for non-mated queries
      * nearest-neighbour distances for mated queries
      * rank-1 correctness for mated queries
5. Select thresholds that satisfy user-specified FPIR targets.
6. Optionally re-apply those thresholds to an existing
   `celeba_1n_query_details.csv` file from the previous large-gallery run and
   write revised 1:N summary tables without rerunning image embedding.

Important practical point
-------------------------
This script reuses the expensive part of the earlier pipeline:
    image -> raccoon_l -> 512-D embedding

Only the threshold-selection logic changes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd


# All earlier scripts use 512-dimensional, L2-normalised embeddings.
EXPECTED_DIM = 512


def package_version(name: str) -> str:
    """Return an installed package version without importing the package."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_report() -> dict:
    """Record enough software information to reproduce the recalibration."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "faiss_cpu": package_version("faiss-cpu"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
    }


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a unit-length float32 vector."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("Embedding has an invalid norm.")

    return (vector / norm).astype(np.float32, copy=False)


def write_runtime_report(path: Path) -> None:
    """Write a small reproducibility file beside the outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(runtime_report(), indent=2),
        encoding="utf-8",
    )


def read_celeba_identities(
    image_root: Path,
    identity_file: Path,
) -> Dict[str, List[str]]:
    """
    Read identity_CelebA.txt and group available image paths by identity ID.

    Only identities with actual image files beneath `image_root` are retained.
    """
    grouped: Dict[str, List[str]] = defaultdict(list)

    with identity_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 2:
                raise ValueError(
                    f"Unexpected CelebA identity format on line {line_number}."
                )

            filename, identity = parts
            image_path = image_root / filename

            if image_path.is_file():
                grouped[str(identity)].append(str(image_path))

    for identity in grouped:
        grouped[identity] = sorted(grouped[identity])

    return dict(grouped)


def deterministic_images_for_identity(
    identity: str,
    image_paths: Sequence[str],
    seed: int,
) -> List[str]:
    """
    Shuffle one identity's image list reproducibly.

    This matches the deterministic policy used in the earlier experiments.
    """
    result = list(image_paths)
    rng = random.Random(f"{seed}:{identity}")
    rng.shuffle(result)
    return result


def load_embedding_cache(
    cache_path: Path,
    expected_model_pack: str | None = None,
) -> Dict[str, np.ndarray]:
    """
    Load an earlier embedding cache created by the CelebA scripts.

    The cache stores:
        paths
        embeddings
        metadata_json

    The function checks the embedding dimension and, when requested, the model
    pack name.  The embeddings are re-normalised defensively because later
    code assumes exact unit vectors.
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)

    if "paths" not in data or "embeddings" not in data:
        raise ValueError(
            f"{cache_path} does not look like a compatible embedding cache."
        )

    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))

    if expected_model_pack is not None:
        model_pack = metadata.get("model_pack")
        if model_pack is not None and model_pack != expected_model_pack:
            raise ValueError(
                f"Cache model_pack={model_pack!r}, expected "
                f"{expected_model_pack!r}."
            )

    paths = [str(path) for path in data["paths"].tolist()]
    matrix = data["embeddings"].astype(np.float32)

    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Expected (*, {EXPECTED_DIM}) embeddings, got {matrix.shape}."
        )

    embeddings = {
        path: l2_normalize(vector)
        for path, vector in zip(paths, matrix)
    }

    return embeddings


def exact_search(gallery: np.ndarray, queries: np.ndarray):
    """
    Exact 1-nearest-neighbour FAISS search.

    IndexFlatL2 returns SQUARED Euclidean distance.
    """
    index = faiss.IndexFlatL2(gallery.shape[1])
    index.add(gallery.astype(np.float32, copy=False))

    distances, indices = index.search(
        queries.astype(np.float32, copy=False),
        1,
    )

    return distances[:, 0], indices[:, 0]


def choose_threshold_for_fpir(
    nonmated_distance_l2: np.ndarray,
    target_fpir: float,
) -> dict:
    """
    Select the largest threshold whose observed search-level FPIR does not
    exceed the requested target.

    Input distances are TRUE L2 distances from each non-mated query to its
    nearest gallery neighbour.
    """
    if len(nonmated_distance_l2) == 0:
        raise ValueError("No non-mated distances were provided.")

    if not (0.0 <= target_fpir < 1.0):
        raise ValueError("target_fpir must lie in [0, 1).")

    distances = np.sort(
        np.asarray(nonmated_distance_l2, dtype=np.float64)
    )
    m = len(distances)

    # If target FPIR is exactly 0, the largest safe threshold must fall below
    # the smallest observed non-mated nearest-neighbour distance.
    if target_fpir == 0.0:
        tau = float(np.nextafter(distances[0], -np.inf))
    else:
        # We can accept at most floor(target * m) false positive
        # identifications among m non-mated searches.
        allowed = int(math.floor(target_fpir * m))

        if allowed == 0:
            tau = float(np.nextafter(distances[0], -np.inf))
        else:
            tau = float(distances[allowed - 1])

    observed_fpir = float(np.mean(distances <= tau))

    return {
        "target_fpir": float(target_fpir),
        "tau_l2": tau,
        "tau_sq_l2": float(tau * tau),
        "observed_fpir_on_calibration": observed_fpir,
        "nonmated_queries_in_calibration": m,
    }


def build_repeated_search_calibration(
    grouped: Dict[str, List[str]],
    split_data: dict,
    embeddings: Dict[str, np.ndarray],
    seed: int,
    gallery_size: int,
    repeats: int,
    nonmated_per_repeat: int,
):
    """
    Generate search-level calibration data from the identities reserved for
    calibration in `celeba_identity_split.json`.

    Design
    ------
    Each repeat works as follows:
      * sample `gallery_size` identities from the calibration pool;
      * first deterministic image per identity -> gallery;
      * second deterministic image per gallery identity -> mated query;
      * identities not chosen for the gallery -> non-mated query pool;
      * one image per non-gallery identity is used as a non-mated query;
      * FAISS exact search returns the nearest gallery identity and distance.

    Only embeddings already present in the cache are used.  If an image lacks a
    cached embedding, that subject/query is skipped.  This is deliberate: the
    point of the recalibration is to avoid recomputing InsightFace embeddings.
    """
    calibration_ids = [
        identity
        for identity in split_data["calibration_ids"]
        if identity in grouped and len(grouped[identity]) >= 2
    ]

    if gallery_size >= len(calibration_ids):
        raise ValueError(
            "gallery_size must be smaller than the number of calibration "
            "identities so that some non-mated identities remain."
        )

    rng_master = random.Random(seed)

    all_nonmated_dist_l2: List[float] = []
    all_mated_rows: List[dict] = []
    repeat_rows: List[dict] = []

    for repeat_index in range(repeats):
        # A repeat-specific RNG makes every trial reproducible and independent
        # of Python dictionary ordering.
        repeat_seed = rng_master.randint(0, 10**9)
        rng = random.Random(repeat_seed)

        candidate_ids = list(calibration_ids)
        rng.shuffle(candidate_ids)

        gallery_ids = sorted(
            candidate_ids[:gallery_size],
            key=lambda x: int(x),
        )
        non_gallery_ids = sorted(
            candidate_ids[gallery_size:],
            key=lambda x: int(x),
        )

        gallery_vectors = []
        gallery_image_paths = []
        gallery_identity_labels = []

        # Each identity uses the first deterministic image as the gallery
        # candidate and the second as its mated query candidate.
        for identity in gallery_ids:
            ordered = deterministic_images_for_identity(
                identity,
                grouped[identity],
                seed,
            )

            gallery_path = ordered[0]

            if gallery_path not in embeddings:
                continue

            gallery_vectors.append(embeddings[gallery_path])
            gallery_image_paths.append(gallery_path)
            gallery_identity_labels.append(identity)

        if len(gallery_vectors) < 2:
            raise RuntimeError(
                f"Repeat {repeat_index}: too few valid gallery embeddings."
            )

        gallery = np.vstack(gallery_vectors).astype(np.float32)

        # ---------------- Mated queries ---------------- #
        mated_vectors = []
        mated_rows_this_repeat = []

        for identity, gallery_path in zip(
            gallery_identity_labels,
            gallery_image_paths,
        ):
            ordered = deterministic_images_for_identity(
                identity,
                grouped[identity],
                seed,
            )

            if len(ordered) < 2:
                continue

            mated_path = ordered[1]

            if mated_path not in embeddings:
                continue

            mated_vectors.append(embeddings[mated_path])
            mated_rows_this_repeat.append(
                {
                    "repeat_index": repeat_index,
                    "repeat_seed": repeat_seed,
                    "query_type": "mated",
                    "true_identity": identity,
                    "query_path": mated_path,
                }
            )

        if not mated_vectors:
            raise RuntimeError(
                f"Repeat {repeat_index}: no valid mated queries."
            )

        mated_matrix = np.vstack(mated_vectors).astype(np.float32)
        mated_dist_sq, mated_nearest_idx = exact_search(
            gallery,
            mated_matrix,
        )

        for row, distance_sq, nearest_index in zip(
            mated_rows_this_repeat,
            mated_dist_sq,
            mated_nearest_idx,
        ):
            nearest_index = int(nearest_index)
            nearest_id = gallery_identity_labels[nearest_index]
            row["nearest_identity"] = nearest_id
            row["distance_sq_l2"] = float(distance_sq)
            row["distance_l2"] = float(
                math.sqrt(max(0.0, float(distance_sq)))
            )
            row["rank1_correct"] = bool(nearest_id == row["true_identity"])

            all_mated_rows.append(row)

        # ---------------- Non-mated queries ---------------- #
        nonmated_pool = []

        for identity in non_gallery_ids:
            ordered = deterministic_images_for_identity(
                identity,
                grouped[identity],
                seed,
            )

            # The first image is sufficient because the identity is absent
            # from the gallery, so any search is non-mated by construction.
            candidate_path = ordered[0]

            if candidate_path in embeddings:
                nonmated_pool.append((identity, candidate_path))

        rng.shuffle(nonmated_pool)

        if nonmated_per_repeat > 0:
            nonmated_pool = nonmated_pool[:nonmated_per_repeat]

        if not nonmated_pool:
            raise RuntimeError(
                f"Repeat {repeat_index}: no non-mated queries."
            )

        nonmated_vectors = np.vstack(
            [embeddings[path] for _, path in nonmated_pool]
        ).astype(np.float32)

        nonmated_dist_sq, _ = exact_search(gallery, nonmated_vectors)
        nonmated_dist_l2 = np.sqrt(
            np.maximum(nonmated_dist_sq.astype(np.float64), 0.0)
        )

        all_nonmated_dist_l2.extend(nonmated_dist_l2.tolist())

        repeat_rows.append(
            {
                "repeat_index": repeat_index,
                "repeat_seed": repeat_seed,
                "gallery_size_requested": gallery_size,
                "gallery_size_actual": len(gallery_identity_labels),
                "mated_queries": len(mated_rows_this_repeat),
                "nonmated_queries": len(nonmated_pool),
                "nonmated_distance_l2_median": float(
                    np.median(nonmated_dist_l2)
                ),
                "nonmated_distance_l2_min": float(
                    np.min(nonmated_dist_l2)
                ),
            }
        )

    return (
        np.asarray(all_nonmated_dist_l2, dtype=np.float64),
        pd.DataFrame(all_mated_rows),
        pd.DataFrame(repeat_rows),
    )


def evaluate_threshold_on_mated_queries(
    mated_df: pd.DataFrame,
    tau_l2: float,
) -> dict:
    """
    Given calibration mated queries and a threshold, compute search-level
    utility metrics on those mated queries.
    """
    if mated_df.empty:
        raise ValueError("No mated query rows were provided.")

    accepted = mated_df["distance_l2"].to_numpy() <= tau_l2
    rank1 = mated_df["rank1_correct"].to_numpy(dtype=bool)
    successful_detection = rank1 & accepted

    m = len(mated_df)

    return {
        "mated_queries_in_calibration": int(m),
        "rank1_identification_rate_on_calibration": float(np.mean(rank1)),
        "detection_rate_on_calibration": float(np.mean(successful_detection)),
        "fnir_on_calibration": float(1.0 - np.mean(successful_detection)),
        "wrong_nearest_identity_count_on_calibration": int(np.sum(~rank1)),
        "correct_identity_but_over_threshold_count_on_calibration": int(
            np.sum(rank1 & ~accepted)
        ),
    }


def apply_thresholds_to_existing_query_details(
    query_details_csv: Path,
    thresholds_df: pd.DataFrame,
    out_dir: Path,
):
    """
    Re-threshold an existing CelebA 1:N query-details file.

    Compatibility
    -------------
    Two result schemas have been used during the experiments:

    Newer schema:
        gallery_size_requested
        gallery_size_actual

    Earlier schema used in the user's completed run:
        gallery_size

    In the earlier schema, the accurate *actual* gallery size is stored in the
    companion `celeba_1n_summary.csv` file as:
        requested_gallery_size
        actual_gallery_size

    This function detects the schema automatically.  If the companion summary
    file is present beside the query-details CSV, it uses that mapping.  If it
    is not present, `gallery_size` is retained as the requested size and the
    actual size is left equal to the requested size with a warning.

    No FAISS or InsightFace work is repeated here.  The nearest identity and
    nearest-neighbour distance have already been saved in the query-details
    file; only the threshold decision is recomputed.
    """
    df = pd.read_csv(query_details_csv)

    # Columns that are genuinely required regardless of which historical
    # gallery-size schema produced the CSV.
    required_base = {
        "query_type",
        "true_identity",
        "nearest_identity",
        "distance_l2",
        "distance_sq_l2",
    }

    missing_base = required_base.difference(df.columns)
    if missing_base:
        raise ValueError(
            f"{query_details_csv} is missing required columns: "
            f"{sorted(missing_base)}"
        )

    # ------------------------------------------------------------------
    # Normalise the gallery-size columns to one internal representation.
    # ------------------------------------------------------------------
    if {
        "gallery_size_requested",
        "gallery_size_actual",
    }.issubset(df.columns):
        # The file already uses the newer schema.
        work = df.copy()

    elif "gallery_size" in df.columns:
        # The user's completed CelebA run uses this schema.
        work = df.copy()
        work["gallery_size_requested"] = work["gallery_size"].astype(int)

        # Try to recover the exact actual gallery size from the companion
        # summary file generated by the same run.
        companion_summary = query_details_csv.parent / "celeba_1n_summary.csv"

        actual_size_map = {}

        if companion_summary.is_file():
            summary = pd.read_csv(companion_summary)

            if {
                "requested_gallery_size",
                "actual_gallery_size",
            }.issubset(summary.columns):
                actual_size_map = dict(
                    zip(
                        summary["requested_gallery_size"].astype(int),
                        summary["actual_gallery_size"].astype(int),
                    )
                )

                print(
                    "[SCHEMA] Loaded requested→actual gallery-size mapping "
                    f"from {companion_summary}"
                )

        if actual_size_map:
            work["gallery_size_actual"] = (
                work["gallery_size_requested"]
                .map(actual_size_map)
                .fillna(work["gallery_size_requested"])
                .astype(int)
            )
        else:
            # This fallback does not alter threshold metrics because FPIR/FNIR
            # use the saved query rows.  It only affects the descriptive
            # `gallery_size_actual` value written to the revised summary.
            print(
                "[WARN] Could not recover actual gallery sizes from the "
                "companion summary; using requested gallery size as actual."
            )
            work["gallery_size_actual"] = (
                work["gallery_size_requested"].astype(int)
            )

    else:
        raise ValueError(
            f"{query_details_csv} contains neither the newer "
            "'gallery_size_requested/gallery_size_actual' columns nor the "
            "earlier 'gallery_size' column."
        )

    all_rows = []
    summary_rows = []

    for _, threshold_row in thresholds_df.iterrows():
        threshold_name = str(threshold_row["threshold_name"])
        tau_l2 = float(threshold_row["tau_l2"])
        tau_sq = float(tau_l2 * tau_l2)

        temp = work.copy()

        # Store the new operating point explicitly on every row so the revised
        # CSV remains self-contained.
        temp["threshold_name"] = threshold_name
        temp["tau_l2"] = tau_l2
        temp["tau_sq_l2"] = tau_sq

        # Search-level decision: a query is flagged as a previous voter when
        # its nearest-neighbour distance lies within the threshold.
        temp["accepted_as_previous_voter"] = (
            temp["distance_l2"].astype(float) <= tau_l2
        )

        # Recompute Rank-1 from identity labels instead of trusting a historical
        # Boolean column.  This makes the calculation independent of any prior
        # threshold.
        temp["rank1_correct_recomputed"] = False
        mated_mask = temp["query_type"].astype(str).str.lower() == "mated"

        temp.loc[mated_mask, "rank1_correct_recomputed"] = (
            temp.loc[mated_mask, "true_identity"].astype(str)
            == temp.loc[mated_mask, "nearest_identity"].astype(str)
        )

        # Mated query: correct only when the right identity is Rank-1 AND the
        # threshold accepts that match.
        temp["decision_correct_rethresholded"] = False
        temp.loc[mated_mask, "decision_correct_rethresholded"] = (
            temp.loc[mated_mask, "rank1_correct_recomputed"]
            & temp.loc[mated_mask, "accepted_as_previous_voter"]
        )

        # Non-mated query: correct outcome is rejection.  Any accepted nearest
        # neighbour is a false positive identification.
        nonmated_mask = (
            temp["query_type"].astype(str).str.lower() == "nonmated"
        )
        temp.loc[nonmated_mask, "decision_correct_rethresholded"] = (
            ~temp.loc[nonmated_mask, "accepted_as_previous_voter"]
        )

        all_rows.append(temp)

        for requested_n in sorted(
            temp["gallery_size_requested"].dropna().astype(int).unique()
        ):
            sub = temp[
                temp["gallery_size_requested"].astype(int) == requested_n
            ]

            mated = sub[
                sub["query_type"].astype(str).str.lower() == "mated"
            ].copy()

            nonmated = sub[
                sub["query_type"].astype(str).str.lower() == "nonmated"
            ].copy()

            if len(mated) == 0 or len(nonmated) == 0:
                print(
                    f"[WARN] Skipping N={requested_n}: "
                    "both mated and non-mated rows are required."
                )
                continue

            rank1_rate = float(
                np.mean(mated["rank1_correct_recomputed"])
            )

            fnir = float(
                1.0 - np.mean(
                    mated["decision_correct_rethresholded"]
                )
            )

            fpir = float(
                np.mean(
                    nonmated["accepted_as_previous_voter"]
                )
            )

            summary_rows.append(
                {
                    "threshold_name": threshold_name,
                    "tau_l2": tau_l2,
                    "tau_sq_l2": tau_sq,
                    "gallery_size_requested": int(requested_n),
                    "gallery_size_actual": int(
                        sub["gallery_size_actual"].iloc[0]
                    ),
                    "mated_queries": int(len(mated)),
                    "nonmated_queries": int(len(nonmated)),
                    "rank1_identification_rate": rank1_rate,
                    "fnir": fnir,
                    "fpir": fpir,
                    "mated_failures": int(
                        len(mated)
                        - np.sum(
                            mated["decision_correct_rethresholded"]
                        )
                    ),
                    "nonmated_false_positive_identifications": int(
                        np.sum(
                            nonmated["accepted_as_previous_voter"]
                        )
                    ),
                }
            )

    if not all_rows:
        raise RuntimeError("No re-thresholded rows were produced.")

    all_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    details_path = (
        out_dir / "celeba_1n_query_details_rethresholded.csv"
    )
    summary_path = (
        out_dir / "celeba_1n_summary_rethresholded.csv"
    )

    all_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OUT] Re-thresholded query details: {details_path}")
    print(f"[OUT] Re-thresholded summary:       {summary_path}")

    return summary_df

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)

    parser.add_argument(
        "--model-pack",
        default="raccoon_l",
        help="Used only to sanity-check the embedding cache metadata.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260919,
        help="Base seed for deterministic image selection and repeat sampling.",
    )
    parser.add_argument(
        "--gallery-size",
        type=int,
        default=800,
        help=(
            "Calibration gallery size.  Must be smaller than the number of "
            "calibration identities so that absent identities remain for "
            "non-mated search trials."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=25,
        help="Number of repeated calibration search trials.",
    )
    parser.add_argument(
        "--nonmated-per-repeat",
        type=int,
        default=200,
        help=(
            "Maximum number of non-mated search queries per repeat.  "
            "Set 0 to use all available non-gallery identities."
        ),
    )

    parser.add_argument(
        "--target-fpirs",
        default="0.01,0.003,0.001",
        help=(
            "Comma-separated search-level FPIR targets, for example "
            "'0.01,0.003,0.001'."
        ),
    )
    parser.add_argument(
        "--primary-threshold-name",
        default="fpir_0.003",
        help=(
            "Which threshold row should be marked as the preferred operating "
            "point in the output JSON.  The name format is fpir_<value>."
        ),
    )

    parser.add_argument(
        "--existing-query-details",
        type=Path,
        default=None,
        help=(
            "Optional path to the earlier celeba_1n_query_details.csv.  "
            "If supplied, the script will re-threshold that file and write "
            "revised summary tables without rerunning FAISS search."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "calibration_1n_fpir_raccoon_l",
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grouped = read_celeba_identities(args.images, args.identity_file)
    split_data = json.loads(args.split_file.read_text(encoding="utf-8"))

    embeddings = load_embedding_cache(
        args.embedding_cache,
        expected_model_pack=args.model_pack,
    )

    (
        nonmated_dist_l2,
        mated_df,
        repeat_df,
    ) = build_repeated_search_calibration(
        grouped=grouped,
        split_data=split_data,
        embeddings=embeddings,
        seed=args.seed,
        gallery_size=args.gallery_size,
        repeats=args.repeats,
        nonmated_per_repeat=args.nonmated_per_repeat,
    )

    repeat_df.to_csv(
        args.out_dir / "repeat_level_calibration_stats.csv",
        index=False,
    )

    # Save raw non-mated nearest-neighbour distances because they are the core
    # data used for FPIR threshold selection.
    pd.DataFrame(
        {"nonmated_nearest_distance_l2": nonmated_dist_l2}
    ).to_csv(
        args.out_dir / "nonmated_nearest_distance_l2.csv",
        index=False,
    )

    mated_df.to_csv(
        args.out_dir / "mated_search_results_calibration.csv",
        index=False,
    )

    target_fpirs = [
        float(value.strip())
        for value in args.target_fpirs.split(",")
        if value.strip()
    ]

    rows = []

    # Include the historical 1.15 threshold as a reference row.
    historical_thresholds = [
        ("legacy_1.15", 1.15),
    ]

    for target_fpir in target_fpirs:
        selected = choose_threshold_for_fpir(
            nonmated_distance_l2=nonmated_dist_l2,
            target_fpir=target_fpir,
        )

        threshold_name = f"fpir_{target_fpir}"
        row = {
            "threshold_name": threshold_name,
            **selected,
            **evaluate_threshold_on_mated_queries(
                mated_df=mated_df,
                tau_l2=selected["tau_l2"],
            ),
        }
        rows.append(row)

    for threshold_name, tau_l2 in historical_thresholds:
        row = {
            "threshold_name": threshold_name,
            "target_fpir": None,
            "tau_l2": float(tau_l2),
            "tau_sq_l2": float(tau_l2 * tau_l2),
            "observed_fpir_on_calibration": float(
                np.mean(nonmated_dist_l2 <= tau_l2)
            ),
            "nonmated_queries_in_calibration": int(len(nonmated_dist_l2)),
            **evaluate_threshold_on_mated_queries(
                mated_df=mated_df,
                tau_l2=tau_l2,
            ),
        }
        rows.append(row)

    thresholds_df = pd.DataFrame(rows)
    thresholds_df.to_csv(
        args.out_dir / "fpir_threshold_candidates.csv",
        index=False,
    )

    preferred_rows = thresholds_df[
        thresholds_df["threshold_name"] == args.primary_threshold_name
    ]

    if len(preferred_rows) != 1:
        raise SystemExit(
            "The requested --primary-threshold-name was not found in the "
            "generated threshold table."
        )

    preferred = preferred_rows.iloc[0].to_dict()

    summary_json = {
        "method": (
            "search-level 1:N FPIR calibration on CelebA calibration "
            "identities using cached raccoon_l embeddings"
        ),
        "seed": args.seed,
        "gallery_size_for_calibration": args.gallery_size,
        "repeats": args.repeats,
        "nonmated_per_repeat": args.nonmated_per_repeat,
        "target_fpirs": target_fpirs,
        "primary_threshold_name": args.primary_threshold_name,
        "preferred_threshold": preferred,
        "nonmated_queries_total": int(len(nonmated_dist_l2)),
        "mated_queries_total": int(len(mated_df)),
    }

    (args.out_dir / "fpir_calibration_summary.json").write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    write_runtime_report(
        args.out_dir / "runtime_environment.json"
    )

    if args.existing_query_details is not None:
        apply_thresholds_to_existing_query_details(
            query_details_csv=args.existing_query_details,
            thresholds_df=thresholds_df,
            out_dir=args.out_dir,
        )

    print("\n=== Preferred 1:N operating point ===")
    print(
        f"threshold_name : {preferred['threshold_name']}"
    )
    print(
        f"tau_l2         : {preferred['tau_l2']:.6f}"
    )
    print(
        f"tau_sq_l2      : {preferred['tau_sq_l2']:.6f}"
    )
    print(
        "FPIR(cal)      : "
        f"{preferred['observed_fpir_on_calibration']:.6f}"
    )
    print(
        "FNIR(cal)      : "
        f"{preferred['fnir_on_calibration']:.6f}"
    )
    print(
        "Rank-1(cal)    : "
        f"{preferred['rank1_identification_rate_on_calibration']:.6f}"
    )

    print("\nOutput directory:")
    print(args.out_dir)


if __name__ == "__main__":
    main()
