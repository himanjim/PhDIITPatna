#!/usr/bin/env python3
"""
celeba_rethreshold_existing_results.py

Fast recovery utility for the completed FPIR calibration run.

Use this when `celeba_1n_fpir_recalibration.py` already completed threshold
selection but failed only while re-thresholding the older query-details CSV.

The script does NOT:
    * run InsightFace;
    * load embedding caches;
    * build a FAISS index;
    * repeat FPIR calibration.

It only reads:
    1. the already-produced `fpir_threshold_candidates.csv`;
    2. the existing `celeba_1n_query_details.csv`;
    3. the companion `celeba_1n_summary.csv` when available.

It then applies each threshold to the saved nearest-neighbour distances and
writes the revised 1:N summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalise_gallery_columns(
    query_details_csv: Path,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert either historical gallery-size schema to one internal schema."""
    if {
        "gallery_size_requested",
        "gallery_size_actual",
    }.issubset(df.columns):
        return df.copy()

    if "gallery_size" not in df.columns:
        raise ValueError(
            "Query-details CSV has neither 'gallery_size' nor the newer "
            "'gallery_size_requested/gallery_size_actual' columns."
        )

    work = df.copy()
    work["gallery_size_requested"] = work["gallery_size"].astype(int)

    # The user's completed run stores exact actual gallery sizes here:
    # requested 1000 -> actual 998, requested 2500 -> actual 2497, etc.
    companion = query_details_csv.parent / "celeba_1n_summary.csv"
    mapping = {}

    if companion.is_file():
        summary = pd.read_csv(companion)

        if {
            "requested_gallery_size",
            "actual_gallery_size",
        }.issubset(summary.columns):
            mapping = dict(
                zip(
                    summary["requested_gallery_size"].astype(int),
                    summary["actual_gallery_size"].astype(int),
                )
            )
            print(f"[SCHEMA] Using actual gallery sizes from: {companion}")

    if mapping:
        work["gallery_size_actual"] = (
            work["gallery_size_requested"]
            .map(mapping)
            .fillna(work["gallery_size_requested"])
            .astype(int)
        )
    else:
        print(
            "[WARN] Companion summary not found or incompatible. "
            "Using requested gallery size as actual."
        )
        work["gallery_size_actual"] = work[
            "gallery_size_requested"
        ].astype(int)

    return work


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--thresholds",
        type=Path,
        required=True,
        help="Path to fpir_threshold_candidates.csv",
    )
    parser.add_argument(
        "--query-details",
        type=Path,
        required=True,
        help="Path to existing celeba_1n_query_details.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.thresholds.is_file():
        raise SystemExit(
            f"Threshold table not found: {args.thresholds}"
        )

    if not args.query_details.is_file():
        raise SystemExit(
            f"Query-details file not found: {args.query_details}"
        )

    thresholds = pd.read_csv(args.thresholds)
    details = pd.read_csv(args.query_details)

    required_threshold_columns = {
        "threshold_name",
        "tau_l2",
    }

    missing = required_threshold_columns.difference(thresholds.columns)
    if missing:
        raise SystemExit(
            f"Threshold file is missing columns: {sorted(missing)}"
        )

    required_detail_columns = {
        "query_type",
        "true_identity",
        "nearest_identity",
        "distance_l2",
        "distance_sq_l2",
    }

    missing = required_detail_columns.difference(details.columns)
    if missing:
        raise SystemExit(
            f"Query-details file is missing columns: {sorted(missing)}"
        )

    details = normalise_gallery_columns(
        args.query_details,
        details,
    )

    summary_rows = []
    revised_rows = []

    for _, threshold in thresholds.iterrows():
        name = str(threshold["threshold_name"])
        tau_l2 = float(threshold["tau_l2"])
        tau_sq = tau_l2 * tau_l2

        work = details.copy()
        work["threshold_name"] = name
        work["tau_l2_rethresholded"] = tau_l2
        work["tau_sq_l2_rethresholded"] = tau_sq

        work["accepted_as_previous_voter_rethresholded"] = (
            work["distance_l2"].astype(float) <= tau_l2
        )

        query_type = work["query_type"].astype(str).str.lower()
        mated_mask = query_type == "mated"
        nonmated_mask = query_type == "nonmated"

        work["rank1_correct_recomputed"] = False
        work.loc[mated_mask, "rank1_correct_recomputed"] = (
            work.loc[mated_mask, "true_identity"].astype(str)
            == work.loc[mated_mask, "nearest_identity"].astype(str)
        )

        work["decision_correct_rethresholded"] = False

        work.loc[mated_mask, "decision_correct_rethresholded"] = (
            work.loc[mated_mask, "rank1_correct_recomputed"]
            & work.loc[
                mated_mask,
                "accepted_as_previous_voter_rethresholded",
            ]
        )

        work.loc[nonmated_mask, "decision_correct_rethresholded"] = (
            ~work.loc[
                nonmated_mask,
                "accepted_as_previous_voter_rethresholded",
            ]
        )

        revised_rows.append(work)

        for requested_n in sorted(
            work["gallery_size_requested"].astype(int).unique()
        ):
            sub = work[
                work["gallery_size_requested"].astype(int)
                == requested_n
            ]

            mated = sub[
                sub["query_type"].astype(str).str.lower()
                == "mated"
            ]

            nonmated = sub[
                sub["query_type"].astype(str).str.lower()
                == "nonmated"
            ]

            if len(mated) == 0 or len(nonmated) == 0:
                continue

            rank1_rate = float(
                np.mean(mated["rank1_correct_recomputed"])
            )

            duplicate_detection_rate = float(
                np.mean(
                    mated["decision_correct_rethresholded"]
                )
            )

            fnir = 1.0 - duplicate_detection_rate

            fpir = float(
                np.mean(
                    nonmated[
                        "accepted_as_previous_voter_rethresholded"
                    ]
                )
            )

            summary_rows.append(
                {
                    "threshold_name": name,
                    "tau_l2": tau_l2,
                    "tau_sq_l2": tau_sq,
                    "gallery_size_requested": int(requested_n),
                    "gallery_size_actual": int(
                        sub["gallery_size_actual"].iloc[0]
                    ),
                    "mated_queries": int(len(mated)),
                    "nonmated_queries": int(len(nonmated)),
                    "rank1_identification_rate": rank1_rate,
                    "duplicate_detection_rate": (
                        duplicate_detection_rate
                    ),
                    "fnir": fnir,
                    "fpir": fpir,
                    "mated_failures": int(
                        len(mated)
                        - np.sum(
                            mated[
                                "decision_correct_rethresholded"
                            ]
                        )
                    ),
                    "nonmated_false_positive_identifications": int(
                        np.sum(
                            nonmated[
                                "accepted_as_previous_voter_rethresholded"
                            ]
                        )
                    ),
                }
            )

    revised = pd.concat(revised_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    revised_path = (
        args.out_dir
        / "celeba_1n_query_details_rethresholded.csv"
    )
    summary_path = (
        args.out_dir
        / "celeba_1n_summary_rethresholded.csv"
    )

    revised.to_csv(revised_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== Re-thresholding complete ===")
    print(f"Thresholds applied: {len(thresholds)}")
    print(f"Original query rows: {len(details):,}")
    print(f"Revised details:     {revised_path}")
    print(f"Revised summary:     {summary_path}")

    print("\n=== Revised 1:N summary ===")
    display_columns = [
        "threshold_name",
        "tau_l2",
        "gallery_size_requested",
        "gallery_size_actual",
        "rank1_identification_rate",
        "fnir",
        "fpir",
    ]
    print(summary[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
