#!/usr/bin/env python3
"""
pgu_beard_raccoon.py

PGU-Face beard/stubble robustness experiment for the user's final extracted
dataset.

Actual local structure
----------------------
The five original archives have already been merged.  The working dataset is:

    data/pgu/
        1/
            im01.jpg            # one observed spelling
            img02.jpg
            img03.jpg
            img04.jpg
            text.txt
        2/
            img01.jpg
            img02.jpg
            img03.jpg
            img04.jpg
            text.txt
        ...
        224/
            ...

The immediate numeric parent directory is the subject identity.

Why this parser is deliberately permissive about filenames
-----------------------------------------------------------
The released PGU-Face copy is not completely uniform.  The result audit showed
valid condition files using several forms, including:

    img01.jpg
    im01.jpg
    img 01.jpg
    image 01.jpg

The same naming variation may occur for conditions 02, 03 and 04.  These are
filename inconsistencies, not different experimental conditions.  The parser
therefore accepts all of the following forms, case-insensitively:

    img01, img 01, img_01, img-01
    image01, image 01, image_01, image-01
    im01, im 01, im_01, im-01

Only condition numbers 1..4 are accepted.  Other image files are written to
the audit CSV rather than guessed.

Published experimental conditions
----------------------------------
    condition 1 : nearly clean-shaven
    condition 2 : nearly clean-shaven + sunglasses
    condition 3 : beard/stubble
    condition 4 : beard/stubble + sunglasses

Primary election-security experiment
------------------------------------
For clean -> beard:

    1. condition-1 image is treated as the person's earlier vote and inserted
       into the FAISS gallery;
    2. condition-3 image is treated as the same person's later appearance;
    3. that later image is searched 1:N against the entire gallery;
    4. the repeat voter is detected only when:
           nearest subject == true subject
       AND squared-L2 distance <= the frozen threshold.

The threshold comes from the independent CelebA calibration experiment.  PGU
is therefore a held-out robustness test rather than a source of threshold
tuning.

The script can also be run with --audit-only.  This parses and validates the
dataset without loading InsightFace or calculating embeddings.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd

from experiment_common import (
    FaceEmbedder,
    embed_paths_with_cache,
    wilson_interval,
    write_runtime_report,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The published condition number is the only semantic element we need from
# the filename.  The prefix and separator are allowed to vary because the
# distributed dataset contains inconsistent spellings.
#
# Examples accepted:
#   img01       -> condition 1
#   im01        -> condition 1
#   img 01      -> condition 1
#   image 01    -> condition 1
#   IMG_03      -> condition 3
#   image-04    -> condition 4
CONDITION_PATTERN = re.compile(
    r"^(?:img|image|im)[\s_-]*0?([1-4])$",
    flags=re.IGNORECASE,
)


def subject_sort_key(subject_id: str):
    """
    Sort numeric subject IDs numerically.

    Without this helper, ordinary string sorting would place "10" before "2".
    """
    try:
        return (0, int(subject_id))
    except ValueError:
        return (1, subject_id)


def parse_condition(path: Path):
    """
    Return condition 1..4 for a recognised PGU filename.

    Returning None is preferable to guessing.  The caller records unrecognised
    image files in the audit report for manual inspection.
    """
    match = CONDITION_PATTERN.fullmatch(path.stem.strip())
    if match is None:
        return None
    return int(match.group(1))


def scan_pgu(root: Path):
    """
    Parse the final flattened PGU-Face directory.

    Ground truth is taken from the immediate parent directory:
        data/pgu/137/img03.jpg
              -> subject 137
              -> condition 3

    Returns
    -------
    subjects
        dict[subject_id][condition] = image path

    audit_rows
        one row for every image-like file encountered, including any file that
        could not be interpreted.

    duplicate_rows
        conflicts where two files map to the same subject and condition.
        Such conflicts are never resolved silently.
    """
    subjects: Dict[str, Dict[int, str]] = defaultdict(dict)
    audit_rows: List[dict] = []
    duplicate_rows: List[dict] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        # text.txt and the age spreadsheet are metadata, not face images.
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        subject_id = path.parent.name.strip()
        condition = parse_condition(path)

        row = {
            "path": str(path),
            "subject_id": subject_id,
            "filename": path.name,
            "condition": condition,
            "status": "ok",
        }

        # The actual released subject folders are numeric.  Requiring this
        # protects against accidentally interpreting an extraction directory
        # or unrelated folder as a person identity.
        if not subject_id.isdigit():
            row["status"] = "non_numeric_subject_directory"
            audit_rows.append(row)
            continue

        if condition is None:
            row["status"] = "image_filename_not_recognised"
            audit_rows.append(row)
            continue

        if condition in subjects[subject_id]:
            duplicate_rows.append(
                {
                    "subject_id": subject_id,
                    "condition": condition,
                    "first_path": subjects[subject_id][condition],
                    "duplicate_path": str(path),
                }
            )
            row["status"] = "duplicate_subject_condition"
            audit_rows.append(row)
            continue

        subjects[subject_id][condition] = str(path)
        audit_rows.append(row)

    return dict(subjects), audit_rows, duplicate_rows


def validate_subjects(subjects: Dict[str, Dict[int, str]]):
    """
    Build one validation row per subject.

    The complete PGU-Face release should have four recognised condition images
    for every subject.
    """
    rows = []

    for subject_id in sorted(subjects, key=subject_sort_key):
        conditions = subjects[subject_id]
        present = sorted(conditions)

        missing = [
            condition
            for condition in (1, 2, 3, 4)
            if condition not in conditions
        ]

        rows.append(
            {
                "subject_id": subject_id,
                "conditions_present": ",".join(map(str, present)),
                "conditions_missing": ",".join(map(str, missing)),
                "condition_count": len(present),
                "has_clean_condition_1": 1 in conditions,
                "has_clean_glasses_condition_2": 2 in conditions,
                "has_beard_condition_3": 3 in conditions,
                "has_beard_glasses_condition_4": 4 in conditions,
                "complete_four_conditions": len(present) == 4,
            }
        )

    return rows


def write_dataset_audit(
    out_dir: Path,
    audit_rows: List[dict],
    validation_rows: List[dict],
    duplicate_rows: List[dict],
):
    """Write parser diagnostics before any expensive model inference starts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(audit_rows).to_csv(
        out_dir / "pgu_file_audit.csv",
        index=False,
    )

    pd.DataFrame(validation_rows).to_csv(
        out_dir / "pgu_subject_validation.csv",
        index=False,
    )

    # Explicit columns keep the file readable even when there are no duplicate
    # rows and pandas would otherwise write an entirely blank CSV.
    pd.DataFrame(
        duplicate_rows,
        columns=[
            "subject_id",
            "condition",
            "first_path",
            "duplicate_path",
        ],
    ).to_csv(
        out_dir / "pgu_duplicate_mappings.csv",
        index=False,
    )


def print_dataset_summary(
    subjects: Dict[str, Dict[int, str]],
    audit_rows: List[dict],
    validation_rows: List[dict],
    duplicate_rows: List[dict],
):
    """Print the integrity figures that should be checked before publication."""
    mapped_images = sum(
        len(conditions)
        for conditions in subjects.values()
    )

    complete_subjects = sum(
        1
        for row in validation_rows
        if row["complete_four_conditions"]
    )

    unexpected_images = sum(
        1
        for row in audit_rows
        if row["status"] != "ok"
    )

    print("\n=== PGU-Face integrity check ===")
    print(f"Parsed subjects:              {len(subjects)}")
    print(f"Mapped condition images:      {mapped_images}")
    print(f"Subjects with all 4 images:   {complete_subjects}")
    print(f"Unexpected image files:       {unexpected_images}")
    print(f"Duplicate mappings:           {len(duplicate_rows)}")

    if len(subjects) == 224 and mapped_images == 896 and complete_subjects == 224:
        print("[DATA] Complete 224-subject / 896-image structure recognised.")
    else:
        print(
            "[WARN] Final publication run should ideally recognise "
            "224 subjects, 896 images and 224 complete four-condition subjects."
        )

    return {
        "parsed_subjects": len(subjects),
        "mapped_images": mapped_images,
        "complete_subjects": complete_subjects,
        "unexpected_images": unexpected_images,
        "duplicate_mappings": len(duplicate_rows),
    }


def exact_search(gallery: np.ndarray, queries: np.ndarray):
    """
    Run exact 1-nearest-neighbour FAISS search.

    IndexFlatL2 returns SQUARED Euclidean distance.  Therefore the true-L2
    threshold tau is converted to tau**2 for the final decision.
    """
    index = faiss.IndexFlatL2(gallery.shape[1])
    index.add(gallery.astype(np.float32, copy=False))

    distances, indices = index.search(
        queries.astype(np.float32, copy=False),
        1,
    )

    return distances[:, 0], indices[:, 0]


def evaluate_direction(
    direction_name: str,
    gallery_condition: int,
    query_condition: int,
    subjects: Dict[str, Dict[int, str]],
    embeddings: Dict[str, np.ndarray],
    tau_l2: float,
    threshold_name: str,
):
    """
    Evaluate one appearance-change direction against a complete 1:N gallery.

    A subject contributes only when both required conditions exist and both
    images have usable embeddings.  This keeps the reported identification
    rate separate from face-detection/acquisition failures.
    """
    eligible = [
        subject_id
        for subject_id, conditions in subjects.items()
        if gallery_condition in conditions
        and query_condition in conditions
    ]
    eligible.sort(key=subject_sort_key)

    gallery_ids = []
    gallery_paths = []
    gallery_vectors = []

    for subject_id in eligible:
        path = subjects[subject_id][gallery_condition]

        if path in embeddings:
            gallery_ids.append(subject_id)
            gallery_paths.append(path)
            gallery_vectors.append(embeddings[path])

    if len(gallery_ids) < 2:
        raise RuntimeError(
            f"Too few valid gallery embeddings for {direction_name}."
        )

    gallery = np.vstack(gallery_vectors).astype(np.float32)
    gallery_set = set(gallery_ids)

    query_ids = []
    query_paths = []
    query_vectors = []

    for subject_id in eligible:
        if subject_id not in gallery_set:
            continue

        path = subjects[subject_id][query_condition]

        if path in embeddings:
            query_ids.append(subject_id)
            query_paths.append(path)
            query_vectors.append(embeddings[path])

    if not query_vectors:
        raise RuntimeError(f"No valid queries for {direction_name}.")

    queries = np.vstack(query_vectors).astype(np.float32)
    distances_sq, nearest_indices = exact_search(
        gallery,
        queries,
    )

    tau_sq = float(tau_l2 * tau_l2)

    rank1_correct_count = 0
    duplicate_detected_count = 0
    wrong_nearest_count = 0
    correct_but_over_threshold_count = 0

    detail_rows = []

    for true_id, query_path, distance_sq, nearest_index in zip(
        query_ids,
        query_paths,
        distances_sq,
        nearest_indices,
    ):
        nearest_index = int(nearest_index)
        nearest_id = gallery_ids[nearest_index]

        rank1_correct = nearest_id == true_id
        within_threshold = float(distance_sq) <= tau_sq

        # In the election de-duplication interpretation, correct identification
        # alone is not enough: the match must also clear the operating
        # threshold before the earlier vote is flagged.
        duplicate_detected = rank1_correct and within_threshold

        rank1_correct_count += int(rank1_correct)
        duplicate_detected_count += int(duplicate_detected)
        wrong_nearest_count += int(not rank1_correct)
        correct_but_over_threshold_count += int(
            rank1_correct and not within_threshold
        )

        detail_rows.append(
            {
                "threshold_name": threshold_name,
                "direction": direction_name,
                "true_subject": true_id,
                "query_path": query_path,
                "nearest_subject": nearest_id,
                "nearest_gallery_path": gallery_paths[nearest_index],
                "distance_sq_l2": float(distance_sq),
                "distance_l2": float(
                    np.sqrt(max(0.0, float(distance_sq)))
                ),
                "tau_l2": float(tau_l2),
                "tau_sq_l2": tau_sq,
                "rank1_correct": bool(rank1_correct),
                "within_threshold": bool(within_threshold),
                "duplicate_detected_correctly": bool(duplicate_detected),
            }
        )

    query_count = len(detail_rows)
    failure_count = query_count - duplicate_detected_count

    fnir = failure_count / query_count
    fnir_low, fnir_high = wilson_interval(
        failure_count,
        query_count,
    )

    summary = {
        "threshold_name": threshold_name,
        "direction": direction_name,
        "gallery_condition": gallery_condition,
        "query_condition": query_condition,
        "gallery_size": len(gallery_ids),
        "queries": query_count,
        "tau_l2": float(tau_l2),
        "tau_sq_l2": tau_sq,
        "rank1_identification_rate": (
            rank1_correct_count / query_count
        ),
        "duplicate_detection_rate": (
            duplicate_detected_count / query_count
        ),
        "fnir": fnir,
        "fnir_95ci_low": fnir_low,
        "fnir_95ci_high": fnir_high,
        "wrong_nearest_identity_count": wrong_nearest_count,
        "correct_identity_but_over_threshold_count": (
            correct_but_over_threshold_count
        ),
        "distance_sq_mean": float(np.mean(distances_sq)),
        "distance_sq_median": float(np.median(distances_sq)),
        "distance_sq_p95": float(np.percentile(distances_sq, 95)),
        "distance_sq_max": float(np.max(distances_sq)),
    }

    return summary, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="PGU root containing numeric subject folders 1 ... 224.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        help=(
            "calibration_summary.json from the independent CelebA calibration. "
            "Not required with --audit-only."
        ),
    )

    parser.add_argument("--model-pack", default="raccoon_l")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--strict-one-face", action="store_true")
    parser.add_argument("--include-sunglasses", action="store_true")

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache") / "pgu_raccoon_l.npz",
    )
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "pgu_beard_raccoon_l",
    )
    parser.add_argument(
        "--also-test-legacy-1-15",
        action="store_true",
        help="Also evaluate the earlier tau_L2=1.15 as a secondary reference.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Only parse and validate the dataset. "
            "No model is loaded and no embeddings are calculated."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Stop unless exactly 224 subjects, 896 mapped condition images "
            "and 224 complete subjects are recognised."
        ),
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.root.is_dir():
        raise SystemExit(f"PGU-Face root not found: {args.root}")

    # ---------------- Parse and audit before touching the GPU ---------------- #

    subjects, audit_rows, duplicate_rows = scan_pgu(args.root)
    validation_rows = validate_subjects(subjects)

    write_dataset_audit(
        args.out_dir,
        audit_rows,
        validation_rows,
        duplicate_rows,
    )

    counts = print_dataset_summary(
        subjects,
        audit_rows,
        validation_rows,
        duplicate_rows,
    )

    if duplicate_rows:
        raise SystemExit(
            "Duplicate subject/condition mappings were found. "
            "Inspect pgu_duplicate_mappings.csv before continuing."
        )

    if args.require_complete:
        complete = (
            counts["parsed_subjects"] == 224
            and counts["mapped_images"] == 896
            and counts["complete_subjects"] == 224
            and counts["unexpected_images"] == 0
            and counts["duplicate_mappings"] == 0
        )

        if not complete:
            raise SystemExit(
                "Dataset did not pass the strict 224/896 completeness check. "
                "Inspect the audit CSV files."
            )

    if args.audit_only:
        print(
            "\n[AUDIT] Completed without loading InsightFace. "
            "Inspect pgu_file_audit.csv and pgu_subject_validation.csv."
        )
        return

    if args.calibration_json is None:
        raise SystemExit(
            "--calibration-json is required unless --audit-only is used."
        )

    if not args.calibration_json.is_file():
        raise SystemExit(
            f"Calibration JSON not found: {args.calibration_json}"
        )

    calibration = json.loads(
        args.calibration_json.read_text(encoding="utf-8")
    )
    primary_tau = float(calibration["selected"]["tau_l2"])

    # ---------------- Select images for the experiment ---------------- #

    primary_subjects = [
        subject_id
        for subject_id, conditions in subjects.items()
        if 1 in conditions and 3 in conditions
    ]
    primary_subjects.sort(key=subject_sort_key)

    print(
        "\nSubjects with condition 1 (clean) and condition 3 "
        f"(beard/stubble): {len(primary_subjects)}"
    )

    if len(primary_subjects) < 2:
        raise SystemExit(
            "Not enough subjects have both clean and beard conditions."
        )

    required_paths = set()

    for subject_id in primary_subjects:
        required_paths.add(subjects[subject_id][1])
        required_paths.add(subjects[subject_id][3])

    if args.include_sunglasses:
        for conditions in subjects.values():
            if 2 in conditions and 4 in conditions:
                required_paths.add(conditions[2])
                required_paths.add(conditions[4])

    print(f"Images selected for embedding: {len(required_paths)}")

    # ---------------- Generate/reuse InsightFace embeddings ---------------- #

    embedder = FaceEmbedder(
        model_pack=args.model_pack,
        device="cpu" if args.cpu else "cuda",
        strict_one_face=args.strict_one_face,
    )

    embeddings, embedding_failures = embed_paths_with_cache(
        sorted(required_paths),
        embedder,
        args.cache,
        recompute=args.recompute,
        progress_every=25,
    )

    pd.DataFrame(embedding_failures).to_csv(
        args.out_dir / "embedding_failures.csv",
        index=False,
    )

    print(f"Successful embeddings:        {len(embeddings)}")
    print(f"Embedding failures:           {len(embedding_failures)}")

    # ---------------- Run the 1:N appearance-change tests ---------------- #

    directions = [
        ("clean_to_beard", 1, 3),
        ("beard_to_clean", 3, 1),
    ]

    if args.include_sunglasses:
        directions.extend(
            [
                ("clean_glasses_to_beard_glasses", 2, 4),
                ("beard_glasses_to_clean_glasses", 4, 2),
            ]
        )

    thresholds = [("recalibrated", primary_tau)]

    if args.also_test_legacy_1_15:
        thresholds.append(("legacy_1.15", 1.15))

    summaries = []
    query_details = []

    for threshold_name, tau_l2 in thresholds:
        for direction_name, gallery_condition, query_condition in directions:
            summary, rows = evaluate_direction(
                direction_name=direction_name,
                gallery_condition=gallery_condition,
                query_condition=query_condition,
                subjects=subjects,
                embeddings=embeddings,
                tau_l2=tau_l2,
                threshold_name=threshold_name,
            )

            summaries.append(summary)
            query_details.extend(rows)

            print(f"\n[{threshold_name} / {direction_name}]")
            print(f"  Gallery size        : {summary['gallery_size']}")
            print(f"  Queries             : {summary['queries']}")
            print(
                "  Rank-1              : "
                f"{summary['rank1_identification_rate']:.6f}"
            )
            print(
                "  Duplicate detection : "
                f"{summary['duplicate_detection_rate']:.6f}"
            )
            print(f"  FNIR                : {summary['fnir']:.6f}")
            print(
                "  Wrong nearest ID    : "
                f"{summary['wrong_nearest_identity_count']}"
            )
            print(
                "  Correct ID > tau    : "
                f"{summary['correct_identity_but_over_threshold_count']}"
            )

    pd.DataFrame(summaries).to_csv(
        args.out_dir / "pgu_beard_summary.csv",
        index=False,
    )

    pd.DataFrame(query_details).to_csv(
        args.out_dir / "pgu_beard_query_details.csv",
        index=False,
    )

    write_runtime_report(
        args.out_dir / "runtime_environment.json",
        model_pack=args.model_pack,
    )

    run_config = {
        "dataset": "PGU-Face",
        "local_layout": "<root>/<numeric_subject_id>/<condition_image>",
        "accepted_filename_regex": CONDITION_PATTERN.pattern,
        "model_pack": args.model_pack,
        "calibration_json": str(args.calibration_json),
        "primary_tau_l2": primary_tau,
        "parsed_subjects": counts["parsed_subjects"],
        "mapped_condition_images": counts["mapped_images"],
        "complete_four_condition_subjects": counts["complete_subjects"],
        "primary_clean_beard_subjects": len(primary_subjects),
        "include_sunglasses": args.include_sunglasses,
        "embedding_failures": len(embedding_failures),
    }

    (args.out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print("\n=== Output files ===")
    for filename in (
        "pgu_beard_summary.csv",
        "pgu_beard_query_details.csv",
        "pgu_file_audit.csv",
        "pgu_subject_validation.csv",
        "pgu_duplicate_mappings.csv",
        "embedding_failures.csv",
        "runtime_environment.json",
        "run_config.json",
    ):
        print(f"  {args.out_dir / filename}")


if __name__ == "__main__":
    main()
