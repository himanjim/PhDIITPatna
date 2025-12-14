#!/usr/bin/env python3
"""
Generate two deterministic CSV fixtures for the election bench:

1) candidates.csv  (public world state)
   - Columns: constituency_id, candidate_id, candidate_name, party_code, symbol_hash
   - candidate_id is UNIQUE ACROSS THE ENTIRE DATASET (e.g., cand-000001 …)
   - symbol_hash is a deterministic hex tag derived from state/constituency/candidate_id

2) voters.csv  (presence-only PDC)
   - Columns: voter_id_star, constituency_id, state_code, status
   - voter_id_star is UNIQUE ACROSS THE ENTIRE DATASET
     -> We derive a 24-hex (96-bit) ID from SHA-256(state || global_serial),
        and in the (astronomically unlikely) event of a collision, we bump a suffix
        until uniqueness is achieved. This remains deterministic for a given input set.

The script is deterministic given the same arguments.
"""

import argparse
import csv
import hashlib
import os
from typing import Iterable, Tuple, Set


# -----------------------------
# Helpers
# -----------------------------
def dhash(*parts: Iterable[str], n: int = 16) -> str:
    """
    Deterministic short hex digest.
    Concatenate the input parts with '|' and return the first n hex chars of SHA-256.
    """
    msg = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(msg).hexdigest()[:n]


def unique_hex_id(seed_parts: Tuple[str, ...], n: int, used: Set[str]) -> str:
    """
    Produce a unique n-hex ID from SHA-256(seed_parts). If (extremely unlikely) the
    truncated hex collides with an existing ID in 'used', append a numeric suffix
    and re-hash until uniqueness is achieved. Deterministic for a given dataset.
    """
    base = dhash(*seed_parts, n=n)
    if base not in used:
        used.add(base)
        return base
    # Collision fallback: deterministically derive a new one by adding a suffix
    suffix = 1
    while True:
        cand = dhash(*seed_parts, f"#{suffix}", n=n)
        if cand not in used:
            used.add(cand)
            return cand
        suffix += 1


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="ST01",
                    help="State code written to voters.csv (and used in hashes)")
    ap.add_argument("--constituencies", type=int, default=5,
                    help="Number of constituencies to generate (C-001..)")
    ap.add_argument("--candidates", type=int, default=4,
                    help="Candidates per constituency")
    ap.add_argument("--voters", type=int, default=20000,
                    help="Total voters across all constituencies")
    ap.add_argument("--outdir", default="data/generated",
                    help="Output directory for CSVs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------------------------------------
    # 1) Generate candidates.csv (global-unique candidate_id)
    # ---------------------------------------------------------
    cand_csv = os.path.join(args.outdir, "candidates.csv")
    total_candidates = args.constituencies * args.candidates

    # We give every candidate a GLOBAL sequential ID: cand-000001 .. cand-XYZ
    # This guarantees uniqueness across constituencies.
    with open(cand_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["constituency_id", "candidate_id", "candidate_name", "party_code", "symbol_hash"])

        global_cand_serial = 0
        for ci in range(1, args.constituencies + 1):
            const_id = f"C-{ci:03d}"
            for _ in range(args.candidates):
                global_cand_serial += 1
                candidate_id = f"cand-{global_cand_serial:06d}"     # zero-padded, globally unique
                candidate_name = f"Candidate {global_cand_serial}"  # name reflects the global serial
                party_code = f"P{((global_cand_serial - 1) % 6) + 1}"  # cycle P1..P6 globally
                symbol_hash = dhash(args.state, const_id, candidate_id, "symbol", n=32)
                w.writerow([const_id, candidate_id, candidate_name, party_code, symbol_hash])

    # Sanity check: the last ID should equal the total number of candidates
    assert global_cand_serial == total_candidates, "Candidate serial mismatch"

    # ---------------------------------------------------------
    # 2) Generate voters.csv (global-unique voter_id_star)
    # ---------------------------------------------------------
    voters_csv = os.path.join(args.outdir, "voters.csv")
    per_const = args.voters // args.constituencies       # even spread
    leftover = args.voters % args.constituencies         # first 'leftover' constituencies get +1

    used_voters: Set[str] = set()  # track IDs to enforce uniqueness

    with open(voters_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["voter_id_star", "constituency_id", "state_code", "status"])

        global_voter_serial = 0
        for ci in range(1, args.constituencies + 1):
            const_id = f"C-{ci:03d}"
            count = per_const + (1 if ci <= leftover else 0)

            for _ in range(count):
                global_voter_serial += 1
                # Seed is (state, global_serial). Because global_serial is unique,
                # the derived 24-hex is effectively unique; we still guard against
                # truncation collisions with unique_hex_id().
                voter_id_star = unique_hex_id((args.state, str(global_voter_serial), "voter"), n=24, used=used_voters)
                w.writerow([voter_id_star, const_id, args.state, "eligible"])

    assert global_voter_serial == args.voters, "Voter serial mismatch"
    assert len(used_voters) == args.voters, "Duplicate voter_id_star detected (should be impossible)"

    print(f"OK: wrote {cand_csv} (total candidates={total_candidates})")
    print(f"OK: wrote {voters_csv} (total voters={args.voters})")


if __name__ == "__main__":
    main()

