"""
Generate a compact Uttar Pradesh booth preload file for booth_pdc ingestion.

The script produces one CSV row per booth across all constituencies in the state.
Booth identifiers, officer anchors, and polling windows are derived
deterministically from the configured election parameters, while device
fingerprints are intentionally randomised to mimic deployment diversity. The
resulting file is suitable for preload and scale testing, not for real election
operations.
"""
#!/usr/bin/env python3
# booth_pdc_compact_up.py
import csv, hashlib, secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Generation parameters for the synthetic booth dataset. These values control the
# state scope, constituency count, polling-day window, closure ratio, and output
# path used for the compact CSV fixture.
# ===== CONFIG =====
STATE_CODE = "UP"
NUM_CONSTITUENCIES = 80             # C-001 .. C-080 (even split)
TOTAL_BOOTHS = 170_000              # 1.7 lakh
ELECTION_DAY = (2025, 11, 15)       # YYYY, M, D (pilot day)
OPEN_IST = (8, 0)                   # 08:00 IST
CLOSE_IST = (17, 0)                 # 17:00 IST
CLOSED_RATIO = 0.0                  # set to 0.01 for ~1% closed, else 0.0
OUT_CSV = Path("/tmp/booth_pdc_compact.csv")
# ===================

# Convert IST window → UTC epoch seconds (compact)
IST = timezone(timedelta(hours=5, minutes=30))
open_local = datetime(*ELECTION_DAY, *OPEN_IST, tzinfo=IST)
close_local = datetime(*ELECTION_DAY, *CLOSE_IST, tzinfo=IST)
OPEN_EPOCH = int(open_local.astimezone(timezone.utc).timestamp())
CLOSE_EPOCH = int(close_local.astimezone(timezone.utc).timestamp())

# Exact per-constituency split
assert TOTAL_BOOTHS % NUM_CONSTITUENCIES == 0, "170000 must split evenly across 80"
PER_CON = TOTAL_BOOTHS // NUM_CONSTITUENCIES  # 2125 booths per constituency

def booth_anchor_hex(sc: str, cid: str, bid: str, nhex: int = 24) -> str:
    """
    Derive a stable pseudonymous anchor for a booth-scoped entity.

    The value is used here as a compact deterministic identifier rather than as a
    cryptographic protection mechanism. Its role is to give each booth a stable
    officer-style anchor that can be regenerated across repeated runs.
    """
    h = hashlib.sha1(f"{sc}|{cid}|{bid}".encode()).hexdigest()
    return h[:nhex]

def officer_id_star(sc: str, cid: str, bid: str) -> str:
    """
    Return the pseudonymous officer identifier associated with one booth record.

    In this fixture generator, the officer anchor is deterministically tied to the
    booth rather than to a separately modelled personnel roster.
    """
    return booth_anchor_hex(sc, cid, bid, 24)

def short_fpr_hex(bytes_len: int = 16) -> str:
    """
    Return a random hexadecimal device fingerprint of the requested byte length.

    This field is intentionally non-deterministic so that the generated booth file
    does not unrealistically reuse the same device fingerprint across all booths.
    """
    return secrets.token_hex(bytes_len)

def device_id(cid: str, bid: str, idx: int) -> str:
    """
    Construct a compact synthetic device identifier from constituency, booth, and
    running index values.

    The identifier is formatted for readability and bulk generation rather than to
    emulate any specific production device-naming standard.
    """
    return f"D{cid[2:]}{bid[2:]}{idx%1000:03d}"  # e.g., DC001B0001007

# Compact column order for the generated preload CSV. The abbreviated headings are
# chosen to reduce file size during large-scale fixture generation.
headers = ["sc","cid","bid","st","o","c","oid","r","did","f"]

# Write one row per booth, distributing booths evenly across constituencies and
# marking a small configurable fraction as closed when CLOSED_RATIO is non-zero.
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(headers)
    written = 0
    idx = 0
    for i in range(1, NUM_CONSTITUENCIES + 1):
        cid = f"C-{i:03d}"
        for b in range(1, PER_CON + 1):
            bid = f"B-{b:04d}"
            st = "A"
            if CLOSED_RATIO > 0.0 and secrets.randbelow(10_000) < int(CLOSED_RATIO * 10_000):
                st = "X"
            row = [
                STATE_CODE,         # sc
                cid,                # cid
                bid,                # bid
                st,                 # st
                OPEN_EPOCH,         # o
                CLOSE_EPOCH,        # c
                officer_id_star(STATE_CODE, cid, bid),  # oid (24-hex)
                "P",                # r  (Presiding Officer)
                device_id(cid, bid, idx),               # did
                short_fpr_hex(16)   # f  (32-hex; ~16 bytes)
            ]
            w.writerow(row)
            written += 1
            idx += 1

print(f"Wrote {written} rows to {OUT_CSV}")

