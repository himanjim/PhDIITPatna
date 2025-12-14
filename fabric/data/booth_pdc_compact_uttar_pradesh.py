#!/usr/bin/env python3
# booth_pdc_compact_up.py
import csv, hashlib, secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path

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
    """24-hex pseudonymous anchor (like voter_id_star length)"""
    h = hashlib.sha1(f"{sc}|{cid}|{bid}".encode()).hexdigest()
    return h[:nhex]

def officer_id_star(sc: str, cid: str, bid: str) -> str:
    return booth_anchor_hex(sc, cid, bid, 24)

def short_fpr_hex(bytes_len: int = 16) -> str:
    """Compact fingerprint (hex)"""
    return secrets.token_hex(bytes_len)

def device_id(cid: str, bid: str, idx: int) -> str:
    return f"D{cid[2:]}{bid[2:]}{idx%1000:03d}"  # e.g., DC001B0001007

headers = ["sc","cid","bid","st","o","c","oid","r","did","f"]

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

