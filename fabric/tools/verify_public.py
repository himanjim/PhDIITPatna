#!/usr/bin/env python3
"""verify_public.py - the public (Tier A) verifier for one constituency.

The tool takes a verification pack produced by tools/export_freeze.js and
carries out the checks that the key-ceremony component lists in Section 1.14,
using published material only. It never needs access to a private data
collection, to the voter roll, or to any trustee key.

    (0) pack integrity        every file matches the digest in the manifest
    (i) freeze commitment     HR recomputed from S equals the published HR
   (ii) ballot commitments    SHA256(C_s) equals the hC recorded for that serial
  (iii) encrypted aggregate   the product of the current ciphertexts mod n^2
                              equals the published C_tally
   (iv) decryption            trustee partial decryptions D_i and proofs pi_i,
                              the deterministic combine, and the decoding of the
                              option counts
    (v) authorisation         the quorum certificate QC_TVP over h_TVP
   (vi) cast-or-audit         with the released K_day: every commitment in the
                              cast set was a "cast" coin, every published opening
                              was an "open" coin, the two sets are disjoint, the
                              query log is covered, and each opening re-derives
                              to its commitment

Threshold decryption and its proofs are specified but not implemented in the
present prototype, so checks (iv) and (v) normally report "not available"
rather than failing. That is deliberate: a verifier must be able to tell the
difference between a check that failed and a check that could not be run, and
the summary states which properties are therefore not yet established.

Usage
    python3 verify_public.py --pack PACKDIR
    python3 verify_public.py --pack PACKDIR --key-day <hex>
    python3 verify_public.py --pack PACKDIR --key-day <hex> --total <decimal> --json

Exit status: 0 when every check that could be run passed, 1 when a check
failed, 2 on a usage or input error.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOL = "verify_public.py"
VERSION = "1.0.0"
HR_DOMAIN = "HR_LIST_V1"
AUDIT_COIN_DOMAIN = b"AUDIT_COIN_V1"
AUDIT_KEY_COMMIT_DOMAIN = b"AUDIT_KEY_COMMIT_V1"
S_FIELDS = ("serial", "hC", "txID", "epoch", "castTime", "status", "reason")

PASS, FAIL, NA = "pass", "fail", "not available"


# --------------------------------------------------------------------------- #
# canonical JSON (RFC 8785, the subset the pack uses)
# --------------------------------------------------------------------------- #
def jcs(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("JCS: non-finite number")
        return repr(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, str):
        return _jcs_string(value)
    if isinstance(value, list):
        return "[" + ",".join(jcs(v) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted((k for k in value if value[k] is not None or True), key=lambda k: k)
        return "{" + ",".join(_jcs_string(k) + ":" + jcs(value[k]) for k in items) + "}"
    raise ValueError("JCS: unsupported value of type %s" % type(value).__name__)


def _jcs_string(s: str) -> str:
    out = ['"']
    for ch in s:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif code == 0x08:
            out.append("\\b")
        elif code == 0x09:
            out.append("\\t")
        elif code == 0x0A:
            out.append("\\n")
        elif code == 0x0C:
            out.append("\\f")
        elif code == 0x0D:
            out.append("\\r")
        elif code < 0x20:
            out.append("\\u%04x" % code)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def parse_hex(value: str, what: str) -> int:
    text = (value or "").strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if text == "":
        raise ValueError("%s is empty" % what)
    try:
        return int(text, 16)
    except ValueError as exc:
        raise ValueError("%s is not hexadecimal: %r" % (what, value)) from exc


def hex_forms(value: int) -> List[str]:
    """Hex spellings a client may have submitted for the same integer.

    The contract hashes the ciphertext string exactly as it was submitted, so a
    verifier that re-derives the integer has to try the spellings that the
    clients in this system produce: canonical lowercase with no leading zero,
    the same padded to an even number of digits, and the 0x-prefixed form.
    """
    canonical = format(value, "x")
    padded = canonical if len(canonical) % 2 == 0 else "0" + canonical
    forms = [canonical, padded, "0x" + canonical, "0x" + padded]
    seen, out = set(), []
    for f in forms:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    ref: str
    name: str
    status: str
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pack:
    directory: Path
    manifest: Dict[str, Any]
    raw: Dict[str, str]
    S: List[Dict[str, str]]
    ballots: List[Dict[str, str]]
    openings: List[Dict[str, Any]]
    querylog: List[Dict[str, str]]
    params: Dict[str, Any]
    public_key: Dict[str, str]
    options: List[str]
    tally: Optional[Dict[str, Any]]
    results: Optional[Dict[str, Any]]


def load_pack(directory: Path) -> Pack:
    def read(name: str, required: bool = True) -> Tuple[Optional[Any], Optional[str]]:
        path = directory / name
        if not path.exists():
            if required:
                raise FileNotFoundError("the pack has no %s" % name)
            return None, None
        text = path.read_text(encoding="utf-8")
        return json.loads(text), text

    manifest, _ = read("manifest.json")
    raw: Dict[str, str] = {}
    data: Dict[str, Any] = {}
    for name in ("S.json", "ballots.json", "openings.json", "querylog.json", "params.json", "publickey.json", "options.json"):
        value, text = read(name, required=name in ("S.json", "ballots.json"))
        if value is None:
            value = [] if name.endswith(("S.json", "ballots.json", "openings.json", "querylog.json", "options.json")) else {}
            text = None
        data[name] = value
        if text is not None:
            raw[name] = text
    for name in ("tally.json", "results.json"):
        value, text = read(name, required=False)
        data[name] = value
        if text is not None:
            raw[name] = text

    return Pack(
        directory=directory,
        manifest=manifest,
        raw=raw,
        S=data["S.json"],
        ballots=data["ballots.json"],
        openings=data["openings.json"],
        querylog=data["querylog.json"],
        params=data["params.json"],
        public_key=data["publickey.json"],
        options=data["options.json"],
        tally=data["tally.json"],
        results=data["results.json"],
    )


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #
def check_integrity(pack: Pack) -> Check:
    listed = pack.manifest.get("files", {})
    if not listed:
        return Check("0", "pack integrity", NA, "the manifest lists no file digests")
    bad, missing = [], []
    for name, entry in sorted(listed.items()):
        text = pack.raw.get(name)
        if text is None:
            missing.append(name)
            continue
        digest = sha256_hex(text.rstrip("\n"))
        if digest != entry.get("sha256"):
            bad.append(name)
    if missing or bad:
        detail = ""
        if missing:
            detail += "missing: %s. " % ", ".join(missing)
        if bad:
            detail += "digest mismatch: %s." % ", ".join(bad)
        return Check("0", "pack integrity", FAIL, detail.strip())
    return Check("0", "pack integrity", PASS, "%d files match the manifest" % len(listed))


def check_freeze_commitment(pack: Pack, expected: Optional[str]) -> Check:
    rows = []
    for row in pack.S:
        rows.append({f: str(row.get(f, "")) for f in S_FIELDS})
    recomputed = sha256_hex(HR_DOMAIN + jcs(rows))
    published = (pack.manifest.get("HR") or "").lower()
    target = (expected or published).lower()
    if not target:
        return Check("i", "freeze commitment HR", NA, "no published HR to compare with", {"recomputed": recomputed})
    if recomputed != target:
        return Check("i", "freeze commitment HR", FAIL, "recomputed %s, published %s" % (recomputed, target))
    if expected and published and expected.lower() != published:
        return Check("i", "freeze commitment HR", FAIL, "the manifest and the expected value disagree")
    return Check("i", "freeze commitment HR", PASS, "HR = %s over %d tuples" % (recomputed, len(rows)))


def check_ballot_commitments(pack: Pack) -> Check:
    checked, bad, no_ciphertext = 0, [], []
    for row in pack.ballots:
        if row.get("status") != "current":
            continue
        enc = row.get("encOneHex") or ""
        if enc == "":
            no_ciphertext.append(row.get("serial"))
            continue
        checked += 1
        if sha256_hex(enc) != (row.get("hC") or "").lower():
            bad.append(row.get("serial"))
    if bad:
        return Check("ii", "ballot commitments", FAIL, "%d of %d current ballots do not match their commitment: %s" % (len(bad), checked, ", ".join(map(str, bad[:5]))))
    if no_ciphertext:
        return Check("ii", "ballot commitments", FAIL, "%d current ballots carry no published ciphertext: %s" % (len(no_ciphertext), ", ".join(map(str, no_ciphertext[:5]))))
    if checked == 0:
        return Check("ii", "ballot commitments", NA, "the pack holds no current ballot with a ciphertext")
    return Check("ii", "ballot commitments", PASS, "%d current ballots hash to their recorded hC" % checked)


def check_aggregate(pack: Pack) -> Check:
    if not pack.tally:
        return Check("iii", "encrypted aggregate", NA, "the pack carries no published C_tally (tally.json)")
    published = pack.tally.get("cTally") or pack.tally.get("C_tally") or pack.tally.get("aggregate")
    if isinstance(published, dict):
        published = published.get("PACKED") or next(iter(published.values()), None)
    if not published:
        return Check("iii", "encrypted aggregate", NA, "tally.json holds no aggregate value")
    try:
        n = parse_hex(pack.public_key.get("n", ""), "public key n")
    except ValueError as exc:
        return Check("iii", "encrypted aggregate", NA, str(exc))
    n2_hex = pack.public_key.get("n2") or ""
    n2 = parse_hex(n2_hex, "public key n2") if n2_hex else n * n

    product, counted = 1, 0
    for row in pack.ballots:
        if row.get("status") != "current" or not row.get("encOneHex"):
            continue
        product = (product * parse_hex(row["encOneHex"], "ciphertext of %s" % row.get("serial"))) % n2
        counted += 1
    if counted == 0:
        return Check("iii", "encrypted aggregate", NA, "no current ciphertext to multiply")
    want = parse_hex(str(published), "published C_tally") % n2
    if product != want:
        return Check("iii", "encrypted aggregate", FAIL, "the product of %d ciphertexts differs from the published aggregate" % counted)
    return Check("iii", "encrypted aggregate", PASS, "the product of %d ciphertexts equals the published C_tally" % counted)


def check_decryption(pack: Pack, total: Optional[int]) -> List[Check]:
    out: List[Check] = []
    have_shares = (pack.directory / "transcript_3.json").exists()
    if not have_shares:
        out.append(Check("iv-a", "partial decryptions and proofs", NA, "no transcript_3 in the pack: threshold decryption is specified but not implemented"))
    else:
        out.append(Check("iv-a", "partial decryptions and proofs", NA, "transcript_3 is present, but this tool does not yet verify the proof format"))

    if total is None:
        out.append(Check("iv-b", "decoded option counts", NA, "no decrypted total supplied (--total)"))
        return out

    slot_bits = int(pack.params.get("PACKED_SLOT_BITS") or 0)
    slots = int(pack.params.get("PACKED_SLOTS") or 0)
    if slot_bits <= 0 or slots <= 0:
        out.append(Check("iv-b", "decoded option counts", NA, "the pack does not publish PACKED_SLOT_BITS and PACKED_SLOTS"))
        return out

    counts = decode_packed(total, slot_bits, slots)
    ballots_counted = sum(1 for row in pack.ballots if row.get("status") == "current" and row.get("encOneHex"))
    if sum(counts) != ballots_counted:
        out.append(Check("iv-b", "decoded option counts", FAIL, "the decoded counts sum to %d but %d ballots were counted" % (sum(counts), ballots_counted), {"counts": counts}))
        return out
    published = None
    if pack.results:
        published = pack.results.get("counts") or pack.results.get("m")
    if published is not None:
        if [int(x) for x in published] != counts:
            out.append(Check("iv-b", "decoded option counts", FAIL, "the decoded counts differ from the published result", {"decoded": counts, "published": published}))
            return out
        out.append(Check("iv-b", "decoded option counts", PASS, "the decoded counts equal the published result: %s" % counts, {"counts": counts}))
        return out
    out.append(Check("iv-b", "decoded option counts", PASS, "counts decode cleanly and sum to the number of counted ballots: %s" % counts, {"counts": counts}))
    return out


def decode_packed(total: int, slot_bits: int, slots: int) -> List[int]:
    mask = (1 << slot_bits) - 1
    return [(total >> (slot_bits * j)) & mask for j in range(slots)]


def check_authorisation(pack: Pack) -> Check:
    if (pack.directory / "transcript_4.json").exists():
        return Check("v", "quorum certificate QC_TVP", NA, "transcript_4 is present, but this tool does not yet verify trustee signatures")
    return Check("v", "quorum certificate QC_TVP", NA, "no transcript_4 in the pack: result authorisation is specified but not implemented")


def audit_coin(key_day: bytes, h_c: str, one_in: int) -> bool:
    if len(key_day) < 16:
        raise ValueError("K_day is %d bytes, the contract requires at least 16" % len(key_day))
    if one_in < 1:
        raise ValueError("the audit rate r must be at least 1")
    mac = hmac.new(key_day, AUDIT_COIN_DOMAIN + h_c.strip().lower().encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(mac[:8], "big") % one_in == 0


def check_audit(pack: Pack, key_day_hex: Optional[str]) -> List[Check]:
    one_in = int(pack.params.get("AUDIT_ONE_IN") or 0)
    commit = (pack.params.get("AUDIT_KEY_COMMIT") or "").strip().lower()
    if not key_day_hex:
        return [Check("vi", "cast-or-audit reconciliation", NA, "K_day was not supplied (--key-day); it is published only after the poll closes")]
    try:
        key_day = bytes.fromhex(key_day_hex.strip().lower().removeprefix("0x"))
    except ValueError:
        return [Check("vi", "cast-or-audit reconciliation", FAIL, "K_day is not hexadecimal")]
    out: List[Check] = []

    if commit:
        got = hashlib.sha256(AUDIT_KEY_COMMIT_DOMAIN + key_day).hexdigest()
        if got != commit:
            return [Check("vi-a", "audit key commitment", FAIL, "the revealed K_day does not match the published commitment")]
        out.append(Check("vi-a", "audit key commitment", PASS, "the revealed K_day matches the commitment published before the poll"))
    else:
        out.append(Check("vi-a", "audit key commitment", NA, "the pack publishes no commitment to K_day"))

    if one_in < 1:
        out.append(Check("vi-b", "cast-or-audit reconciliation", NA, "the pack does not publish the audit rate AUDIT_ONE_IN"))
        return out

    # Every row of S is a ballot a terminal recorded, whatever status it was
    # later given, so the coin check covers excluded ballots too: an exclusion
    # decided after the freeze cannot excuse a terminal that ignored the coin.
    cast = [str(row.get("hC", "")).lower() for row in pack.S]
    opened = [str(o.get("hC", "")).lower() for o in pack.openings]
    queried = [str(q.get("hC", "")).lower() for q in pack.querylog]

    should_have_opened = [h for h in cast if audit_coin(key_day, h, one_in)]
    opened_without_cause = [h for h in opened if not audit_coin(key_day, h, one_in)]
    both = sorted(set(cast) & set(opened))
    uncovered = [h for h in queried if h not in set(cast) | set(opened)]

    problems = []
    if should_have_opened:
        problems.append("%d cast ballots that the coin had selected for opening" % len(should_have_opened))
    if opened_without_cause:
        problems.append("%d openings the coin had not selected" % len(opened_without_cause))
    if both:
        problems.append("%d commitments that appear both as cast and as opened" % len(both))
    if uncovered:
        problems.append("%d query-log entries that are neither cast nor opened" % len(uncovered))
    evidence = {
        "castBallots": len(cast),
        "openings": len(opened),
        "queries": len(queried),
        "shouldHaveOpened": should_have_opened[:20],
        "openedWithoutCause": opened_without_cause[:20],
        "castAndOpened": both[:20],
        "queriedButUnaccounted": uncovered[:20],
    }
    if problems:
        out.append(Check("vi-b", "cast-or-audit reconciliation", FAIL, "; ".join(problems), evidence))
    else:
        out.append(Check("vi-b", "cast-or-audit reconciliation", PASS, "%d cast ballots and %d openings reconcile at rate 1 in %d" % (len(cast), len(opened), one_in), evidence))

    out.append(check_openings_rederive(pack))
    return out


def check_openings_rederive(pack: Pack) -> Check:
    """Re-derive every published opening and check that it hashes to its commitment."""
    if not pack.openings:
        return Check("vi-c", "openings re-derive to their commitments", NA, "the pack holds no audit opening")
    try:
        n = parse_hex(pack.public_key.get("n", ""), "public key n")
    except ValueError as exc:
        return Check("vi-c", "openings re-derive to their commitments", NA, str(exc))
    g_hex = pack.public_key.get("g") or ""
    g = parse_hex(g_hex, "public key g") if g_hex else n + 1
    n2_hex = pack.public_key.get("n2") or ""
    n2 = parse_hex(n2_hex, "public key n2") if n2_hex else n * n
    slot_bits = int(pack.params.get("PACKED_SLOT_BITS") or 0)
    if slot_bits <= 0:
        return Check("vi-c", "openings re-derive to their commitments", NA, "the pack does not publish PACKED_SLOT_BITS")

    bad = []
    for opening in pack.openings:
        m = 1 << (slot_bits * int(opening["optionIndex"]))
        r = parse_hex(opening["randomnessHex"], "randomness for %s" % opening["hC"])
        c = (pow(g, m, n2) * pow(r, n, n2)) % n2
        want = str(opening["hC"]).lower()
        if not any(sha256_hex(form) == want for form in hex_forms(c)):
            bad.append(opening["hC"])
    if bad:
        return Check("vi-c", "openings re-derive to their commitments", FAIL, "%d of %d openings do not re-derive: %s" % (len(bad), len(pack.openings), ", ".join(bad[:5])))
    return Check("vi-c", "openings re-derive to their commitments", PASS, "all %d openings re-derive to their published commitment" % len(pack.openings))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_checks(pack: Pack, key_day_hex: Optional[str], expected_hr: Optional[str], total: Optional[int]) -> List[Check]:
    checks = [check_integrity(pack), check_freeze_commitment(pack, expected_hr), check_ballot_commitments(pack), check_aggregate(pack)]
    checks.extend(check_decryption(pack, total))
    checks.append(check_authorisation(pack))
    checks.extend(check_audit(pack, key_day_hex))
    return checks


def summarise(checks: List[Check]) -> str:
    passed = sum(1 for c in checks if c.status == PASS)
    failed = sum(1 for c in checks if c.status == FAIL)
    absent = sum(1 for c in checks if c.status == NA)
    lines = ["%s %s" % (TOOL, VERSION), ""]
    width = max(len(c.name) for c in checks)
    for c in checks:
        lines.append("  (%-4s) %-*s  %-13s %s" % (c.ref, width, c.name, c.status.upper(), c.detail))
    lines.append("")
    lines.append("  %d passed, %d failed, %d not available" % (passed, failed, absent))
    if failed == 0 and absent:
        lines.append("  the properties behind the checks marked 'not available' are not established by this run")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog=TOOL, description="public (Tier A) verifier for one constituency")
    parser.add_argument("--pack", required=True, help="directory written by export_freeze.js")
    parser.add_argument("--key-day", dest="key_day", help="K_day in hex, published after the poll closes")
    parser.add_argument("--expect-hr", dest="expect_hr", help="freeze commitment HR anchored on the ledger")
    parser.add_argument("--total", help="the decrypted packed total, as a decimal or 0x-prefixed integer")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    try:
        pack = load_pack(Path(args.pack))
    except (OSError, ValueError) as exc:
        sys.stderr.write("%s: %s\n" % (TOOL, exc))
        return 2

    total = None
    if args.total:
        text = args.total.strip().lower()
        try:
            total = int(text, 16) if text.startswith("0x") else int(text)
        except ValueError:
            sys.stderr.write("%s: --total is neither decimal nor 0x-hexadecimal\n" % TOOL)
            return 2

    try:
        checks = run_checks(pack, args.key_day, args.expect_hr, total)
    except (ValueError, KeyError) as exc:
        sys.stderr.write("%s: %s\n" % (TOOL, exc))
        return 2

    if args.json:
        print(json.dumps({"tool": TOOL, "version": VERSION, "pack": str(pack.directory),
                          "checks": [c.__dict__ for c in checks],
                          "failed": sum(1 for c in checks if c.status == FAIL)}, indent=1))
    else:
        print(summarise(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
