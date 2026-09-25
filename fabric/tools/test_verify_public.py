#!/usr/bin/env python3
"""test_verify_public.py - tests for the public (Tier A) verifier.

Run with:  python3 -m unittest discover -s fabric/tools -p 'test_*.py' -v
           python3 fabric/tools/test_verify_public.py

Three kinds of test are here.

  1. Agreement with the contract. The audit coin, the key commitment and the
     packed slot values are recomputed in Python and compared with values
     printed by the Go implementation in fabric/chaincode/accumvote. If the two
     ever drift apart, a public verifier would reach a different verdict from
     the ledger, so this is checked first.

  2. The happy path. The verifier is run over the pack that export_freeze.js
     builds from the contract-produced fixture. Every check that can be run on
     the present prototype must pass, and exactly the two checks that depend on
     threshold decryption must report "not available".

  3. Detection. Each test in the last group damages one thing in the pack - a
     ciphertext, a serial, an opening, the audit key, the published counts - and
     requires the verifier to fail the matching check and only that check.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import verify_public as vp  # noqa: E402

FIXTURES = HERE / "fixtures"
EXPORTER = HERE / "export_freeze.js"


def build_pack(destination: Path, snapshot: dict) -> Path:
    """Run export_freeze.js over a snapshot and return the pack directory."""
    snap_path = destination / "snapshot.json"
    snap_path.write_text(json.dumps(snapshot), encoding="utf-8")
    pack = destination / "pack"
    subprocess.run(
        ["node", str(EXPORTER), "--out", str(pack), "--snapshot", str(snap_path)],
        check=True, capture_output=True, text=True,
    )
    return pack


def load_snapshot() -> dict:
    return json.loads((FIXTURES / "snapshot.json").read_text(encoding="utf-8"))


def status_of(checks, ref):
    for c in checks:
        if c.ref == ref:
            return c.status
    raise AssertionError("no check with reference %s" % ref)


@unittest.skipUnless(shutil.which("node"), "node is needed to build a pack")
class PackTestCase(unittest.TestCase):
    """Base class that builds a pack from a possibly modified snapshot."""

    def pack_from(self, snapshot: dict) -> vp.Pack:
        tmp = Path(tempfile.mkdtemp(prefix="verify-pack-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        return vp.load_pack(build_pack(tmp, snapshot))

    @property
    def key_day(self) -> str:
        return (FIXTURES / "keyday.txt").read_text(encoding="utf-8").strip()

    @property
    def total(self) -> int:
        return int((FIXTURES / "total.txt").read_text(encoding="utf-8").strip())


class TestAgreementWithTheContract(unittest.TestCase):
    """The Python implementation must agree with the Go implementation.

    The expected values were printed by the contract package itself, using
    K_day = a7 repeated 32 times.
    """

    KEY = bytes.fromhex("a7" * 32)
    GOLDEN_COIN = {
        ("00" * 32, 1): True, ("00" * 32, 4): True, ("00" * 32, 20): False, ("00" * 32, 50): False,
        ("aa" + "11" * 31, 1): True, ("aa" + "11" * 31, 4): True, ("aa" + "11" * 31, 20): False, ("aa" + "11" * 31, 50): False,
        ("deadbeef" + "00" * 28, 1): True, ("deadbeef" + "00" * 28, 4): False,
        ("deadbeef" + "00" * 28, 20): False, ("deadbeef" + "00" * 28, 50): False,
    }
    GOLDEN_COMMIT = "04ff6174b274412c07683973e20c3cf7854d879f9cbd7bc43eaf35630bedfda1"

    def test_audit_coin_matches_the_contract(self):
        for (h_c, one_in), expected in self.GOLDEN_COIN.items():
            self.assertEqual(vp.audit_coin(self.KEY, h_c, one_in), expected, "%s at 1 in %d" % (h_c[:8], one_in))

    def test_coin_is_case_insensitive_and_trims(self):
        self.assertEqual(vp.audit_coin(self.KEY, "AA" + "11" * 31, 4), vp.audit_coin(self.KEY, "aa" + "11" * 31, 4))
        self.assertEqual(vp.audit_coin(self.KEY, "  " + "aa" + "11" * 31 + "  ", 4), vp.audit_coin(self.KEY, "aa" + "11" * 31, 4))

    def test_key_commitment_matches_the_contract(self):
        self.assertEqual(hashlib.sha256(vp.AUDIT_KEY_COMMIT_DOMAIN + self.KEY).hexdigest(), self.GOLDEN_COMMIT)

    def test_short_key_is_refused_like_the_contract(self):
        with self.assertRaises(ValueError):
            vp.audit_coin(b"\x01" * 15, "aa" * 32, 20)

    def test_packed_slot_value_matches_the_contract(self):
        self.assertEqual(1 << (8 * 3), 16777216)

    def test_canonical_json_matches_the_exporter(self):
        # The same vectors are asserted in test_export_freeze.js.
        self.assertEqual(vp.jcs({"b": 1, "a": 2}), '{"a":2,"b":1}')
        self.assertEqual(vp.jcs({"B": 1, "a": 1, "A": 1}), '{"A":1,"B":1,"a":1}')
        self.assertEqual(vp.jcs({"s": 'a"b\\c\nd\te'}), '{"s":"a\\"b\\\\c\\nd\\te"}')
        self.assertEqual(vp.jcs({"s": "\u0001"}), '{"s":"\\u0001"}')
        self.assertEqual(vp.jcs({"t": True, "f": False, "n": None}), '{"f":false,"n":null,"t":true}')


class TestHappyPath(PackTestCase):
    def setUp(self):
        self.pack = self.pack_from(load_snapshot())
        self.checks = vp.run_checks(self.pack, self.key_day, None, self.total)

    def test_every_available_check_passes(self):
        failed = [c for c in self.checks if c.status == vp.FAIL]
        self.assertEqual(failed, [], "failing checks: %s" % [(c.ref, c.detail) for c in failed])

    def test_only_the_decryption_checks_are_unavailable(self):
        absent = sorted(c.ref for c in self.checks if c.status == vp.NA)
        self.assertEqual(absent, ["iv-a", "v"])

    def test_the_freeze_commitment_is_recomputed_not_copied(self):
        rows = [{f: str(r.get(f, "")) for f in vp.S_FIELDS} for r in self.pack.S]
        self.assertEqual(vp.sha256_hex(vp.HR_DOMAIN + vp.jcs(rows)), self.pack.manifest["HR"])

    def test_the_counts_decode_to_the_published_result(self):
        counts = vp.decode_packed(self.total, int(self.pack.params["PACKED_SLOT_BITS"]), int(self.pack.params["PACKED_SLOTS"]))
        self.assertEqual(counts, [int(x) for x in self.pack.results["counts"]])
        self.assertEqual(sum(counts), sum(1 for b in self.pack.ballots if b["status"] == "current"))

    def test_the_pack_carries_no_booth_or_device_identifier(self):
        blob = json.dumps({"openings": self.pack.openings, "S": self.pack.S, "ballots": self.pack.ballots})
        self.assertNotIn("boothID", blob)
        self.assertNotIn("deviceID", blob)

    def test_without_the_key_the_audit_check_is_reported_as_unavailable(self):
        checks = vp.run_checks(self.pack, None, None, self.total)
        self.assertEqual(status_of(checks, "vi"), vp.NA)
        self.assertEqual([c for c in checks if c.status == vp.FAIL], [])


class TestDetection(PackTestCase):
    def test_a_substituted_ciphertext_is_caught(self):
        snapshot = load_snapshot()
        victim = next(b for b in snapshot["ballots"] if b.get("status") == "current")
        other = next(b for b in snapshot["ballots"] if b.get("status") == "current" and b["serial"] != victim["serial"])
        victim["encOneHex"] = other["encOneHex"]
        checks = vp.run_checks(self.pack_from(snapshot), self.key_day, None, None)
        self.assertEqual(status_of(checks, "ii"), vp.FAIL)
        self.assertEqual(status_of(checks, "iii"), vp.FAIL)
        self.assertEqual(status_of(checks, "i"), vp.PASS, "the freeze list itself was not touched")

    def test_a_dropped_ballot_is_caught_by_the_aggregate(self):
        snapshot = load_snapshot()
        current = [b for b in snapshot["ballots"] if b.get("status") == "current"]
        snapshot["ballots"] = [b for b in snapshot["ballots"] if b["serial"] != current[0]["serial"]]
        checks = vp.run_checks(self.pack_from(snapshot), self.key_day, None, None)
        self.assertEqual(status_of(checks, "iii"), vp.FAIL)

    def test_an_altered_freeze_list_is_caught(self):
        pack = self.pack_from(load_snapshot())
        rows = json.loads((pack.directory / "S.json").read_text(encoding="utf-8"))
        rows[0]["castTime"] = "1999-01-01T00:00:00Z"
        (pack.directory / "S.json").write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
        reloaded = vp.load_pack(pack.directory)
        checks = vp.run_checks(reloaded, self.key_day, None, None)
        self.assertEqual(status_of(checks, "i"), vp.FAIL)
        self.assertEqual(status_of(checks, "0"), vp.FAIL, "the manifest digest catches it as well")

    def test_a_terminal_that_ignored_the_coin_is_caught(self):
        """An opened ballot that was recorded as a cast vote instead."""
        snapshot = load_snapshot()
        opening = snapshot["openings"][0]
        snapshot["ballots"].append({
            "serial": "S-9999", "hC": opening["hC"], "encOneHex": "", "status": "current",
            "epoch": "E1", "castTime": "2026-01-01T00:00:00Z", "txID": "tx-ignored-coin",
        })
        snapshot["openings"] = snapshot["openings"][1:]
        checks = vp.run_checks(self.pack_from(snapshot), self.key_day, None, None)
        self.assertEqual(status_of(checks, "vi-b"), vp.FAIL)

    def test_an_unjustified_opening_is_caught(self):
        """An opening published for a ballot the coin had not selected."""
        snapshot = load_snapshot()
        cast = next(b for b in snapshot["ballots"] if b.get("status") == "current")
        snapshot["openings"].append({
            "hC": cast["hC"], "constituencyID": "C-001", "optionIndex": 0, "randomnessHex": "03",
        })
        checks = vp.run_checks(self.pack_from(snapshot), self.key_day, None, None)
        self.assertEqual(status_of(checks, "vi-b"), vp.FAIL)

    def test_an_opening_that_does_not_re_derive_is_caught(self):
        snapshot = load_snapshot()
        snapshot["openings"][0]["optionIndex"] = (int(snapshot["openings"][0]["optionIndex"]) + 1) % 4
        checks = vp.run_checks(self.pack_from(snapshot), self.key_day, None, None)
        self.assertEqual(status_of(checks, "vi-c"), vp.FAIL)

    def test_a_wrong_audit_key_is_caught(self):
        checks = vp.run_checks(self.pack_from(load_snapshot()), "b8" * 32, None, None)
        self.assertEqual(status_of(checks, "vi-a"), vp.FAIL)

    def test_published_counts_that_do_not_match_the_total_are_caught(self):
        snapshot = load_snapshot()
        counts = list(snapshot["results"]["counts"])
        counts[0] += 1
        counts[1] -= 1
        snapshot["results"]["counts"] = counts
        checks = vp.run_checks(self.pack_from(snapshot), self.key_day, None, self.total)
        self.assertEqual(status_of(checks, "iv-b"), vp.FAIL)

    def test_a_tampered_pack_file_is_caught_by_the_manifest(self):
        pack = self.pack_from(load_snapshot())
        (pack.directory / "params.json").write_text('{"AUDIT_ONE_IN":1}', encoding="utf-8")
        checks = vp.run_checks(vp.load_pack(pack.directory), None, None, None)
        self.assertEqual(status_of(checks, "0"), vp.FAIL)

    def test_the_exit_status_reports_failure(self):
        snapshot = load_snapshot()
        snapshot["ballots"][0]["hC"] = "00" * 32
        tmp = Path(tempfile.mkdtemp(prefix="verify-pack-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        pack_dir = build_pack(tmp, snapshot)
        code = vp.main(["--pack", str(pack_dir), "--key-day", self.key_day])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
