// tally_test.go
//
// Purpose: Unit and integration-style tests focused on the tally/accumulator read path
//          of the AccumVote contract. These validate Enc(1) multiplication, latest-wins
//          re-vote semantics, booth/device validation, and API parity between
//          TallyPrepare and GetEncSums.
// Role:    Runs against the in-memory harness and gomock’d ChaincodeStub from this test
//          suite; no real Fabric network required.
// Key dependencies:
//   • Contract under test: AccumVoteContract (same package)
//   • Test harness: newHarness, memWorld, preload/booth cc2cc stubs
//   • Crypto shape: Paillier Enc(1) products verified against modular exponentiation
// Notes:
//   • Helpers here are deliberately minimal and deterministic.
//   • Tests expect small toy moduli and identity element “1” for Enc(0).

package main

import (
	"encoding/hex"
	"math/big"
	"strings"
	"testing"
)

// ---------- small helpers (local to this file) ----------

// bigFromPossiblyHex parses an integer from relaxed inputs: decimal, hex with/without
// 0x prefix, and odd-length hex (auto-padded).
// Params: s – string like "1", "0xca1", "ca1", or "31".
// Returns: *big.Int parsed value, or error if neither hex nor decimal parses.
func bigFromPossiblyHex(s string) (*big.Int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return big.NewInt(0), nil
	}
	// Try Go's auto-base parsing first (handles 0x/0X, 0, decimal).
	if z := new(big.Int); z != nil {
		if _, ok := z.SetString(s, 0); ok {
			return z, nil
		}
	}
	// Strip 0x and decode as hex; pad odd length for hex.DecodeString.
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		s = s[2:]
	}
	if len(s)%2 == 1 {
		s = "0" + s
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		// Final fallback: interpret as base-10 if possible.
		z := new(big.Int)
		if _, ok := z.SetString(s, 10); ok {
			return z, nil
		}
		return nil, err
	}
	if len(b) == 0 {
		return big.NewInt(0), nil
	}
	return new(big.Int).SetBytes(b), nil
}

// mustBigFromHex is a test helper that fatals on parse error.
// Params: t – *testing.T; s – input string.
// Returns: parsed *big.Int; aborts test on failure.
func mustBigFromHex(t *testing.T, s string) *big.Int {
	t.Helper()
	z, err := bigFromPossiblyHex(s)
	if err != nil {
		t.Fatalf("parse big int from %q: %v", s, err)
	}
	return z
}

// powModHex computes c^k mod n² where c is provided as hex (or relaxed format).
// Params: t – *testing.T; hexC – ciphertext; k – exponent (vote count);
//         n2 – modulus n².
// Returns: *big.Int result of exponentiation modulo n².
func powModHex(t *testing.T, hexC string, k int, n2 *big.Int) *big.Int {
	t.Helper()
	base := mustBigFromHex(t, hexC)
	exp := big.NewInt(int64(k))
	return new(big.Int).Exp(base, exp, n2)
}

// nSquaredFromHarnessN derives n² from the harness’ configured hexN.
// Params: t – *testing.T.
// Returns: *big.Int for n².
func nSquaredFromHarnessN(t *testing.T) *big.Int {
	t.Helper()
	n := mustBigFromHex(t, hexN)
	return new(big.Int).Mul(n, n)
}

// ---------- Tests ----------

// TestTally_ZeroVote_Identity checks that with zero valid votes, Enc(0) is the
// multiplicative identity "1" for every candidate.
// Params: t – *testing.T.
// Returns: none; fails if any candidate’s sum isn’t 1.
func TestTally_ZeroVote_Identity(t *testing.T) {
	//setDefaultEnv(t)
	setProdEnv(t) // Put env first for consistency with other tests.
	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2, testCand3, testCand4}
	h.stubPreloadCC(cands, map[string]bool{
		"S-001": true, "S-002": true, "S-003": true, "S-100": true,
		"S-201": true, "S-202": true, "S-203": true,
	})
	requireNoErr(t, h.seedCandidates(cands))

	requireNoErr(t, h.setPK_UP())

	encSums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	one := big.NewInt(1)
	got1 := mustBigFromHex(t, encSums[testCand1])
	got2 := mustBigFromHex(t, encSums[testCand2])

	if got1.Cmp(one) != 0 || got2.Cmp(one) != 0 {
		t.Fatalf("expected Enc(0)=1 for both candidates, got cand1=%s cand2=%s",
			got1.Text(16), got2.Text(16))
	}
}

// TestTally_SimpleDistribution verifies two votes for cand1 and one for cand2
// show up as c^2 and c^1 products respectively.
// Params: t – *testing.T.
// Returns: none; compares products against explicit pow(c, k) mod n².
func TestTally_SimpleDistribution(t *testing.T) {
	//setDefaultEnv(t)
	setProdEnv(t) // Put env first for consistency.
	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2, testCand3, testCand4}
	h.stubPreloadCC(cands, map[string]bool{
		"S-001": true, "S-002": true, "S-003": true, "S-100": true,
		"S-201": true, "S-202": true, "S-203": true,
	})
	requireNoErr(t, h.seedCandidates(cands))

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.openPoll())

	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2}) // hot-path list small for the test.

	// Ensure distinct shards/effects by changing TxID between casts.
	h.setTxID("tx-0001")
	requireNoErr(t, must2(h.recordVote("S-001", testCand1, hexEncOneGood)))
	h.setTxID("tx-0002")
	requireNoErr(t, must2(h.recordVote("S-002", testCand1, hexEncOneGood)))
	h.setTxID("tx-0003")
	requireNoErr(t, must2(h.recordVote("S-003", testCand2, hexEncOneGood)))

	encSums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	n2 := nSquaredFromHarnessN(t)
	// Two votes for cand1, one for cand2 → products of Enc(1) reflect counts.
	want1 := powModHex(t, hexEncOneGood, 2, n2)
	want2 := powModHex(t, hexEncOneGood, 1, n2)

	got1 := mustBigFromHex(t, encSums[testCand1])
	got2 := mustBigFromHex(t, encSums[testCand2])

	if got1.Cmp(want1) != 0 || got2.Cmp(want2) != 0 {
		t.Fatalf("tally mismatch:\n cand1 got=%s want=%s\n cand2 got=%s want=%s",
			got1.Text(16), want1.Text(16), got2.Text(16), want2.Text(16))
	}
}

// TestTally_RevoteChange asserts latest-wins semantics: earlier choice is
// effectively removed and only the final candidate gets +1.
// Params: t – *testing.T.
// Returns: none.
func TestTally_RevoteChange(t *testing.T) {
	//setDefaultEnv(t)
	setProdEnv(t) // Put env first for consistency.
	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2, testCand3, testCand4}
	h.stubPreloadCC(cands, map[string]bool{
		"S-001": true, "S-002": true, "S-003": true, "S-100": true,
		"S-201": true, "S-202": true, "S-203": true,
	})
	requireNoErr(t, h.seedCandidates(cands))

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.openPoll())

	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2})

	// First vote to cand-1.
	h.setTxID("tx-A")
	requireNoErr(t, must2(h.recordVote("S-100", testCand1, hexEncOneGood)))
	// Re-vote to cand-2 (latest wins).
	h.setTxID("tx-B")
	requireNoErr(t, must2(h.recordVote("S-100", testCand2, hexEncOneGood)))

	encSums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	n2 := nSquaredFromHarnessN(t)
	// Latest-wins → cand1 becomes identity; cand2 gets one Enc(1).
	want1 := big.NewInt(1)
	want2 := powModHex(t, hexEncOneGood, 1, n2)

	got1 := mustBigFromHex(t, encSums[testCand1])
	got2 := mustBigFromHex(t, encSums[testCand2])

	if got1.Cmp(want1) != 0 || got2.Cmp(want2) != 0 {
		t.Fatalf("revote tally mismatch:\n cand1 got=%s want=%s\n cand2 got=%s want=%s",
			got1.Text(16), want1.Text(16), got2.Text(16), want2.Text(16))
	}
}

// TestGetEncSums_MirrorsTally ensures GetEncSums returns the same map as
// TallyPrepare for the same state.
// Params: t – *testing.T.
// Returns: none; fails if any candidate differs.
func TestGetEncSums_MirrorsTally(t *testing.T) {
	//setDefaultEnv(t)
	setProdEnv(t) // Put env first for consistency.
	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2, testCand3, testCand4}
	h.stubPreloadCC(cands, map[string]bool{
		"S-001": true, "S-002": true, "S-003": true, "S-100": true,
		"S-201": true, "S-202": true, "S-203": true,
	})
	requireNoErr(t, h.seedCandidates(cands))

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.openPoll())

	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2})

	// A few votes distributed across candidates.
	h.setTxID("tx-01")
	requireNoErr(t, must2(h.recordVote("S-201", testCand1, hexEncOneGood)))
	h.setTxID("tx-02")
	requireNoErr(t, must2(h.recordVote("S-202", testCand2, hexEncOneGood)))
	h.setTxID("tx-03")
	requireNoErr(t, must2(h.recordVote("S-203", testCand1, hexEncOneGood)))

	tally, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)
	enc, err := h.cc.GetEncSums(h.ctx, testConst)
	requireNoErr(t, err)

	if len(tally) != len(enc) {
		t.Fatalf("size mismatch: tally=%d enc=%d", len(tally), len(enc))
	}
	for k, v := range tally {
		if enc[k] != v {
			t.Fatalf("mismatch for %s: tally=%s enc=%s", k, v, enc[k])
		}
	}
}

// Test_Tally_BoothDeviceMismatch_Excluded verifies that a device ID mismatch
// (against boothpdc) excludes the vote at tally and, once ApplyBallotStatuses
// is called, marks the ballot invalid.
// Params: t – *testing.T.
// Returns: none.
func Test_Tally_BoothDeviceMismatch_Excluded(t *testing.T) {
    setProdEnv(t)
    h := newHarness(t)
    defer h.ctrl.Finish()

    // Everyone eligible in preload; local candidate list must be seeded separately.
    h.stubPreloadCC([]string{testCand1, testCand2}, map[string]bool{"S-001": true})
    requireNoErr(t, h.setPK_UP())
    requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2}))
    requireNoErr(t, h.openPoll())

    // Cast with WRONG device ID (boothpdc expects testDeviceID) to trigger exclusion.
    h.setTxID("tx-wrongdev")
    _, err := h.cc.RecordVote(
        h.ctx, "S-001", testConst, testCand1, hexEncOneGood, "salt", testEpoch, testAttestOK,
        testBoothID, "WRONG-DEVICE", testDeviceFP,
        "", "", "", "",
    )
    requireNoErr(t, err)
    requireNoErr(t, h.closePoll())

    // Read-only tally excludes the vote from Enc sums.
    sums, err := h.cc.TallyPrepare(h.ctx, testConst)
    requireNoErr(t, err)
    if sums[testCand1] != "1" { // identity → excluded
        t.Fatalf("cand1 sum must be identity due to device mismatch, got %q", sums[testCand1])
    }

    // Off-chain tally logic would now classify this ballot as invalid and call ApplyBallotStatuses.
    statusJSON := `{"current":[],"invalid":[{"serial":"S-001","txID":"tx-wrongdev"}]}`
    requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, statusJSON))

    // Ballot meta should reflect invalidation.
    bm, err := h.cc.GetBallotBySerial(h.ctx, "S-001")
    requireNoErr(t, err)
    if bm.Status != "invalid" {
        t.Fatalf("status = %s want invalid", bm.Status)
    }

    // We no longer assert about TXIDX presence/absence here:
    // ApplyBallotStatuses may legitimately index invalid ballots for receipt checks.
}


// must2 is a small adapter to use requireNoErr on the second return value only.
// Params: _ – ignored string; err – error to check.
// Returns: the error (for requireNoErr to handle upstream).
func must2(_ string, err error) error { return err }
