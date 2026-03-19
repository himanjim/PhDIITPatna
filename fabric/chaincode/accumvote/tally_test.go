// tally_test.go exercises the encrypted read path of the contract.
//
// The file checks that tally preparation reproduces the expected Paillier products,
// that latest-vote-wins is reflected in the encrypted sums, that booth/device
// validation can exclude otherwise well-formed ballots, and that GetEncSums stays
// behaviourally aligned with TallyPrepare.
package main

import (
	"encoding/hex"
	"math/big"
	"strings"
	"testing"
)

// ---------- small helpers (local to this file) ----------

// bigFromPossiblyHex parses a test integer from either decimal notation or a loose
// hexadecimal notation with or without a prefix. The helper exists because the test
// suite compares values coming from several formatting conventions but wants one
// canonical big.Int representation for arithmetic assertions.
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

// MustBigFromHex is a test helper that fatals on parse error.
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

// powModHex raises a ciphertext to an integer power modulo n² so that the tests can
// state expected Paillier tally behaviour directly. In these tests, exponentiation
// corresponds to repeated multiplication of Enc(1), which represents a plaintext
// vote count under homomorphic accumulation.
func powModHex(t *testing.T, hexC string, k int, n2 *big.Int) *big.Int {
	t.Helper()
	base := mustBigFromHex(t, hexC)
	exp := big.NewInt(int64(k))
	return new(big.Int).Exp(base, exp, n2)
}

// nSquaredFromHarnessN derives the Paillier modulus square used by the harness. The
// helper keeps the arithmetic expectations in this file tied to the same configured
// public key material used by the contract under test.
func nSquaredFromHarnessN(t *testing.T) *big.Int {
	t.Helper()
	n := mustBigFromHex(t, hexN)
	return new(big.Int).Mul(n, n)
}

// ---------- Tests ----------

// TestTally_ZeroVote_Identity checks the empty-election baseline. With no accepted
// vote records contributing to the constituency, the encrypted sum for each
// candidate must remain at the Paillier identity element, which corresponds to an
// encrypted zero count.
func TestTally_ZeroVote_Identity(t *testing.T) {
	// SetDefaultEnv(t)
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

// TestTally_SimpleDistribution checks that tally preparation multiplies one Enc(1)
// contribution per valid latest ballot. Two ballots are cast for candidate 1 and
// one for candidate 2, so the expected encrypted sums are Enc(1)^2 for candidate 1
// and Enc(1)^1 for candidate 2 modulo n².
func TestTally_SimpleDistribution(t *testing.T) {
	// SetDefaultEnv(t)
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

	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2}) // Hot-path list small for the test.

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

// TestTally_RevoteChange checks that the read path honours the latest-vote-wins
// rule. One serial number first selects candidate 1 and then candidate 2. The final
// encrypted tally must therefore treat candidate 1 as if no current vote remains and
// must assign exactly one encrypted contribution to candidate 2.
func TestTally_RevoteChange(t *testing.T) {
	// SetDefaultEnv(t)
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

// TestGetEncSums_MirrorsTally checks API parity between GetEncSums and
// TallyPrepare. The two methods are expected to expose the same encrypted sums for
// the same ledger state, even though they are distinct query entry points.
func TestGetEncSums_MirrorsTally(t *testing.T) {
	// SetDefaultEnv(t)
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

// Test_Tally_BoothDeviceMismatch_Excluded checks that a ballot can be structurally
// accepted at cast time and still be excluded later when booth or device validation
// fails during tally preparation. After tally exclusion, the ballot is explicitly
// marked invalid through ApplyBallotStatuses so that the public metadata matches the
// tally decision.
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
    if sums[testCand1] != "1" { // Identity → excluded
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


// must2 discards the first return value of a helper that returns `(string, error)`
// and leaves only the error for assertion. It is used to keep the call sites short
// when a test cares about acceptance or rejection but not about the returned JSON.
func must2(_ string, err error) error { return err }
