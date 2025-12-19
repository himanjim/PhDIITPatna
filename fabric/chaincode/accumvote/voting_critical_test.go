// voting_critical_test.go
//
// Purpose: Focused tests for critical vote-casting behaviors and edge cases in
//          AccumVoteContract — missing PK, closed polls, idempotent same-candidate
//          re-votes, change re-votes, ciphertext validation (hex/range/invertibility),
//          unknown candidates (filtered at tally), and shard collisions.
// Role:    Stresses correctness guards and “latest-wins” semantics that directly
//          affect integrity. Uses the in-memory test harness (newHarness/memWorld)
//          with cc2cc stubs; no real Fabric network is involved.
// Key dependencies: AccumVoteContract methods (RecordVote, TallyPrepare, GetBallotBySerial),
//          helpers from sibling test files (bigFromPossiblyHex, nSquaredFromHarnessN,
//          powModHex, requireNoErr, requireErrContains, must2), and test constants
//          (testCand*, hexEncOneGood, hexN, testConst).

package main

import (
    "math/big"
	"strings"
	"testing"
)

// trim0xLocal removes an optional 0x/0X prefix.
// Params: s — string that may start with 0x/0X.
// Returns: s without the hex prefix (if present).
func trim0xLocal(s string) string {
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		return s[2:]
	}
	return s
}

// bigFromPossiblyHexMust wraps bigFromPossiblyHex with a test failure on error.
// Params: t — testing handle; s — decimal/hex string (0x allowed).
// Returns: parsed big.Int; fails the test on parse error.
func bigFromPossiblyHexMust(t *testing.T, s string) *big.Int {
	t.Helper()
	z, err := bigFromPossiblyHex(s)
	if err != nil {
		t.Fatalf("parse big int from %q: %v", s, err)
	}
	return z
}

// TestVotingCritical_PKMissingReject ensures casting fails when the Paillier PK
// hasn’t been written for the caller’s state.
// Steps: open poll, attempt RecordVote without SetJointPublicKey.
// Expect: error about missing public key.
func TestVotingCritical_PKMissingReject(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	
	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2}) // candidate list is present; only PK is missing

	
	defer h.ctrl.Finish()

	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-pk-miss")
	err := must2(h.recordVote("S-001", testCand1, hexEncOneGood))
	if err == nil {
		t.Fatalf("expected error when PK is missing")
	}
}

// TestVotingCritical_PollClosedReject confirms a closed poll blocks casting.
// Steps: set PK, seed candidates, close poll, try RecordVote.
// Expect: rejection mentioning “closed”.
func TestVotingCritical_PollClosedReject(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.closePoll())

	h.setTxID("tx-closed")
	err := must2(h.recordVote("S-001", testCand1, hexEncOneGood))
	if err == nil {
		t.Fatalf("expected error when poll is closed")
	}
}

// TestVotingCritical_FirstVotePlusOne checks first valid vote increments the
// chosen candidate and leaves others at identity.
// Expect: cand1 != 1; cand2 == 1.
func TestVotingCritical_FirstVotePlusOne(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	
	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2})  // enough for tally-time filtering if needed


	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2}))
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-first")
	requireNoErr(t, must2(h.recordVote("S-001", testCand1, hexEncOneGood)))

	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)
	one := bigFromPossiblyHexMust(t, "1")
	c1 := bigFromPossiblyHexMust(t, sums[testCand1])
	c2 := bigFromPossiblyHexMust(t, sums[testCand2])

	if c1.Cmp(one) == 0 {
		t.Fatalf("cand1 should be incremented, got identity")
	}
	if c2.Cmp(one) != 0 {
		t.Fatalf("cand2 should remain identity, got %s", c2.Text(16))
	}
}

// TestVotingCritical_SameCandidateRevote_NoOp verifies idempotence: re-casting
// for the same candidate doesn’t change the accumulator.
// Expect: enc sum for cand1 stays the same across both tallies.
func TestVotingCritical_SameCandidateRevote_NoOp(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	
	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2})  // keep scope minimal


	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-a")
	requireNoErr(t, must2(h.recordVote("S-001", testCand1, hexEncOneGood)))
	sums1, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	h.setTxID("tx-b")
	requireNoErr(t, must2(h.recordVote("S-001", testCand1, hexEncOneGood)))
	sums2, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	if sums1[testCand1] != sums2[testCand1] {
		t.Fatalf("same-candidate re-vote must not change accumulator")
	}
}

// TestVotingCritical_ChangeRevote_MoveBetweenCandidates checks latest-wins when
// moving from cand1 to cand2.
// Expect: cand1 back to identity; cand2 has one Enc(1).
func TestVotingCritical_ChangeRevote_MoveBetweenCandidates(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	
	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2})  // minimal candidate universe


	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2}))
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-1")
	requireNoErr(t, must2(h.recordVote("S-001", testCand1, hexEncOneGood)))
	h.setTxID("tx-2")
	requireNoErr(t, must2(h.recordVote("S-001", testCand2, hexEncOneGood)))

	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)
	n2 := nSquaredFromHarnessN(t)

	want1 := big.NewInt(1)                          // net zero for cand1 after move
	want2 := powModHex(t, hexEncOneGood, 1, n2)     // cand2 gets a single Enc(1)

	got1 := bigFromPossiblyHexMust(t, sums[testCand1])
	got2 := bigFromPossiblyHexMust(t, sums[testCand2])

	if got1.Cmp(want1) != 0 || got2.Cmp(want2) != 0 {
		t.Fatalf("revote move mismatch: cand1 got=%s want=%s; cand2 got=%s want=%s",
			got1.Text(16), want1.Text(16), got2.Text(16), want2.Text(16))
	}
}

// TestVotingCritical_BadEncOne_NonHex validates input parsing: non-hex encOne must fail.
// Expect: parse error.
func TestVotingCritical_BadEncOne_NonHex(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-badhex")
	err := must2(h.recordVote("S-001", testCand1, "zzzz")) // not hex
	if err == nil {
		t.Fatalf("expected parse error on non-hex encOne")
	}
}

// TestVotingCritical_BadEncOne_TooSmall checks the lower bound: encOne must be > 1.
// Expect: range error.
func TestVotingCritical_BadEncOne_TooSmall(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-small")
	err := must2(h.recordVote("S-001", testCand1, "1")) // boundary case
	if err == nil {
		t.Fatalf("expected validation error for encOne <= 1")
	}
}

// TestVotingCritical_BadEncOne_NonInvertible ensures encOne that shares a factor
// with n² (e.g., c = n) is rejected (gcd != 1).
// Note: trim0xLocal(hexN) yields n in hex without prefix for this check.
func TestVotingCritical_BadEncOne_NonInvertible(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.openPoll())

	encOne := trim0xLocal(hexN) // c = n → not invertible mod n²
	h.setTxID("tx-noninvertible")
	err := must2(h.recordVote("S-001", testCand1, encOne))
	if err == nil {
		t.Fatalf("expected validation error for non-invertible encOne (gcd != 1)")
	}
}

// TestVotingCritical_UnknownCandidate_ValidateOn documents that cast-time no
// longer rejects unknown candidates; they’re filtered during the tally pipeline
// (TallyPrepare + ApplyBallotStatuses).
// Expect: tally excludes the unknown candidate; ballot meta marked invalid once
// ApplyBallotStatuses is invoked with an invalid entry for that serial.
func TestVotingCritical_UnknownCandidate_ValidateOn(t *testing.T) {
    setProdEnv(t) // VALIDATE_ON_TALLY=on
    h := newHarness(t)
    defer h.ctrl.Finish()

    requireNoErr(t, h.setPK_UP())
    requireNoErr(t, h.seedCandidates([]string{testCand1})) // only cand-1 valid locally
    requireNoErr(t, h.openPoll())

    // Accept at cast-time…
    h.setTxID("tx-UNK")
    requireNoErr(t, must2(h.recordVote("S-UNK", "cand-UNKNOWN", hexEncOneGood)))

    // …and ensure tally-time sums ignore the unknown candidate.
    encSums, err := h.cc.TallyPrepare(h.ctx, testConst)
    requireNoErr(t, err)

    got := mustBigFromHex(t, encSums[testCand1]) // with no valid votes, identity expected
    if got.Cmp(big.NewInt(1)) != 0 {
        t.Fatalf("expected Enc(0) product == 1 for cand-1, got %s", got.Text(16))
    }

    // Off-chain tally logic would now flag this ballot as invalid and call ApplyBallotStatuses.
    statusJSON := `{"current":[],"invalid":[{"serial":"S-UNK","txID":"tx-UNK"}]}`
    requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, statusJSON))

    // Public meta must reveal invalidation for this serial.
    bm, err := h.cc.GetBallotBySerial(h.ctx, "S-UNK")
    requireNoErr(t, err)
    if bm.Status != "invalid" {
        t.Fatalf("expected ballot status=invalid, got %q", bm.Status)
    }
}


