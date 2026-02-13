// Voting_critical_test.go
//
// Purpose: Focused tests for critical vote-casting behaviors and edge cases in
// AccumVoteContract — missing PK, closed polls, idempotent same-candidate
// Re-votes, change re-votes, ciphertext validation (hex/range/invertibility),
// Unknown candidates (filtered at tally), and shard collisions.
// Role: Stresses correctness guards and “latest-wins” semantics that directly
// Affect integrity. Uses the in-memory test harness (newHarness/memWorld)
// With cc2cc stubs; no real Fabric network is involved.
// Key dependencies: AccumVoteContract methods (RecordVote, TallyPrepare, GetBallotBySerial),
// Helpers from sibling test files (bigFromPossiblyHex, nSquaredFromHarnessN,
// PowModHex, requireNoErr, requireErrContains, must2), and test constants
// (testCand*, hexEncOneGood, hexN, testConst).

package main

import (
    "math/big"
	"strings"
	"testing"
)

// Trim0xLocal removes an optional 0x/0X prefix.
// Params: s — string that may start with 0x/0X.
// Returns: s without the hex prefix (if present).
func trim0xLocal(s string) string {
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		return s[2:]
	}
	return s
}

// BigFromPossiblyHexMust wraps bigFromPossiblyHex with a test failure on error.
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

// TestVotingCritical_PKMissingReject verifies: Voting Critical P K Missing Reject.
func TestVotingCritical_PKMissingReject(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	
	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2}) // Candidate list is present; only PK is missing

	
	defer h.ctrl.Finish()

	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-pk-miss")
	err := must2(h.recordVote("S-001", testCand1, hexEncOneGood))
	if err == nil {
		t.Fatalf("expected error when PK is missing")
	}
}

// TestVotingCritical_PollClosedReject verifies: Voting Critical Poll Closed Reject.
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

// TestVotingCritical_FirstVotePlusOne verifies: Voting Critical First Vote Plus One.
func TestVotingCritical_FirstVotePlusOne(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	
	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2})  // Enough for tally-time filtering if needed


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

// TestVotingCritical_SameCandidateRevote_NoOp verifies: Voting Critical Same Candidate Revote No Op.
func TestVotingCritical_SameCandidateRevote_NoOp(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	
	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2})  // Keep scope minimal


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

// TestVotingCritical_ChangeRevote_MoveBetweenCandidates verifies: Voting Critical Change Revote Move Between Candidates.
func TestVotingCritical_ChangeRevote_MoveBetweenCandidates(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	
	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2})  // Minimal candidate universe


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

	want1 := big.NewInt(1)                          // Net zero for cand1 after move
	want2 := powModHex(t, hexEncOneGood, 1, n2)     // Cand2 gets a single Enc(1)

	got1 := bigFromPossiblyHexMust(t, sums[testCand1])
	got2 := bigFromPossiblyHexMust(t, sums[testCand2])

	if got1.Cmp(want1) != 0 || got2.Cmp(want2) != 0 {
		t.Fatalf("revote move mismatch: cand1 got=%s want=%s; cand2 got=%s want=%s",
			got1.Text(16), want1.Text(16), got2.Text(16), want2.Text(16))
	}
}

// TestVotingCritical_BadEncOne_NonHex verifies: Voting Critical Bad Enc One Non Hex.
func TestVotingCritical_BadEncOne_NonHex(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-badhex")
	err := must2(h.recordVote("S-001", testCand1, "zzzz")) // Not hex
	if err == nil {
		t.Fatalf("expected parse error on non-hex encOne")
	}
}

// TestVotingCritical_BadEncOne_TooSmall verifies: Voting Critical Bad Enc One Too Small.
func TestVotingCritical_BadEncOne_TooSmall(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-small")
	err := must2(h.recordVote("S-001", testCand1, "1")) // Boundary case
	if err == nil {
		t.Fatalf("expected validation error for encOne <= 1")
	}
}

// TestVotingCritical_BadEncOne_NonInvertible verifies: Voting Critical Bad Enc One Non Invertible.
func TestVotingCritical_BadEncOne_NonInvertible(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()
	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1}))
	requireNoErr(t, h.openPoll())

	encOne := trim0xLocal(hexN) // C = n → not invertible mod n²
	h.setTxID("tx-noninvertible")
	err := must2(h.recordVote("S-001", testCand1, encOne))
	if err == nil {
		t.Fatalf("expected validation error for non-invertible encOne (gcd != 1)")
	}
}

// TestVotingCritical_UnknownCandidate_ValidateOn verifies: Voting Critical Unknown Candidate Validate On.
func TestVotingCritical_UnknownCandidate_ValidateOn(t *testing.T) {
    setProdEnv(t) // VALIDATE_ON_TALLY=on
    h := newHarness(t)
    defer h.ctrl.Finish()

    requireNoErr(t, h.setPK_UP())
    requireNoErr(t, h.seedCandidates([]string{testCand1})) // Only cand-1 valid locally
    requireNoErr(t, h.openPoll())

    // Accept at cast-time…
    h.setTxID("tx-UNK")
    requireNoErr(t, must2(h.recordVote("S-UNK", "cand-UNKNOWN", hexEncOneGood)))

    // …and ensure tally-time sums ignore the unknown candidate.
    encSums, err := h.cc.TallyPrepare(h.ctx, testConst)
    requireNoErr(t, err)

    got := mustBigFromHex(t, encSums[testCand1]) // With no valid votes, identity expected
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
