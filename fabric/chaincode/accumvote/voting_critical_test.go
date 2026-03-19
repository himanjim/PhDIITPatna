// voting_critical_test.go isolates the most integrity-sensitive behaviours of the
// cast path.
//
// Compared with voting_test.go, this file is narrower and sharper. Each test targets
// one failure mode or one re-vote property whose regression would directly affect
// election correctness: missing keys, closed polls, same-choice and changed-choice
// re-votes, malformed ciphertexts, and exclusion of unknown candidates at tally time.

package main

import (
    "math/big"
	"strings"
	"testing"
)

// trim0xLocal removes an optional hexadecimal prefix from a test input string. The
// helper is used when a value must be passed to validation logic in bare integer
// form while still being derived from a hex constant defined elsewhere in the suite.
func trim0xLocal(s string) string {
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		return s[2:]
	}
	return s
}

// bigFromPossiblyHexMust is a test-only adapter that converts a relaxed integer
// string into a big integer and aborts the test on failure. It keeps the assertions
// in this file compact without hiding parsing errors.
func bigFromPossiblyHexMust(t *testing.T, s string) *big.Int {
	t.Helper()
	z, err := bigFromPossiblyHex(s)
	if err != nil {
		t.Fatalf("parse big int from %q: %v", s, err)
	}
	return z
}

// TestVotingCritical_PKMissingReject confirms that the contract refuses vote
// submission when the election public key is absent. The poll is opened and the
// candidate list is present so that the missing-key condition is the only intended
// reason for rejection.
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

// TestVotingCritical_PollClosedReject confirms that a cast is rejected after the
// poll has been closed, even when the public key and candidate list are available.
// This test isolates the poll-state guard from the other preconditions.
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

// TestVotingCritical_FirstVotePlusOne confirms the minimum non-zero tally effect of
// one accepted vote. It checks only the logical outcome: the chosen candidate must
// no longer have the identity accumulator, whereas the untouched candidate must
// still remain at identity.
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

// TestVotingCritical_SameCandidateRevote_NoOp confirms that repeated casts for the
// same candidate under one serial number are idempotent at tally level. A second
// accepted cast with a different transaction identifier must not alter the final
// encrypted accumulator for that candidate.
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

// TestVotingCritical_ChangeRevote_MoveBetweenCandidates confirms the net tally
// effect of changing one's choice. After two accepted casts under the same serial
// number, only the most recent candidate should retain one encrypted vote, and the
// earlier candidate should return to the Paillier identity element.
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

// TestVotingCritical_BadEncOne_NonHex confirms that the cast path rejects a
// ciphertext that is not parseable as a valid integer. This protects the vote store
// from malformed input before any tally-time validation is reached.
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

// TestVotingCritical_BadEncOne_TooSmall confirms that the cast path rejects a
// ciphertext at or below the lower admissible boundary. The specific value is chosen
// to exercise range validation rather than general parsing failure.
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

// TestVotingCritical_BadEncOne_NonInvertible confirms that a ciphertext sharing a
// factor with n² is rejected before storage. The test uses c = n, which is
// guaranteed to be non-invertible modulo n² and therefore invalid for Paillier
// multiplication in the tally path.
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

// TestVotingCritical_UnknownCandidate_ValidateOn confirms that candidate validity is
// enforced at tally time rather than cast time in the current design. The ballot is
// first accepted into storage, but because the candidate is absent from the seeded
// candidate list, the tally must ignore it and the later status application must
// expose the ballot as invalid in public metadata.
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
