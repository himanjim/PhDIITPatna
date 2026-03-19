// voting_test.go covers the cast path of the contract from the perspective of the
// in-memory harness.
//
// The tests focus on what RecordVote contributes to later tally outcomes: rejection
// when required preconditions are missing, correct handling of first votes and
// re-votes, rejection of malformed ciphertext inputs, and a coarse guard on the
// number of state operations used by the re-vote path.

package main

import (
	"testing"
	"strings"
)

const testSerial = "S-001"

// normHex reduces a ciphertext or accumulator string to a canonical form suitable
// for equality assertions in tests. The helper exists only to make the assertions
// stable across minor formatting differences such as optional 0x prefixes or odd
// hex lengths.
func normHex(s string) string {
	return canonHex(s)
}

// canonHex canonicalises a hex string by removing a leading 0x prefix, converting
// to lowercase, and left-padding odd-length values. The function does not validate
// semantic correctness; it only normalises formatting for string comparison.
func canonHex(s string) string {
    s = strings.TrimPrefix(strings.ToLower(s), "0x")
    if len(s)%2 == 1 {
        s = "0" + s
    }
    return s
}

// AsIdentityIfBlank maps contract identity renderings to "1" for easier assertions.
// Params: s — value returned from chaincode (may be "", "0", "0x0").
// Returns: "1" when s is an identity-like string; otherwise returns s unchanged.
func asIdentityIfBlank(s string) string {
    s = strings.TrimSpace(s)
    if s == "" || s == "0" || s == "0x0" || s == "0x" {
        return "1"
    }
    return s
}

// TestVoting_PKMissingReject checks that vote submission fails when no Paillier
// public key has been installed for the resolved state. The candidate list and poll
// state are prepared so that the failure can be attributed specifically to missing
// key material rather than to an earlier guard.
func TestVoting_PKMissingReject(t *testing.T) {
	// SetDefaultEnv(t)
	setProdEnv(t)

	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2}
	h.stubPreloadCandidatesOnly(cands) // Only candidate list needed; voter table irrelevant here
	// RequireNoErr(t, h.cc.RefreshCandidateListFromPreload(h.ctx, testConst))
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2, testCand3, testCand4})) // Seed more than used; harmless

	requireNoErr(t, h.openPoll())
	_, err := h.recordVote(testSerial, testCand1, hexEncOneGood)
	requireErrContains(t, err, "public key") // Contract error text: "public key for state ... not set"
}

// TestVoting_PollClosedReject checks that RecordVote refuses an otherwise valid
// submission once the poll has been marked closed. The public key is installed
// first so that the rejection path isolates the poll-state guard.
func TestVoting_PollClosedReject(t *testing.T) {
	// SetDefaultEnv(t)
	setProdEnv(t)

	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2}
	h.stubPreloadCandidatesOnly(cands)
	// RequireNoErr(t, h.cc.RefreshCandidateListFromPreload(h.ctx, testConst))
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2, testCand3, testCand4}))

	requireNoErr(t, h.setPK_UP())  // PK must exist for a clean "closed" rejection
	requireNoErr(t, h.closePoll()) // Explicit close to trigger guard
	_, err := h.recordVote(testSerial, testCand1, hexEncOneGood)
	requireErrContains(t, err, "closed")
}

// TestVoting_FirstVotePlusOne checks the baseline tally effect of one accepted
// ballot. After one cast for candidate 1, the encrypted sum for candidate 1 should
// equal the supplied Enc(1) value, while the accumulator for candidate 2 should
// remain at the Paillier identity element.
func TestVoting_FirstVotePlusOne(t *testing.T) {
	setProdEnv(t)

	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2}
	h.stubPreloadCandidatesOnly(cands)
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2, testCand3, testCand4}))

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.openPoll())

	_, err := h.recordVote(testSerial, testCand1, hexEncOneGood)
	requireNoErr(t, err)

	// After a single cast: cand1 = Enc(1), cand2 = identity.
	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	if err != nil { t.Fatalf("tally: %v", err) }

	got1 := normHex(sums[testCand1])
	got2 := normHex(sums[testCand2])
	if got1 != normHex(hexEncOneGood) || got2 != normHex("1") {
		t.Fatalf("unexpected tally: cand1 got=%s want=%s; cand2 got=%s want=%s",
			got1, normHex(hexEncOneGood), got2, normHex("1"))
	}
}

// TestVoting_SameCandidateRevote_NoOp checks that a re-vote for the same candidate
// does not change the encrypted tally outcome. Distinct transaction identifiers are
// used to model two separate casts, but because the selected candidate is unchanged,
– the net encrypted contribution should remain exactly one Enc(1) for that candidate.
func TestVoting_SameCandidateRevote_NoOp(t *testing.T) {
	setProdEnv(t)

	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2}
	h.stubPreloadCandidatesOnly(cands)
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2, testCand3, testCand4}))

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.openPoll())

	// Different TxIDs to mirror realistic distinct casts; still the same candidate.
	h.setTxID("tx-1")
	_, err := h.recordVote(testSerial, testCand1, hexEncOneGood)
	requireNoErr(t, err)

	h.setTxID("tx-2")
	_, err = h.recordVote(testSerial, testCand1, hexEncOneGood)
	requireNoErr(t, err)

	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	if err != nil { t.Fatalf("tally: %v", err) }

	got1 := normHex(sums[testCand1])
	got2 := normHex(sums[testCand2])
	want1 := normHex(hexEncOneGood)
	want2 := normHex("1")

	if got1 != want1 || got2 != want2 {
		t.Fatalf("unexpected tally after same-candidate re-vote: cand1=%s want=%s; cand2=%s want=%s",
			got1, want1, got2, want2)
	}
}

// TestVoting_ChangeRevote_MoveBetweenCandidates checks the latest-vote-wins rule
// when a voter changes preference. A first vote is cast for candidate 1 and a later
// vote for candidate 2 under the same serial number. The final encrypted tally must
// therefore return candidate 1 to the identity element and assign one Enc(1) to
// candidate 2.
func TestVoting_ChangeRevote_MoveBetweenCandidates(t *testing.T) {
	setProdEnv(t)

	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2}
	h.stubPreloadCandidatesOnly(cands)
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2, testCand3, testCand4}))

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.openPoll())

	// Vote cand1 then re-vote to cand2 → cand1 back to identity, cand2 gets Enc(1).
	h.setTxID("tx-1")
	_, err := h.recordVote(testSerial, testCand1, hexEncOneGood)
	requireNoErr(t, err)
	h.setTxID("tx-2")
	_, err = h.recordVote(testSerial, testCand2, hexEncOneGood)
	requireNoErr(t, err)

	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	if err != nil { t.Fatalf("tally: %v", err) }

	got1 := normHex(sums[testCand1])
	got2 := normHex(sums[testCand2])
	want1 := normHex("1")
	want2 := normHex(hexEncOneGood)

	if got1 != want1 || got2 != want2 {
		t.Fatalf("unexpected tally after change re-vote: cand1=%s want=%s, cand2=%s want=%s",
			got1, want1, got2, want2)
	}
}

// TestVoting_BadEncOne_Reject_NonHex checks that RecordVote rejects ciphertext input
// that cannot be parsed as a valid integer. The test ensures that malformed input is
// rejected before it can affect private data or public ballot metadata.
func TestVoting_BadEncOne_Reject_NonHex(t *testing.T) {
	// SetDefaultEnv(t)
	setProdEnv(t)

	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2}
	h.stubPreloadCandidatesOnly(cands)
	// RequireNoErr(t, h.cc.RefreshCandidateListFromPreload(h.ctx, testConst))
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2, testCand3, testCand4}))

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.openPoll())
	_, err := h.recordVote(testSerial, testCand1, "zzzz") // Intentionally invalid hex
	requireErrContains(t, err, "hex")
}

// TestVoting_BadEncOne_Reject_OutOfRange checks that RecordVote rejects ciphertext
// values outside the admissible Paillier range. The specific input is deliberately
// chosen at the lower boundary so that the failure comes from range validation
// rather than from parsing.
func TestVoting_BadEncOne_Reject_OutOfRange(t *testing.T) {
	// SetDefaultEnv(t)
	setProdEnv(t)

	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2}
	h.stubPreloadCandidatesOnly(cands)
	// RequireNoErr(t, h.cc.RefreshCandidateListFromPreload(h.ctx, testConst))
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2, testCand3, testCand4}))

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.openPoll())
	_, err := h.recordVote(testSerial, testCand1, "0x01") // Boundary: not strictly >1
	requireErrContains(t, err, "range")
}

// TestVoting_StateOpsBudget_ChangeRevote places a coarse budget on ledger touches
// during the change-of-choice re-vote path. The test is not a micro-benchmark. Its
// role is to detect accidental growth in state or private-data operations that would
// make the hot path materially heavier than intended.
func TestVoting_StateOpsBudget_ChangeRevote(t *testing.T) {
	// SetDefaultEnv(t)
	setProdEnv(t)

	h := newHarness(t)
	defer h.ctrl.Finish()

	cands := []string{testCand1, testCand2}
	h.stubPreloadCandidatesOnly(cands)
	// RequireNoErr(t, h.cc.RefreshCandidateListFromPreload(h.ctx, testConst))
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2, testCand3, testCand4}))

	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.openPoll())

	_, err := h.recordVote(testSerial, testCand1, hexEncOneGood)
	requireNoErr(t, err)

	// Reset the in-mem counters to measure only the re-vote path.
	h.mem.opsCounts = struct {
		getState, putState int
		getPDC, putPDC     int
		setEvent           int
	}{}

	_, err = h.recordVote(testSerial, testCand2, hexEncOneGood)
	requireNoErr(t, err)

	// These ceilings are intentionally forgiving — goal is to catch accidental
	// Extra reads/writes, not micro-optimize.
	if h.mem.opsCounts.getPDC > 1 || h.mem.opsCounts.putPDC > 1 {
		t.Fatalf("PDC ops too high: get=%d put=%d", h.mem.opsCounts.getPDC, h.mem.opsCounts.putPDC)
	}
	if h.mem.opsCounts.getState > 5 || h.mem.opsCounts.putState > 5 {
		t.Fatalf("WS ops too high: get=%d put=%d", h.mem.opsCounts.getState, h.mem.opsCounts.putState)
	}
	// Events/meta are config-driven; no assertion here.
}
