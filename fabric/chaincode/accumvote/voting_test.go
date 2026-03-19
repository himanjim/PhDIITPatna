/*
voting_test.go exercises the cast path with the in-memory harness under
production-like parameter settings. The tests are deliberately narrow: they
check that RecordVote rejects missing prerequisites, preserves latest-vote-wins
behaviour, rejects malformed ciphertexts, and does not expand the read or write
footprint unexpectedly on a re-vote. The objective is contract-boundary
correctness, not end-to-end Fabric benchmarking.
*/

package main

import (
	"testing"
	"strings"
)

const testSerial = "S-001"

// NormHex normalizes a hex string for equality checks.
// Params: s — input hex string (may include 0x).
// Returns: canonical lowercase hex with even length (via canonHex).
func normHex(s string) string {
	return canonHex(s)
}

// canonHex normalises hexadecimal strings for assertions so that harmless
// differences in prefix, case, or odd-length padding do not cause false test
// failures. It is used only at comparison time; it does not alter contract
// output.
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

// TestVoting_PKMissingReject verifies: Voting P K Missing Reject.
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

// TestVoting_PollClosedReject verifies: Voting Poll Closed Reject.
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

// Establishes the baseline tally behaviour for a single valid cast. After one
// vote for cand1, the test expects cand1 to hold Enc(1) and every untouched
// candidate to remain at the Paillier identity element.
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

// TestVoting_SameCandidateRevote_NoOp verifies: Voting Same Candidate Revote No Op.
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

// Demonstrates the latest-vote-wins rule on the accumulator path. The same
// serial first votes for cand1 and then re-votes for cand2; the expected result
// is that cand1 returns to the identity element and cand2 carries the single
// effective encrypted contribution.
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

// TestVoting_BadEncOne_Reject_NonHex verifies: Voting Bad Enc One Reject Non Hex.
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

// TestVoting_BadEncOne_Reject_OutOfRange verifies: Voting Bad Enc One Reject Out Of Range.
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

// Guards the short write path against accidental growth in ledger operations.
// The test resets the in-memory counters after the first cast and then measures
// only the re-vote path, using generous ceilings so that structural regressions
// are caught without turning the test into a micro-benchmark.
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
