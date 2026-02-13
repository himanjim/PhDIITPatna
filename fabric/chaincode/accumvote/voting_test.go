// Voting_test.go
//
// Purpose: Tests for the vote-casting path of AccumVoteContract: preconditions,
// First-vote behavior, re-vote semantics, input validation, and a light
// State-ops budget sanity check.
// Role: Exercises RecordVote + TallyPrepare via the in-memory harness (no real Fabric),
// Using cc2cc stubs for candidate list. Focus is on correctness signals
// (accept/reject, accumulator effects) rather than throughput.
// Key dependencies: newHarness/memWorld test harness, AccumVoteContract, helper
// Functions like requireNoErr/requireErrContains, and constants such as
// TestCand1..4, hexEncOneGood.

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

// CanonHex normalizes hex for comparisons: drop 0x, lowercase, and left-pad to even length.
// Params: s — input string like "0x1", "A", "0a".
// Returns: normalized hex string suitable for direct string compare.
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

// TestVoting_FirstVotePlusOne verifies: Voting First Vote Plus One.
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

// TestVoting_ChangeRevote_MoveBetweenCandidates verifies: Voting Change Revote Move Between Candidates.
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

// TestVoting_StateOpsBudget_ChangeRevote verifies: Voting State Ops Budget Change Revote.
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
