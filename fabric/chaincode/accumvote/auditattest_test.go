// auditattest_test.go tests voter-requested openings and the query-log
// attestation (auditattest.go).
package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"
)

// TestRequestedOpening_RecordedAndChecked: a voter-requested opening is published
// with requested = "voter", a coin opening is not marked, and a requested
// opening without an indicator query, or for a recorded ballot, is reported.
func TestRequestedOpening_RecordedAndChecked(t *testing.T) {
	h, key := setupPackedPoll(t)
	defer h.ctrl.Finish()

	coinHC := sha256HexOf(key.packedBallotHex(t, 1, 8, 2001))
	voterHC := sha256HexOf(key.packedBallotHex(t, 2, 8, 2003))
	requireNoErr(t, h.cc.RecordAuditOpening(h.ctx, testConst, coinHC, 1, "7d1", testBoothID, testDeviceID))
	requireNoErr(t, h.cc.RecordRequestedAuditOpening(h.ctx, testConst, voterHC, 2, "7d3", testBoothID, testDeviceID))
	requireErrContains(t, h.cc.RecordRequestedAuditOpening(h.ctx, testConst, voterHC, 2, "7d3", testBoothID, testDeviceID), "already recorded")

	out, err := h.cc.ExportAuditOpenings(h.ctx)
	requireNoErr(t, err)
	var rows []AuditLogEntry
	requireNoErr(t, json.Unmarshal([]byte(out), &rows))
	got := map[string]string{}
	for _, r := range rows {
		got[r.HC] = r.Requested
	}
	if len(rows) != 2 || got[coinHC] != "" || got[voterHC] != RequestedByVoter {
		t.Fatalf("unexpected openings %+v", rows)
	}

	unqueried, alsoCast := CheckRequestedOpenings([]string{voterHC}, []string{"aa"}, []string{coinHC, voterHC})
	if len(unqueried) != 0 || len(alsoCast) != 0 {
		t.Fatalf("a justified requested opening was reported: %v %v", unqueried, alsoCast)
	}
	unqueried, alsoCast = CheckRequestedOpenings([]string{voterHC}, []string{voterHC}, []string{coinHC})
	if len(unqueried) != 1 || len(alsoCast) != 1 {
		t.Fatalf("an unjustified requested opening was not reported: %v %v", unqueried, alsoCast)
	}
}

// TestQueryLogAttestation_SignAndVerify: an attestation verifies against the
// published log; a hidden query, a substituted query, another key or an altered
// signature does not.
func TestQueryLogAttestation_SignAndVerify(t *testing.T) {
	seed := sha256.Sum256([]byte("test-audit-hsm"))
	priv := ed25519.NewKeyFromSeed(seed[:])
	pub := hex.EncodeToString(priv.Public().(ed25519.PublicKey))
	log := []string{"cc", "AA", "bb"}

	digest := QueryLogDigest(log)
	a := QueryLogAttestation{ConstituencyID: testConst, Queries: len(log), QueryLogDigest: digest}
	a.Signature = hex.EncodeToString(ed25519.Sign(priv, QueryLogAttestationMessage(a.ConstituencyID, a.Queries, a.QueryLogDigest)))
	requireNoErr(t, VerifyQueryLogAttestation(pub, a, log))
	requireNoErr(t, VerifyQueryLogAttestation(pub, a, []string{"bb", "aa", "CC"}))

	requireErrContains(t, VerifyQueryLogAttestation(pub, a, []string{"aa", "bb"}), "has 2 entries")
	requireErrContains(t, VerifyQueryLogAttestation(pub, a, []string{"aa", "bb", "dd"}), "digest does not match")
	otherSeed := sha256.Sum256([]byte("another-key"))
	otherPub := hex.EncodeToString(ed25519.NewKeyFromSeed(otherSeed[:]).Public().(ed25519.PublicKey))
	requireErrContains(t, VerifyQueryLogAttestation(otherPub, a, log), "does not verify")
	bad := a
	bad.Queries = 4
	requireErrContains(t, VerifyQueryLogAttestation(pub, bad, append(log, "dd")), "does not verify")
}
