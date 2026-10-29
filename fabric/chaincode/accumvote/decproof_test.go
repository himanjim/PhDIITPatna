// decproof_test.go tests the single-key decryption proof (decproof.go) and its
// check in PublishResults.
package main

import (
	"encoding/json"
	"fmt"
	"math/big"
	"testing"
)

// dpAggregate encrypts the given option indices as packed ballots, multiplies
// them, and returns the aggregate, its decrypted total and the proof value rho.
func dpAggregate(t *testing.T, key *paillierTestKey, options []int) (c, m, rho *big.Int) {
	t.Helper()
	c = big.NewInt(1)
	for i, j := range options {
		ci := mustBigFromHex(t, key.packedBallotHex(t, j, 8, int64(1001+2*i)))
		c = mulMod(c, ci, key.n2)
	}
	m = key.decCountT(t, c.Text(16))
	rho, err := DecryptionRandomness(key.n, nil, key.lambda, c, m)
	requireNoErr(t, err)
	return c, m, rho
}

// TestDecryptionProof_ProvesTheTotal: (m, rho) verifies for the aggregate, and
// a different total, a different rho or a different aggregate does not.
func TestDecryptionProof_ProvesTheTotal(t *testing.T) {
	key := newPackedTestKey(t)
	c, m, rho := dpAggregate(t, key, []int{0, 1, 1, 3, 2, 1})
	requireNoErr(t, VerifyDecryption(key.n, nil, c, m, rho))
	requireNoErr(t, VerifyDecryption(key.n, new(big.Int).Add(key.n, big.NewInt(1)), c, m, rho))

	if VerifyDecryption(key.n, nil, c, new(big.Int).Add(m, big.NewInt(1)), rho) == nil {
		t.Fatalf("a wrong total verified")
	}
	if VerifyDecryption(key.n, nil, c, m, new(big.Int).Add(rho, big.NewInt(1))) == nil {
		t.Fatalf("a wrong rho verified")
	}
	other, _, _ := dpAggregate(t, key, []int{0, 1, 1, 3, 2, 2})
	if VerifyDecryption(key.n, nil, other, m, rho) == nil {
		t.Fatalf("the proof verified for a different aggregate")
	}
	counts, err := PackedDecode(m, 8, 4)
	requireNoErr(t, err)
	if fmt.Sprint(counts) != "[1 3 1 1]" {
		t.Fatalf("unexpected counts %v", counts)
	}
}

func dpResults(t *testing.T, c, m, rho *big.Int, counts []uint64) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"counts": counts,
		"decryptionProof": DecryptionProof{
			CTally: c.Text(16), Total: m.Text(16), Rho: rho.Text(16),
		},
	})
	requireNoErr(t, err)
	return string(b)
}

func dpClosedPoll(t *testing.T) (*testHarness, *paillierTestKey) {
	t.Helper()
	h, key := setupPackedPoll(t)
	requireNoErr(t, h.cc.SetParams(h.ctx, `{"PACKED_MODE":true,"PACKED_SLOT_BITS":8,"PACKED_SLOTS":4}`))
	requireNoErr(t, h.closePoll())
	return h, key
}

// TestPublishResults_ChecksDecryptionProof: PublishResults accepts results whose
// proof verifies and whose counts follow from the proved total, and refuses
// results with a wrong count or a wrong total.
func TestPublishResults_ChecksDecryptionProof(t *testing.T) {
	h, key := dpClosedPoll(t)
	defer h.ctrl.Finish()
	c, m, rho := dpAggregate(t, key, []int{0, 1, 1, 3, 2, 1})

	requireNoErr(t, h.cc.PublishResults(h.ctx, testConst, "R1", dpResults(t, c, m, rho, []uint64{1, 3, 1, 1}), "B1"))
	anchor := fetchPublishedAnchor(t, h, testConst, "R1")
	if len(anchor.Results) == 0 {
		t.Fatalf("results were not anchored")
	}

	requireErrContains(t, h.cc.PublishResults(h.ctx, testConst, "R2", dpResults(t, c, m, rho, []uint64{2, 2, 1, 1}), "B2"), "proved total gives")
	requireErrContains(t, h.cc.PublishResults(h.ctx, testConst, "R3", dpResults(t, c, new(big.Int).Add(m, big.NewInt(1)), rho, []uint64{1, 3, 1, 1}), "B3"), "does not equal the aggregate")
}

// TestPublishResults_RequiresProofWhenConfigured: with REQUIRE_DECRYPTION_PROOF
// set, results without a proof are refused; without it, they are accepted as
// before.
func TestPublishResults_RequiresProofWhenConfigured(t *testing.T) {
	h, key := dpClosedPoll(t)
	defer h.ctrl.Finish()
	c, m, rho := dpAggregate(t, key, []int{2, 2, 0})

	requireNoErr(t, h.cc.PublishResults(h.ctx, testConst, "R0", `{"counts":[1,0,2,0]}`, "B0"))
	requireNoErr(t, h.cc.SetParams(h.ctx, `{"REQUIRE_DECRYPTION_PROOF":true}`))
	requireErrContains(t, h.cc.PublishResults(h.ctx, testConst, "R1", `{"counts":[1,0,2,0]}`, "B1"), "no decryption proof")
	requireNoErr(t, h.cc.PublishResults(h.ctx, testConst, "R2", dpResults(t, c, m, rho, []uint64{1, 0, 2, 0}), "B2"))
}
