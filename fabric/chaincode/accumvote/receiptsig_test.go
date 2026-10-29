// receiptsig_test.go tests the signed receipt (receiptsig.go).
package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

// TestReceiptSignature_BindsEveryField: a signed receipt verifies under the
// published key, and changing any field, or using another key, breaks it.
func TestReceiptSignature_BindsEveryField(t *testing.T) {
	seed := sha256.Sum256([]byte("test-gateway"))
	priv := ed25519.NewKeyFromSeed(seed[:])
	pub := hex.EncodeToString(priv.Public().(ed25519.PublicKey))
	f := ReceiptFields{EID: "EID-2026", Serial: "S-0001", TxID: "tx-1", HC: "ABCD", Epoch: "E1", CastTime: "2026-05-01T09:00:00Z"}
	sig := SignReceipt(priv, f)
	requireNoErr(t, VerifyReceiptSignature(pub, f, sig))
	lower := f
	lower.HC = "abcd"
	requireNoErr(t, VerifyReceiptSignature(pub, lower, sig))

	for i, mutate := range []func(*ReceiptFields){
		func(r *ReceiptFields) { r.EID = "EID-2027" },
		func(r *ReceiptFields) { r.Serial = "S-0002" },
		func(r *ReceiptFields) { r.TxID = "tx-2" },
		func(r *ReceiptFields) { r.HC = "abce" },
		func(r *ReceiptFields) { r.Epoch = "E2" },
		func(r *ReceiptFields) { r.CastTime = "2026-05-01T09:00:01Z" },
	} {
		g := f
		mutate(&g)
		if VerifyReceiptSignature(pub, g, sig) == nil {
			t.Fatalf("field %d is not bound by the signature", i)
		}
	}
	otherSeed := sha256.Sum256([]byte("other-gateway"))
	otherPub := hex.EncodeToString(ed25519.NewKeyFromSeed(otherSeed[:]).Public().(ed25519.PublicKey))
	requireErrContains(t, VerifyReceiptSignature(otherPub, f, sig), "does not verify")
}
