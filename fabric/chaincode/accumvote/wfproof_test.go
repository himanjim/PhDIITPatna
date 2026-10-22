// wfproof_test.go tests the ballot well-formedness certificate (wfproof.go).
package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"strings"
	"testing"
)

const (
	wfSlotBits   = 8
	wfNumOptions = 4
)

// wfValidBallot returns a well-formed packed ballot for option j and its
// certificate for the given serial.
func wfValidBallot(t *testing.T, key *paillierTestKey, serial string, j int, r int64) (string, string) {
	t.Helper()
	c := key.packedBallotHex(t, j, wfSlotBits, r)
	proof, err := WFProve(rand.Reader, key.n, nil, wfSlotBits, wfNumOptions, j, big.NewInt(r), serial, c)
	requireNoErr(t, err)
	return c, proof
}

// TestWFProof_AcceptsEveryValidOption: an honest client can certify a ballot for
// every option, and the certificate verifies with g given or defaulted.
func TestWFProof_AcceptsEveryValidOption(t *testing.T) {
	key := newPackedTestKey(t)
	g := new(big.Int).Add(key.n, big.NewInt(1))
	for j := 0; j < wfNumOptions; j++ {
		c, proof := wfValidBallot(t, key, "S-0001", j, int64(1009+2*j))
		requireNoErr(t, WFVerify(key.n, nil, wfSlotBits, wfNumOptions, "S-0001", c, proof))
		requireNoErr(t, WFVerify(key.n, g, wfSlotBits, wfNumOptions, "S-0001", c, proof))
	}
}

// TestWFProof_RejectsIllFormedBallots: the ballots that defeat every tally-time
// check (TestPacked_DecodeCheckedMissesBalancedIllFormedPair), a zero ballot and
// a double vote cannot be certified, and a certificate taken from a valid ballot
// does not transfer to them.
func TestWFProof_RejectsIllFormedBallots(t *testing.T) {
	key := newPackedTestKey(t)
	const k = 100
	opt0, _ := PackedSlotValue(0, wfSlotBits)
	opt1, _ := PackedSlotValue(1, wfSlotBits)
	opt2, _ := PackedSlotValue(2, wfSlotBits)

	plus := new(big.Int).Mul(opt1, big.NewInt(1+k))                           // adds k votes to option 1
	minus := new(big.Int).Mod(new(big.Int).Mul(opt2, big.NewInt(1-k)), key.n) // removes k from option 2
	zero := big.NewInt(0)
	double := new(big.Int).Mul(opt0, big.NewInt(2))

	_, donor := wfValidBallot(t, key, "S-0001", 1, 2003)
	for name, m := range map[string]*big.Int{"balanced +k": plus, "balanced -k": minus, "zero": zero, "double": double} {
		const r = 3001
		c := key.encValueHex(m, r)
		for j := 0; j < wfNumOptions; j++ {
			if _, err := WFProve(rand.Reader, key.n, nil, wfSlotBits, wfNumOptions, j, big.NewInt(r), "S-0001", c); err == nil {
				t.Fatalf("%s: an ill-formed ballot was certified as option %d", name, j)
			}
		}
		if err := WFVerify(key.n, nil, wfSlotBits, wfNumOptions, "S-0001", c, donor); err == nil {
			t.Fatalf("%s: a certificate from a valid ballot verified for an ill-formed one", name)
		}
	}
}

// TestWFProof_BoundToSerialAndCiphertext: a certificate verifies only for the
// serial and the ciphertext it was made for, and any change to it is detected.
func TestWFProof_BoundToSerialAndCiphertext(t *testing.T) {
	key := newPackedTestKey(t)
	c, proof := wfValidBallot(t, key, "S-0001", 2, 4001)
	requireNoErr(t, WFVerify(key.n, nil, wfSlotBits, wfNumOptions, "S-0001", c, proof))

	if WFVerify(key.n, nil, wfSlotBits, wfNumOptions, "S-0002", c, proof) == nil {
		t.Fatalf("certificate moved to another serial still verifies")
	}
	other := key.packedBallotHex(t, 2, wfSlotBits, 4003) // same option, fresh randomness
	if WFVerify(key.n, nil, wfSlotBits, wfNumOptions, "S-0001", other, proof) == nil {
		t.Fatalf("certificate moved to another ciphertext still verifies")
	}
	if WFVerify(key.n, nil, wfSlotBits, wfNumOptions-1, "S-0001", c, proof) == nil {
		t.Fatalf("certificate verified against a different option count")
	}
	parts := strings.Split(proof, ".")
	for i := 1; i < len(parts); i++ {
		bad := append([]string(nil), parts...)
		v, _ := new(big.Int).SetString(bad[i], 16)
		bad[i] = fmt.Sprintf("%x", v.Add(v, big.NewInt(1)))
		if WFVerify(key.n, nil, wfSlotBits, wfNumOptions, "S-0001", c, strings.Join(bad, ".")) == nil {
			t.Fatalf("certificate with component %d altered still verifies", i)
		}
	}
	for _, junk := range []string{"", "wf1", "wf2." + strings.Join(parts[1:], "."), proof + ".00"} {
		if WFVerify(key.n, nil, wfSlotBits, wfNumOptions, "S-0001", c, junk) == nil {
			t.Fatalf("malformed certificate %.20q verified", junk)
		}
	}
}
