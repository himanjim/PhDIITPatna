// decproof.go makes the decryption of the encrypted aggregate publicly
// checkable while the prototype decrypts with a single key.
//
// Paillier encryption is a bijection from (m, ρ) in Z_n × Z_n^* to Z_{n²}^*, so
// a pair (m, ρ) with C = g^m · ρ^n mod n² proves that m is the one and only
// decryption of C. The key holder can recover ρ for the aggregate C_tally, and
// ρ is the product of every ballot's randomness modulo n, so publishing it
// reveals nothing about any single ballot. Anyone can then check the published
// total with one modular exponentiation, and PublishResults refuses a total
// that fails the check.
//
// This proof exists only while one party holds the whole key. Under threshold
// custody no trustee can recover ρ, and the trustee partial decryptions with
// their correctness proofs (key-ceremony component) take its place.
//
// Like packed.go, this file has no Hyperledger Fabric dependency.
package main

import (
	"encoding/json"
	"fmt"
	"math/big"
)

// DecryptionProof is the optional "decryptionProof" member of the results
// payload passed to PublishResults. Values are hexadecimal.
type DecryptionProof struct {
	CTally string `json:"cTally"` // the declared encrypted aggregate C_tally
	Total  string `json:"total"`  // the decrypted packed total m
	Rho    string `json:"rho"`    // ρ with C_tally = g^m · ρ^n mod n²
}

// VerifyDecryption checks that m is the decryption of c: c ≡ g^m · ρ^n (mod n²).
// g may be nil, meaning g = n + 1.
func VerifyDecryption(n, g, c, m, rho *big.Int) error {
	if n == nil || n.Sign() <= 0 || c == nil || m == nil || rho == nil {
		return fmt.Errorf("decryption proof: missing value")
	}
	n2 := new(big.Int).Mul(n, n)
	if g == nil || g.Sign() == 0 {
		g = new(big.Int).Add(n, big.NewInt(1))
	}
	if err := encOneChecks(c, n2); err != nil {
		return fmt.Errorf("decryption proof: aggregate: %w", err)
	}
	if m.Sign() < 0 || m.Cmp(n) >= 0 {
		return fmt.Errorf("decryption proof: total out of range")
	}
	if rho.Sign() <= 0 || rho.Cmp(n) >= 0 || new(big.Int).GCD(nil, nil, rho, n).Cmp(big.NewInt(1)) != 0 {
		return fmt.Errorf("decryption proof: rho is not a unit modulo n")
	}
	want := new(big.Int).Exp(g, m, n2)
	want.Mul(want, new(big.Int).Exp(rho, n, n2)).Mod(want, n2)
	if want.Cmp(c) != 0 {
		return fmt.Errorf("decryption proof: g^m · rho^n does not equal the aggregate")
	}
	return nil
}

// DecryptionRandomness is used by the key holder only. Given the Carmichael
// value λ(n) of the private key, the ciphertext c and its decryption m, it
// returns ρ with c = g^m · ρ^n mod n². g may be nil, meaning g = n + 1.
func DecryptionRandomness(n, g, lambda, c, m *big.Int) (*big.Int, error) {
	n2 := new(big.Int).Mul(n, n)
	if g == nil || g.Sign() == 0 {
		g = new(big.Int).Add(n, big.NewInt(1))
	}
	gm := new(big.Int).Exp(g, m, n2)
	inv := new(big.Int).ModInverse(gm, n2)
	if inv == nil {
		return nil, fmt.Errorf("decryption proof: g^m is not invertible")
	}
	u := new(big.Int).Mul(c, inv)
	u.Mod(u, n2) // u = ρ^n mod n²
	e := new(big.Int).ModInverse(n, lambda)
	if e == nil {
		return nil, fmt.Errorf("decryption proof: n is not invertible modulo λ")
	}
	rho := new(big.Int).Exp(new(big.Int).Mod(u, n), e, n)
	if err := VerifyDecryption(n, g, c, m, rho); err != nil {
		return nil, err
	}
	return rho, nil
}

// checkResultsProof verifies the decryptionProof member of a results payload,
// if there is one, and that published counts follow from the proved total.
// It returns found=false when the payload carries no proof.
func checkResultsProof(resultJSON string, n, g *big.Int, slotBits int) (found bool, err error) {
	var payload struct {
		Counts []uint64         `json:"counts"`
		Proof  *DecryptionProof `json:"decryptionProof"`
	}
	if json.Unmarshal([]byte(resultJSON), &payload) != nil || payload.Proof == nil {
		return false, nil
	}
	p := payload.Proof
	c, err1 := bigFromHex(p.CTally)
	m, err2 := bigFromHex(p.Total)
	rho, err3 := bigFromHex(p.Rho)
	if err1 != nil || err2 != nil || err3 != nil {
		return true, fmt.Errorf("decryption proof: values must be hexadecimal")
	}
	if err := VerifyDecryption(n, g, c, m, rho); err != nil {
		return true, err
	}
	if len(payload.Counts) > 0 {
		if slotBits < MinPackedSlotBits {
			return true, fmt.Errorf("decryption proof: PACKED_SLOT_BITS is not set, so counts cannot be checked")
		}
		decoded, err := PackedDecode(m, slotBits, len(payload.Counts))
		if err != nil {
			return true, fmt.Errorf("decryption proof: %w", err)
		}
		for j := range decoded {
			if decoded[j] != payload.Counts[j] {
				return true, fmt.Errorf("decryption proof: count %d is %d but the proved total gives %d", j, payload.Counts[j], decoded[j])
			}
		}
	}
	return true, nil
}
