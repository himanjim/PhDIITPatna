// wfproof.go implements the ballot well-formedness certificate: a non-interactive
// proof that a packed ballot ciphertext encrypts exactly one admissible option
// value M^j, without revealing j.
//
// # Why it is needed
//
// RecordVote checks only that a ciphertext lies in the ciphertext space. Under
// the packed encoding a ciphertext whose plaintext is not some M^j can move votes
// between options (see TestPacked_DecodeCheckedMissesBalancedIllFormedPair), and
// a remote ballot is encrypted on a device the voter controls. The certificate
// closes that gap.
//
// # Where it is checked
//
// Not in the contract, and the contract does not call this file. The voting
// client creates the certificate with the ciphertext; the verifier gateway checks
// it and refuses the ballot if it fails, before RecordVote is called; and after
// the poll closes the certificates are published in the verification pack
// (proofs.json), keyed by serial, next to the ciphertexts that ExportPublicBallots
// already returns, so that any observer can check them again
// (tools/verify_public.py, check vii). A certificate is bound to its serial and
// its ciphertext, so it needs no ledger anchor: a missing or invalid certificate
// for a counted ballot is reported as a failure. RecordVote, ApplyBallotStatuses
// and the ledger state are unchanged.
//
// # Construction
//
// A standard 1-out-of-K proof of n-th residuosity, made non-interactive with
// SHA-256. For each admissible value m_k the verifier forms
// u_k = c * g^(-m_k) mod n^2. Exactly one u_k, the one for the encrypted option,
// is an n-th power r^n; the prover shows that one of them is, without saying
// which. The certificate is the list (e_k, z_k), k = 0..K-1; the verifier
// recomputes a_k = z_k^n * u_k^(-e_k) mod n^2 and checks that the e_k add up,
// modulo 2^256, to the hash of the statement and the a_k. The hash binds the
// certificate to the serial, so a certificate cannot be moved to another
// voter's ballot.
//
// Soundness needs 2^256 to be smaller than the smallest prime factor of n, which
// holds for any election key (each prime has at least 1536 bits). The toy keys
// in the test suite are far smaller; they exercise the arithmetic, not the
// security bound.
//
// Like packed.go, this file has no Hyperledger Fabric dependency, so the
// gateway, the off-chain pipeline and the tests use the same code.
package main

import (
	"crypto/sha256"
	"fmt"
	"io"
	"math/big"
	"strings"
)

const (
	// WFProofPrefix versions the certificate encoding: "wf1.<e_0>.<z_0>.<e_1>.<z_1>..."
	// with every value in lowercase hexadecimal.
	WFProofPrefix = "wf1"

	wfDomain        = "ACCUMVOTE_WF_V1"
	wfChallengeBits = 256
)

var wfChallengeMod = new(big.Int).Lsh(big.NewInt(1), wfChallengeBits)

// wfStatement holds everything both sides derive from the public inputs.
type wfStatement struct {
	n, n2, g, c *big.Int
	u           []*big.Int // u_k = c * g^(-m_k) mod n^2
}

func newWFStatement(n, g *big.Int, slotBits, numOptions int, cHex string) (*wfStatement, error) {
	if n == nil || n.Sign() <= 0 {
		return nil, fmt.Errorf("wf: missing modulus")
	}
	if numOptions < 1 || numOptions > MaxPackedSlots {
		return nil, fmt.Errorf("wf: number of options %d out of range [1,%d]", numOptions, MaxPackedSlots)
	}
	n2 := new(big.Int).Mul(n, n)
	if g == nil || g.Sign() == 0 {
		g = new(big.Int).Add(n, big.NewInt(1))
	}
	c, err := bigFromHex(cHex)
	if err != nil {
		return nil, fmt.Errorf("wf: ciphertext: %w", err)
	}
	if err := encOneChecks(c, n2); err != nil {
		return nil, fmt.Errorf("wf: ciphertext: %w", err)
	}
	st := &wfStatement{n: n, n2: n2, g: g, c: c, u: make([]*big.Int, numOptions)}
	for k := 0; k < numOptions; k++ {
		m, err := PackedSlotValue(k, slotBits)
		if err != nil {
			return nil, err
		}
		gm := new(big.Int).Exp(g, m, n2)
		inv := new(big.Int).ModInverse(gm, n2)
		if inv == nil {
			return nil, fmt.Errorf("wf: g^m is not invertible modulo n^2")
		}
		st.u[k] = new(big.Int).Mod(new(big.Int).Mul(c, inv), n2)
	}
	return st, nil
}

// challenge hashes the statement and the prover's commitments a_k.
func (st *wfStatement) challenge(serial string, slotBits int, a []*big.Int) *big.Int {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%s|%x|%x|%x|%d|%d", wfDomain, serial, st.n, st.g, st.c, slotBits, len(a))
	for _, ak := range a {
		fmt.Fprintf(h, "|%x", ak)
	}
	return new(big.Int).SetBytes(h.Sum(nil))
}

// randUnit returns a uniformly random element of Z_n^*.
func randUnit(rnd io.Reader, n *big.Int) (*big.Int, error) {
	one := big.NewInt(1)
	for i := 0; i < 64; i++ {
		x, err := randInt(rnd, n)
		if err != nil {
			return nil, err
		}
		if x.Sign() > 0 && new(big.Int).GCD(nil, nil, x, n).Cmp(one) == 0 {
			return x, nil
		}
	}
	return nil, fmt.Errorf("wf: could not sample a unit modulo n")
}

// randInt returns a uniformly random integer in [0, max).
func randInt(rnd io.Reader, max *big.Int) (*big.Int, error) {
	byteLen := (max.BitLen() + 7) / 8
	buf := make([]byte, byteLen+16) // 128 extra bits make the modular bias negligible
	if _, err := io.ReadFull(rnd, buf); err != nil {
		return nil, err
	}
	return new(big.Int).Mod(new(big.Int).SetBytes(buf), max), nil
}

// WFProve creates the certificate for a ballot that encrypts option optionIndex
// with Paillier randomness r. It refuses if (optionIndex, r) does not reproduce
// the ciphertext, so an honest client can never certify an ill-formed ballot.
// g may be nil, meaning g = n + 1.
func WFProve(rnd io.Reader, n, g *big.Int, slotBits, numOptions, optionIndex int, r *big.Int, serial, cHex string) (string, error) {
	st, err := newWFStatement(n, g, slotBits, numOptions, cHex)
	if err != nil {
		return "", err
	}
	if optionIndex < 0 || optionIndex >= numOptions {
		return "", fmt.Errorf("wf: option index %d out of range [0,%d)", optionIndex, numOptions)
	}
	if r == nil || r.Sign() <= 0 || r.Cmp(n) >= 0 || new(big.Int).GCD(nil, nil, r, n).Cmp(big.NewInt(1)) != 0 {
		return "", fmt.Errorf("wf: randomness is not a unit modulo n")
	}
	if new(big.Int).Exp(r, n, st.n2).Cmp(st.u[optionIndex]) != 0 {
		return "", fmt.Errorf("wf: option %d and the given randomness do not reproduce the ciphertext", optionIndex)
	}

	K := numOptions
	a := make([]*big.Int, K)
	e := make([]*big.Int, K)
	z := make([]*big.Int, K)
	sumOthers := new(big.Int)
	for k := 0; k < K; k++ {
		if k == optionIndex {
			continue
		}
		// Simulated branch: pick e_k and z_k, solve for a_k.
		if e[k], err = randInt(rnd, wfChallengeMod); err != nil {
			return "", err
		}
		if z[k], err = randUnit(rnd, n); err != nil {
			return "", err
		}
		a[k] = wfCommitment(st, k, e[k], z[k])
		sumOthers.Add(sumOthers, e[k])
	}
	rho, err := randUnit(rnd, n)
	if err != nil {
		return "", err
	}
	a[optionIndex] = new(big.Int).Exp(rho, n, st.n2)

	chal := st.challenge(serial, slotBits, a)
	ei := new(big.Int).Sub(chal, sumOthers)
	ei.Mod(ei, wfChallengeMod)
	e[optionIndex] = ei
	zi := new(big.Int).Exp(r, ei, n)
	zi.Mul(zi, rho).Mod(zi, n)
	z[optionIndex] = zi

	parts := make([]string, 0, 1+2*K)
	parts = append(parts, WFProofPrefix)
	for k := 0; k < K; k++ {
		parts = append(parts, fmt.Sprintf("%x", e[k]), fmt.Sprintf("%x", z[k]))
	}
	return strings.Join(parts, "."), nil
}

// wfCommitment returns a_k = z_k^n * u_k^(-e_k) mod n^2.
func wfCommitment(st *wfStatement, k int, ek, zk *big.Int) *big.Int {
	zn := new(big.Int).Exp(zk, st.n, st.n2)
	ue := new(big.Int).Exp(st.u[k], ek, st.n2)
	ueInv := new(big.Int).ModInverse(ue, st.n2)
	if ueInv == nil {
		return big.NewInt(0) // cannot happen for a ciphertext that passed encOneChecks
	}
	return zn.Mul(zn, ueInv).Mod(zn, st.n2)
}

// WFVerify checks a certificate against a ciphertext and the serial it was cast
// for. It returns nil only if the ciphertext encrypts one of the numOptions
// admissible values. g may be nil, meaning g = n + 1.
func WFVerify(n, g *big.Int, slotBits, numOptions int, serial, cHex, proof string) error {
	st, err := newWFStatement(n, g, slotBits, numOptions, cHex)
	if err != nil {
		return err
	}
	parts := strings.Split(strings.TrimSpace(proof), ".")
	if len(parts) != 1+2*numOptions || parts[0] != WFProofPrefix {
		return fmt.Errorf("wf: certificate is not a %s proof over %d options", WFProofPrefix, numOptions)
	}
	one := big.NewInt(1)
	a := make([]*big.Int, numOptions)
	sum := new(big.Int)
	for k := 0; k < numOptions; k++ {
		ek, ok1 := new(big.Int).SetString(parts[1+2*k], 16)
		zk, ok2 := new(big.Int).SetString(parts[2+2*k], 16)
		if !ok1 || !ok2 || ek.Sign() < 0 || ek.Cmp(wfChallengeMod) >= 0 ||
			zk.Sign() <= 0 || zk.Cmp(n) >= 0 || new(big.Int).GCD(nil, nil, zk, n).Cmp(one) != 0 {
			return fmt.Errorf("wf: component %d is malformed", k)
		}
		a[k] = wfCommitment(st, k, ek, zk)
		sum.Add(sum, ek)
	}
	sum.Mod(sum, wfChallengeMod)
	if sum.Cmp(st.challenge(serial, slotBits, a)) != 0 {
		return fmt.Errorf("wf: certificate does not verify for serial %s", serial)
	}
	return nil
}
