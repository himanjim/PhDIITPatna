// packed.go implements the packed (slotted) ballot encoding used to move the
// voter's choice out of the plaintext candidateID label and into the Paillier
// ciphertext itself.
//
// # Motivation
//
// In the original prototype each cast carried two things: a plaintext
// candidateID and a ciphertext EncOneHex = Enc(1). The candidate label was what
// actually carried the vote, which had two consequences:
//
//  1. Ballot secrecy depended on private-data access control rather than on
//     cryptography, because every trustee collection member could read the
//     plaintext candidateID.
//  2. The public could never recompute the encrypted tally, because the
//     per-candidate buckets were derived from private data. Public world state
//     held only SHA256(EncOneHex), and a hash is not homomorphic.
//
// Packed encoding fixes both at once. The plaintext of the ballot ciphertext is
// M^j, where j is the index of the chosen option and M = 2^slotBits. Because
// Paillier is additively homomorphic, multiplying every ballot ciphertext yields
// a single aggregate whose plaintext is
//
//	sum_j N_j * M^j
//
// i.e. the per-option counts laid out as base-M digits. One decryption of one
// aggregate yields every count. Every ballot therefore carries the same constant
// candidate label (PackedSentinel), so the tally has exactly one bucket and the
// ciphertexts can be published without revealing anything.
//
// Ill-formedness note (important)
//
// Packing makes a ballot well-formedness proof mandatory rather than optional.
// If a client submits a ciphertext whose plaintext is not exactly some M^j, the
// excess can carry from one digit position into the next and silently corrupt an
// adjacent option's count. PackedDecodeChecked implements the cheap arithmetic
// backstops (bit-length bound and digit-sum reconciliation); it does not and
// cannot replace a 1-of-m zero-knowledge proof verified per ballot.
//
// This file deliberately has no Hyperledger Fabric dependencies so that the same
// logic can be reused by off-chain tooling (tally pipeline, public verifier kit)
// and unit-tested without a ledger.
package main

import (
	"fmt"
	"math/big"
)

const (
	// PackedSentinel is the constant candidate label written to every packed
	// ballot. Seeding CANDLIST::<constituency> with this single value makes
	// TallyPrepare accumulate all ballots into one bucket, which is exactly the
	// packed aggregate.
	PackedSentinel = "PACKED"

	// DefaultPackedSlotBits is the default width of one option slot, in bits.
	// 2^24 = 16,777,216 is comfortably larger than the electorate of any Indian
	// parliamentary or assembly constituency, so a single option's count can
	// never overflow its slot under honest operation.
	DefaultPackedSlotBits = 24

	// MinPackedSlotBits guards against a configuration so narrow that ordinary
	// turnout would overflow a slot.
	MinPackedSlotBits = 8

	// MaxPackedSlots bounds the number of option slots accepted by this
	// implementation, independent of modulus size.
	MaxPackedSlots = 256
)

// PackedSlotValue returns the plaintext value M^optionIndex that encodes a vote
// for the option at optionIndex, where M = 2^slotBits.
//
// The caller encrypts this value under the election public key; the resulting
// ciphertext is what is submitted as encOneHex.
func PackedSlotValue(optionIndex, slotBits int) (*big.Int, error) {
	if slotBits < MinPackedSlotBits {
		return nil, fmt.Errorf("packed: slotBits %d below minimum %d", slotBits, MinPackedSlotBits)
	}
	if optionIndex < 0 || optionIndex >= MaxPackedSlots {
		return nil, fmt.Errorf("packed: option index %d out of range [0,%d)", optionIndex, MaxPackedSlots)
	}
	v := new(big.Int).Lsh(big.NewInt(1), uint(slotBits*optionIndex))
	return v, nil
}

// PackedCapacity reports how many option slots fit inside a plaintext space of
// modulusBits bits at the given slot width.
//
// One slot of headroom is reserved so that the aggregate can never reach the
// modulus and wrap, which would silently corrupt the highest option's count.
func PackedCapacity(modulusBits, slotBits int) int {
	if slotBits <= 0 || modulusBits <= slotBits {
		return 0
	}
	c := modulusBits/slotBits - 1
	if c < 0 {
		return 0
	}
	if c > MaxPackedSlots {
		return MaxPackedSlots
	}
	return c
}

// PackedValidateParams checks that a proposed (modulusBits, slotBits, numSlots)
// configuration can represent every option without slot overflow or modular
// wraparound, given the largest electorate the contest may serve.
//
// It is intended to be run once at election setup and its outcome published in
// the election parameters, not run per ballot.
func PackedValidateParams(modulusBits, slotBits, numSlots, maxElectors int) error {
	if slotBits < MinPackedSlotBits {
		return fmt.Errorf("packed: slotBits %d below minimum %d", slotBits, MinPackedSlotBits)
	}
	if numSlots <= 0 || numSlots > MaxPackedSlots {
		return fmt.Errorf("packed: numSlots %d out of range (0,%d]", numSlots, MaxPackedSlots)
	}
	if cap := PackedCapacity(modulusBits, slotBits); numSlots > cap {
		return fmt.Errorf("packed: %d slots of %d bits do not fit in a %d-bit modulus (capacity %d)",
			numSlots, slotBits, modulusBits, cap)
	}
	// One slot must hold the largest count the contest can produce.
	maxPerSlot := new(big.Int).Lsh(big.NewInt(1), uint(slotBits))
	if maxPerSlot.Cmp(big.NewInt(int64(maxElectors))) <= 0 {
		return fmt.Errorf("packed: slot width 2^%d cannot hold up to %d votes for one option",
			slotBits, maxElectors)
	}
	return nil
}

// PackedDecode splits a decrypted aggregate into its per-option counts.
//
// The aggregate must be the plaintext recovered from the product of all ballot
// ciphertexts. An aggregate wider than numSlots*slotBits bits indicates that at
// least one ballot was ill-formed (or that the parameters were mis-sized), and
// is reported as an error rather than silently truncated.
func PackedDecode(total *big.Int, slotBits, numSlots int) ([]uint64, error) {
	if total == nil {
		return nil, fmt.Errorf("packed: nil aggregate")
	}
	if total.Sign() < 0 {
		return nil, fmt.Errorf("packed: negative aggregate")
	}
	if slotBits < MinPackedSlotBits {
		return nil, fmt.Errorf("packed: slotBits %d below minimum %d", slotBits, MinPackedSlotBits)
	}
	if numSlots <= 0 || numSlots > MaxPackedSlots {
		return nil, fmt.Errorf("packed: numSlots %d out of range (0,%d]", numSlots, MaxPackedSlots)
	}
	if slotBits >= 64 {
		return nil, fmt.Errorf("packed: slotBits %d must be < 64 for uint64 counts", slotBits)
	}
	if total.BitLen() > slotBits*numSlots {
		return nil, fmt.Errorf("packed: aggregate occupies %d bits, exceeds %d slot(s) of %d bits; "+
			"at least one ballot was ill-formed", total.BitLen(), numSlots, slotBits)
	}

	mask := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), uint(slotBits)), big.NewInt(1))
	out := make([]uint64, numSlots)
	tmp := new(big.Int)
	for j := 0; j < numSlots; j++ {
		tmp.Rsh(total, uint(slotBits*j))
		tmp.And(tmp, mask)
		out[j] = tmp.Uint64()
	}
	return out, nil
}

// PackedDecodeChecked decodes an aggregate and additionally reconciles the sum
// of the per-option counts against the number of ballots that were aggregated.
//
// A mismatch means the aggregate does not correspond to expectedBallots
// well-formed unit votes. This catches naive inflation (a ballot encrypting more
// than one vote) but NOT a balanced pair of ill-formed ballots that cancel out,
// which is precisely why a per-ballot 1-of-m proof remains required.
func PackedDecodeChecked(total *big.Int, slotBits, numSlots int, expectedBallots uint64) ([]uint64, error) {
	counts, err := PackedDecode(total, slotBits, numSlots)
	if err != nil {
		return nil, err
	}
	var sum uint64
	for _, c := range counts {
		// Guard against uint64 overflow before it can mask a bad sum.
		if sum+c < sum {
			return nil, fmt.Errorf("packed: per-option counts overflow uint64")
		}
		sum += c
	}
	if sum != expectedBallots {
		return nil, fmt.Errorf("packed: counts sum to %d but %d ballots were aggregated; "+
			"at least one ballot was ill-formed", sum, expectedBallots)
	}
	return counts, nil
}

// PackedResultMap converts decoded counts into the option-keyed map used by the
// published results payload. Options are named by optionIDs[j]; the slice length
// determines how many slots are reported.
func PackedResultMap(counts []uint64, optionIDs []string) (map[string]uint64, error) {
	if len(optionIDs) > len(counts) {
		return nil, fmt.Errorf("packed: %d option IDs but only %d decoded slots", len(optionIDs), len(counts))
	}
	out := make(map[string]uint64, len(optionIDs))
	for j, id := range optionIDs {
		out[id] = counts[j]
	}
	return out, nil
}
