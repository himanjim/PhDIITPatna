// packed_test.go covers the packed (slotted) ballot encoding in isolation from
// the ledger. These tests pin down the arithmetic that makes the encrypted tally
// publicly recomputable, and - just as importantly - pin down the cases the
// encoding CANNOT detect on its own, which is why a per-ballot well-formedness
// proof remains mandatory.
package main

import (
	"math/big"
	"testing"
)

// TestPacked_SlotValueRoundTrip checks that encoding a set of votes as slot
// values and summing them reproduces the original per-option counts.
//
// This is the plaintext mirror of what homomorphic aggregation does to the
// ciphertexts: multiplying ciphertexts adds their plaintexts, so the decrypted
// aggregate is exactly the sum this test builds by hand.
func TestPacked_SlotValueRoundTrip(t *testing.T) {
	const slotBits = 24
	const numSlots = 8

	want := []uint64{5063, 2810, 4372, 91, 0, 17, 0, 1}

	total := big.NewInt(0)
	var ballots uint64
	for option, count := range want {
		v, err := PackedSlotValue(option, slotBits)
		requireNoErr(t, err)
		for i := uint64(0); i < count; i++ {
			total.Add(total, v)
			ballots++
		}
	}

	got, err := PackedDecodeChecked(total, slotBits, numSlots, ballots)
	requireNoErr(t, err)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("slot %d = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestPacked_SlotValueRejectsBadInput checks the guards on slot construction.
func TestPacked_SlotValueRejectsBadInput(t *testing.T) {
	if _, err := PackedSlotValue(0, MinPackedSlotBits-1); err == nil {
		t.Fatal("expected error for slotBits below minimum")
	}
	if _, err := PackedSlotValue(-1, 24); err == nil {
		t.Fatal("expected error for negative option index")
	}
	if _, err := PackedSlotValue(MaxPackedSlots, 24); err == nil {
		t.Fatal("expected error for option index at capacity bound")
	}
}

// TestPacked_DecodeRejectsOverflow checks that an aggregate wider than the
// declared slot space is reported rather than silently truncated.
//
// A truncating decoder would hide exactly the attack this encoding is most
// exposed to: a ballot whose plaintext is larger than any legal slot value.
func TestPacked_DecodeRejectsOverflow(t *testing.T) {
	const slotBits = 8
	const numSlots = 4 // 32 bits of legal space

	over := new(big.Int).Lsh(big.NewInt(1), uint(slotBits*numSlots)) // one bit too wide
	if _, err := PackedDecode(over, slotBits, numSlots); err == nil {
		t.Fatal("expected overflow error for aggregate wider than the slot space")
	}

	// The largest legal aggregate must still decode.
	maxLegal := new(big.Int).Sub(over, big.NewInt(1))
	if _, err := PackedDecode(maxLegal, slotBits, numSlots); err != nil {
		t.Fatalf("largest in-range aggregate must decode: %v", err)
	}
}

// TestPacked_DecodeCheckedCatchesInflatedBallot demonstrates the cheap backstop:
// a client that encrypts more than one vote inflates the digit sum, so the
// reconciliation against the number of aggregated ballots fails.
func TestPacked_DecodeCheckedCatchesInflatedBallot(t *testing.T) {
	const slotBits = 24
	const numSlots = 4

	one, err := PackedSlotValue(1, slotBits)
	requireNoErr(t, err)

	// Three honest ballots for option 1 ...
	total := new(big.Int).Mul(one, big.NewInt(3))
	// ... plus one ill-formed ballot worth 1000 votes for option 1.
	total.Add(total, new(big.Int).Mul(one, big.NewInt(1000)))

	// Four ballots were aggregated, so the counts must sum to four.
	if _, err := PackedDecodeChecked(total, slotBits, numSlots, 4); err == nil {
		t.Fatal("expected digit-sum reconciliation to reject an inflated ballot")
	}
}

// TestPacked_DecodeCheckedMissesBalancedIllFormedPair records the limitation
// that motivates the 1-of-m zero-knowledge proof.
//
// A compromised booth device can add k votes to one option and subtract k from
// another. The digit sum still reconciles, the aggregate still fits, and no
// arithmetic check at tally time can see it. Only a per-ballot well-formedness
// proof closes this, which is why the proof is not optional under packing.
func TestPacked_DecodeCheckedMissesBalancedIllFormedPair(t *testing.T) {
	const slotBits = 24
	const numSlots = 4
	const k = 1000

	opt1, err := PackedSlotValue(1, slotBits)
	requireNoErr(t, err)
	opt2, err := PackedSlotValue(2, slotBits)
	requireNoErr(t, err)

	// Honest baseline: 2000 ballots for option 1, 2000 for option 2.
	total := new(big.Int).Mul(opt1, big.NewInt(2000))
	total.Add(total, new(big.Int).Mul(opt2, big.NewInt(2000)))
	honest, err := PackedDecodeChecked(total, slotBits, numSlots, 4000)
	requireNoErr(t, err)

	// Tampered: one ballot encodes (1+k) for option 1, another encodes (1-k)
	// for option 2. Ballot count is unchanged and the digit sum still matches.
	tampered := new(big.Int).Set(total)
	tampered.Add(tampered, new(big.Int).Mul(opt1, big.NewInt(k)))
	tampered.Sub(tampered, new(big.Int).Mul(opt2, big.NewInt(k)))

	got, err := PackedDecodeChecked(tampered, slotBits, numSlots, 4000)
	if err != nil {
		t.Fatalf("balanced tampering is expected to pass arithmetic checks, got %v", err)
	}
	if got[1] != honest[1]+k || got[2] != honest[2]-k {
		t.Fatalf("expected a silent %d-vote transfer, got %v (honest %v)", k, got, honest)
	}
	t.Logf("documented limitation: %d votes moved from option 2 to option 1 undetected; "+
		"a per-ballot 1-of-m proof is required to detect this", k)
}

// TestPacked_CapacityAndParamValidation pins the sizing rules used at election
// setup: how many options fit in a modulus, and whether a slot is wide enough
// for the largest electorate the contest can serve.
func TestPacked_CapacityAndParamValidation(t *testing.T) {
	// A 3072-bit modulus at 24-bit slots leaves 127 usable slots after the
	// reserved headroom slot.
	if got := PackedCapacity(3072, 24); got != 127 {
		t.Fatalf("PackedCapacity(3072,24) = %d, want 127", got)
	}
	if got := PackedCapacity(16, 24); got != 0 {
		t.Fatalf("PackedCapacity with modulus narrower than a slot = %d, want 0", got)
	}

	// A realistic Indian constituency: 64 options, up to ~3,000,000 electors.
	if err := PackedValidateParams(3072, 24, 64, 3_000_000); err != nil {
		t.Fatalf("realistic parameters must validate: %v", err)
	}
	// A slot too narrow for the electorate must be rejected outright.
	if err := PackedValidateParams(3072, 16, 64, 3_000_000); err == nil {
		t.Fatal("expected rejection: 2^16 cannot hold 3,000,000 votes for one option")
	}
	// More slots than the modulus can carry must be rejected.
	if err := PackedValidateParams(256, 24, 64, 1000); err == nil {
		t.Fatal("expected rejection: 64 slots of 24 bits do not fit a 256-bit modulus")
	}
	if err := PackedValidateParams(3072, 24, 0, 1000); err == nil {
		t.Fatal("expected rejection: zero slots")
	}
}

// TestPacked_ResultMap checks the conversion from decoded slots to the
// option-keyed payload that PublishResults anchors.
func TestPacked_ResultMap(t *testing.T) {
	counts := []uint64{10, 20, 30, 0}
	ids := []string{testCand1, testCand2, testCand3}

	m, err := PackedResultMap(counts, ids)
	requireNoErr(t, err)
	if m[testCand1] != 10 || m[testCand2] != 20 || m[testCand3] != 30 {
		t.Fatalf("unexpected result map: %v", m)
	}
	if len(m) != 3 {
		t.Fatalf("result map has %d entries, want 3", len(m))
	}

	if _, err := PackedResultMap([]uint64{1}, ids); err == nil {
		t.Fatal("expected error when option IDs outnumber decoded slots")
	}
}
