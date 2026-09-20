// auditcoin.go implements the cast-or-audit coin that provides cast-as-intended
// verification without asking the voter to make a security decision.
//
// # Why the coin exists
//
// The original design had Terminal B (the verifier kiosk) open a sealed envelope
// produced by Terminal A (the voting client) and display the candidate it named.
// That is circular: a compromised Terminal A writes both the envelope and the
// ballot, so it can seal choice X while recording choice Y and the check always
// agrees with itself.
//
// A genuine cast-as-intended check requires that the client commit to the
// ciphertext BEFORE it learns whether that ballot will be opened. Classic
// cast-or-audit puts that decision in the voter's hands, which is unusable for a
// multilingual, low-literacy electorate: audit rates collapse and comprehension
// is poor.
//
// So the decision is taken away from the voter and given to a coin that neither
// terminal can influence:
//
//		audit = ( HMAC-SHA256(K_day, "AUDIT_COIN_V1" || hC)[0:8] mod oneIn == 0 )
//
//	  - Terminal A cannot predict the outcome, because it does not hold K_day, and
//	    hC already commits to the exact ciphertext.
//	  - Terminal B cannot fake an outcome, because the function is deterministic.
//	  - After polls close K_day is published, and anyone can recompute the coin for
//	    every hC in the frozen ballot set and confirm that exactly the ballots that
//	    should have been opened were opened, and that each has a published opening.
//
// A device that receives "open" and casts anyway leaves a hole in the published
// audit log that is attributable to that booth and device.
//
// K_day is committed to before polling opens (see AuditKeyCommitment) and
// revealed afterwards, so the authority cannot choose it adaptively.
//
// This file has no Hyperledger Fabric dependencies so the same logic can run in
// the gateway, in the verifier kiosk, and in the public verifier kit.
package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strings"
)

const (
	// auditCoinDomain separates this HMAC use from every other use of K_day.
	auditCoinDomain = "AUDIT_COIN_V1"

	// auditKeyCommitDomain separates the pre-poll commitment to K_day.
	auditKeyCommitDomain = "AUDIT_KEY_COMMIT_V1"

	// DefaultAuditOneIn is the default audit rate: one ballot in twenty is
	// opened. At pilot scale this gives strong detection power; at national
	// scale it may be relaxed (1 in 50) because the absolute number of audits
	// stays very large. The active value is published on the dashboard.
	DefaultAuditOneIn = 20

	// MinAuditKeyLen is the minimum accepted length of K_day in bytes.
	MinAuditKeyLen = 16

	// AuditOutcomeCast and AuditOutcomeOpen are the two coin outcomes, used as
	// stable strings in APIs and in the published audit log.
	AuditOutcomeCast = "cast"
	AuditOutcomeOpen = "open"
)

// AuditCoin returns true when the ballot committed to by hC must be opened
// (audited) instead of cast.
//
// hC is the receipt commitment SHA256(encOneHex) exactly as the contract stores
// it in BAL::<serial> and returns it in the cast acknowledgement. It is
// lowercased and trimmed before hashing so that presentation differences between
// the kiosk, the gateway and the verifier kit cannot change the outcome.
func AuditCoin(keyDay []byte, hC string, oneIn uint32) (bool, error) {
	if len(keyDay) < MinAuditKeyLen {
		return false, fmt.Errorf("auditcoin: key is %d bytes, minimum %d", len(keyDay), MinAuditKeyLen)
	}
	if oneIn == 0 {
		return false, fmt.Errorf("auditcoin: oneIn must be >= 1")
	}
	h := strings.ToLower(strings.TrimSpace(hC))
	if h == "" {
		return false, fmt.Errorf("auditcoin: empty commitment")
	}
	mac := hmac.New(sha256.New, keyDay)
	mac.Write([]byte(auditCoinDomain))
	mac.Write([]byte(h))
	sum := mac.Sum(nil)
	v := binary.BigEndian.Uint64(sum[:8])
	return v%uint64(oneIn) == 0, nil
}

// AuditOutcome is the string form of the coin, suitable for returning to the
// voting client and for recording in the published audit log.
func AuditOutcome(keyDay []byte, hC string, oneIn uint32) (string, error) {
	open, err := AuditCoin(keyDay, hC, oneIn)
	if err != nil {
		return "", err
	}
	if open {
		return AuditOutcomeOpen, nil
	}
	return AuditOutcomeCast, nil
}

// AuditKeyCommitment returns the value published before polling opens so that
// K_day cannot be chosen after the fact. The key itself is released only after
// the poll closes, at which point any observer can recompute every coin.
func AuditKeyCommitment(keyDay []byte) (string, error) {
	if len(keyDay) < MinAuditKeyLen {
		return "", fmt.Errorf("auditcoin: key is %d bytes, minimum %d", len(keyDay), MinAuditKeyLen)
	}
	h := sha256.New()
	h.Write([]byte(auditKeyCommitDomain))
	h.Write(keyDay)
	return hex.EncodeToString(h.Sum(nil)), nil
}

// VerifyAuditKeyCommitment checks a revealed K_day against the commitment that
// was published before polling opened.
func VerifyAuditKeyCommitment(keyDay []byte, commitHex string) error {
	got, err := AuditKeyCommitment(keyDay)
	if err != nil {
		return err
	}
	want := strings.ToLower(strings.TrimSpace(commitHex))
	if !hmac.Equal([]byte(got), []byte(want)) {
		return fmt.Errorf("auditcoin: revealed key does not match published commitment")
	}
	return nil
}

// AuditLogEntry is one published record of an opened ballot. It is what makes
// the voter-facing check reproducible by third parties: anyone can recompute
// Enc(PackedSlotValue(OptionIndex), RandomnessHex) and confirm it hashes to HC.
//
// An opened ballot is never submitted to RecordVote and therefore never enters
// the tally universe.
type AuditLogEntry struct {
	HC             string `json:"hC"`
	ConstituencyID string `json:"constituencyID"`
	OptionIndex    int    `json:"optionIndex"`
	RandomnessHex  string `json:"randomnessHex"`
	BoothID        string `json:"boothID,omitempty"`
	DeviceID       string `json:"deviceID,omitempty"`
	OpenedAt       string `json:"openedAt,omitempty"`
}

// ReconcileAuditLog checks a published audit log against the frozen ballot set.
//
// castHCs are the commitments of every ballot that was actually cast (the hC
// column of the freeze list S). openedHCs are the commitments that appear in the
// published audit log. Every commitment the coin marked "open" must appear in
// the audit log and must NOT appear in the cast set; every commitment the coin
// marked "cast" must appear in the cast set and must NOT have been opened.
//
// The returned slices name the two failure modes: ballots that should have been
// opened but were cast instead (a device ignoring the coin), and ballots that
// were opened without the coin calling for it (an attempt to discard a ballot).
func ReconcileAuditLog(keyDay []byte, oneIn uint32, castHCs, openedHCs []string) (shouldHaveOpened, openedWithoutCause []string, err error) {
	openedSet := make(map[string]struct{}, len(openedHCs))
	for _, h := range openedHCs {
		openedSet[strings.ToLower(strings.TrimSpace(h))] = struct{}{}
	}
	for _, h := range castHCs {
		key := strings.ToLower(strings.TrimSpace(h))
		open, e := AuditCoin(keyDay, key, oneIn)
		if e != nil {
			return nil, nil, e
		}
		if open {
			// The coin said open, yet this ballot appears in the cast set.
			shouldHaveOpened = append(shouldHaveOpened, key)
		}
	}
	for h := range openedSet {
		open, e := AuditCoin(keyDay, h, oneIn)
		if e != nil {
			return nil, nil, e
		}
		if !open {
			openedWithoutCause = append(openedWithoutCause, h)
		}
	}
	return shouldHaveOpened, openedWithoutCause, nil
}
