// auditattest.go closes the insider route around the audit coin.
//
// The coin is only as good as the secrecy of K_day. A voting terminal that knew
// K_day could compute the coin for a ballot before committing to it, and cheat
// only on ballots the coin will not open. Two controls make that detectable or
// pointless, and both are checked by the public verifier:
//
//  1. Query-log attestation. K_day lives as a non-exportable key in the audit
//     service's HSM, generated under trustee observation. At close the HSM signs
//     the number of indicators it issued and a digest of the query log with an
//     Ed25519 key whose public half (AUDIT_SIGNING_KEY) was published before
//     polling. A terminal that grinds by asking for indicators leaves queries
//     that are neither cast nor opened, and hiding them from the published log
//     breaks the signed count.
//  2. Voter-requested openings. A voter may ask to check any ballot whatever the
//     coin says (RecordRequestedAuditOpening). Such an opening must still have
//     an indicator query and must not be a recorded ballot.
//
// This file has no Hyperledger Fabric dependency.
package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// RequestedByVoter marks an opening the voter asked for.
const RequestedByVoter = "voter"

const queryLogDomain = "QUERYLOG_V1"

// QueryLogAttestation is what the audit service's HSM signs at close.
type QueryLogAttestation struct {
	ConstituencyID string `json:"constituencyID"`
	Queries        int    `json:"queries"`
	QueryLogDigest string `json:"queryLogDigest"` // QueryLogDigest of the published query log
	Signature      string `json:"signature"`      // hex Ed25519 signature over QueryLogAttestationMessage
}

// QueryLogDigest is SHA-256 over the canonical query log: the JSON array
// [{"hC":"<lowercase hex>"}, ...] sorted by hC, which is exactly the
// querylog.json that tools/export_freeze.js writes.
func QueryLogDigest(hCs []string) string {
	norm := make([]string, 0, len(hCs))
	for _, h := range hCs {
		norm = append(norm, strings.ToLower(strings.TrimSpace(h)))
	}
	sort.Strings(norm)
	rows := make([]map[string]string, len(norm))
	for i, h := range norm {
		rows[i] = map[string]string{"hC": h}
	}
	b, _ := json.Marshal(rows)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// QueryLogAttestationMessage is the byte string the HSM signs.
func QueryLogAttestationMessage(constituencyID string, queries int, digest string) []byte {
	return []byte(fmt.Sprintf("%s|%s|%d|%s", queryLogDomain, constituencyID, queries, strings.ToLower(digest)))
}

// VerifyQueryLogAttestation checks an attestation against the published query
// log and the pre-published audit signing key.
func VerifyQueryLogAttestation(pubKeyHex string, a QueryLogAttestation, hCs []string) error {
	pub, err := hex.DecodeString(strings.TrimSpace(pubKeyHex))
	if err != nil || len(pub) != ed25519.PublicKeySize {
		return fmt.Errorf("query-log attestation: bad audit signing key")
	}
	sig, err := hex.DecodeString(strings.TrimSpace(a.Signature))
	if err != nil {
		return fmt.Errorf("query-log attestation: signature is not hexadecimal")
	}
	if !ed25519.Verify(pub, QueryLogAttestationMessage(a.ConstituencyID, a.Queries, a.QueryLogDigest), sig) {
		return fmt.Errorf("query-log attestation: signature does not verify")
	}
	if a.Queries != len(hCs) {
		return fmt.Errorf("query-log attestation: HSM issued %d indicators but the log has %d entries", a.Queries, len(hCs))
	}
	if got := QueryLogDigest(hCs); !strings.EqualFold(got, a.QueryLogDigest) {
		return fmt.Errorf("query-log attestation: digest does not match the published log")
	}
	return nil
}

// CheckRequestedOpenings returns the voter-requested openings that are not
// justified: those with no indicator query, and those whose commitment was
// also recorded as a cast ballot.
func CheckRequestedOpenings(requestedHCs, castHCs, queriedHCs []string) (unqueried, alsoCast []string) {
	set := func(xs []string) map[string]struct{} {
		m := make(map[string]struct{}, len(xs))
		for _, x := range xs {
			m[strings.ToLower(strings.TrimSpace(x))] = struct{}{}
		}
		return m
	}
	cast, queried := set(castHCs), set(queriedHCs)
	for _, h := range requestedHCs {
		k := strings.ToLower(strings.TrimSpace(h))
		if _, ok := queried[k]; !ok {
			unqueried = append(unqueried, k)
		}
		if _, ok := cast[k]; ok {
			alsoCast = append(alsoCast, k)
		}
	}
	return unqueried, alsoCast
}
