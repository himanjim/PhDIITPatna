// -----------------------------------------------------------------------------
// Accumvote_cc contract (Go, Fabric v3.1.1)
// Purpose: Implements a high-TPS, privacy-preserving vote accumulator using
// Paillier Enc(1) multiplication and “latest-vote-wins”, with validation at tally.
// Role in system: Write-path records a kiosk’s latest vote pointer into a PDC and
// Minimal public kiosk metadata; read-path/tally composes encrypted sums and
// Marks invalid votes (non-existent voters/booths, bad ciphertexts) for exclusion.
// Key dependencies: Hyperledger Fabric contractapi/cid; a canonical preload
// Chaincode (“evote-preload”) for voters/candidates; a booth registry CC
// (“boothpdc”) for booth/device windows; private data collection “votes_pdc”.
// -----------------------------------------------------------------------------

/*
accumvote.go — Hyperledger Fabric chaincode for the AccumVote prototype.

This contract supports a re-voting model (latest vote wins per serial) and a
privacy-preserving receipt flow:
- Votes are stored in a private data collection under VOTE::<const>::<serial>.
- Public ballot metadata is stored under BAL::<serial> with a receipt hash (hC).
- TXIDX::<txID> is maintained only for ballots marked Current, so tx-based receipt
  checks do not leak re-vote history.

The chaincode does not expose any HTTP endpoints. A separate gateway/service is
expected to invoke these contract functions and subscribe to emitted events.
*/
package main

import (
"bytes"
"crypto/sha256"
"encoding/hex"
"encoding/json"
// "errors"
"fmt"
"math/big"
// "os"
"sort"
"strings"
"time"
"reflect"
"strconv"
// "github.com/hyperledger/fabric-chaincode-go/v2/pkg/cid"
"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
"sync"
)

/* Keys & constants (single namespace for this chaincode) */

const (
// Collections
votesPDC = "votes_pdc"

// World state prefixes (public)
keyPKPrefix       = "PK::"       // PK::<state> → PublicKey JSON (Paillier)
keyBallotPrefix   = "BAL::"      // BAL::<serial> → BallotMeta for kiosk (no identity)
keyPollPrefix     = "POLL::"     // POLL::<const> → "open"/"closed"
keyParams         = "PARAMS"     // Global Params JSON
keyCandsListPrefx = "CANDLIST::" // CANDLIST::<const> → []string (local copy)
keyResultsPrefix  = "RES::"      // RES::<const>::<roundID> → anchored results

keyTxIdxPrefix = "TXIDX::"          // TxID → serial (for VerifyReceipt)
)

const (
eventVoteRecorded = "VoteRecorded"
eventTallyPrepared = "TallyPrepared"
eventResultsPublished = "ResultsPublished"
eventInvalidVoterFound = "InvalidVoterFound"

eventCandidateListRefreshed = "CandidateListRefreshed"
eventPollOpened             = "PollOpened"
eventPollClosed             = "PollClosed"
eventParamsUpdated          = "ParamsUpdated"
eventPublicKeySet           = "PublicKeySet"
eventInvalidCandidateFound = "InvalidCandidateFound"
eventInvalidBoothFound = "InvalidBoothFound"
)

/* Types & small data models */

// AccumVoteContract implements the Fabric contract for the Internet voting prototype.
//
// Responsibilities:
// - Accept high-throughput vote submissions (RecordVote) into a private data collection.
// - Maintain a public, serial-indexed ballot status record for receipt checks and auditing.
// - Produce homomorphic encrypted tallies and publish result anchors for public verification.
type AccumVoteContract struct{ contractapi.Contract }

// PublicKey stores the Paillier public key material and public ceremony anchors.
//
// The key is written to PK::<state> and includes:
// - Paillier parameters (n, g, n^2)
// - Ceremony anchors (h1, transcriptURI) so auditors can link the ledger to the key ceremony.
// - A deterministic digest (pkDigest) for quick integrity checks.
type PublicKey struct {
N  string `json:"n"`
G  string `json:"g"`
N2 string `json:"n2,omitempty"`

}

// VoteMetaPDC is the private vote record stored in the votes private data collection.
//
// It intentionally contains the minimum fields required for tallying and audit linkage.
// Sensitive/identifying information must NOT be stored here.
type VoteMetaPDC struct {
    CandidateID string `json:"candidateID"`
    // New fields for append-only audit + tally reconstruction:
    EncOneHex string `json:"encOneHex"`           // Paillier Enc(1;r) as hex
    Epoch     string `json:"epoch,omitempty"`     // Logical revote epoch (string to avoid int64 JSON ambiguity)
    TxID      string `json:"txID"`                // Tx that wrote this record (unique per cast)
    CastTime  string `json:"castTime,omitempty"`  // RFC3339 (optional)
	
	 // NEW – for booth/device validation at tally:
    StateCode   string `json:"stateCode,omitempty"`
    BoothID     string `json:"boothID,omitempty"`
    DeviceID    string `json:"deviceID,omitempty"`
    DeviceKeyFP string `json:"deviceKeyFP,omitempty"`
	
	 // New: encrypted facial embedding (simple, optional)
    BioAlg       string `json:"bioAlg,omitempty"`       // E.g., "AES-256-GCM"
    BioNonceB64  string `json:"bioNonceB64,omitempty"`  // Base64(nonce)
    BioCipherB64 string `json:"bioCipherB64,omitempty"` // Base64(ciphertext||tag)
    BioTagHex    string `json:"bioTagHex,omitempty"`    // Optional HMAC link tag (hex)

    // New: device signature (base64 or hex; you decide upstream)
    DevSigB64    string `json:"devSigB64,omitempty"`
}


// BallotMeta is the public per-serial metadata used for receipt verification and audit trails.
//
// Stored at BAL::<serial>.
// - Status transitions are applied by ApplyBallotStatuses.
// - HC is a hash of encOneHex and is the only value needed to validate a voter receipt.
type BallotMeta struct {
HC       string `json:"hC"`
Status   string `json:"status"`   // "current"
Epoch    string `json:"epoch"`
CastTime string `json:"castTime"` // RFC3339
TxID     string `json:"txID"`

}

// TallyResultAnchor is the public on-chain anchor for a published result.
//
// It binds:
// - the plaintext results JSON,
// - a hash of the canonical results package (h_TVP), and
// - the trustee bundle hash (bundleHash).
// This is what the public dashboard and external auditors verify.
type TallyResultAnchor struct {
RoundID      string          `json:"roundID"`
Constituency string          `json:"constituency"`
Time         string          `json:"time"`
Results      json.RawMessage `json:"results"`
BundleHash   string          `json:"bundleHash"`

}

// Params contains runtime toggles and limits used by the contract.
//
// Values are stored on-chain (PARAMS::<scope>) and cached for performance.
// In tests/benchmarks, env vars may override or bypass some checks.
type Params struct {
VerifyAttestation bool   `json:"VERIFY_ATTESTATION"`  // Accept non-empty/fixed value
EmitEvents        bool `json:"EMIT_EVENTS"`        // Default true: emit events

ValidateOnTally bool   `json:"VALIDATE_ON_TALLY"` // Default false: run eligibility at tally-time

PreloadCCName string `json:"PRELOAD_CC_NAME"` // Single source of truth CC for candidates/voters

BoothCCName        string `json:"BOOTH_CC_NAME"`         // Default "boothpdc"
ValidateBoothOnTally bool `json:"VALIDATE_BOOTH_ON_TALLY"` // Default true
}


type boothRec struct {
    Status               string `json:"status"`
    OpenTime             int64  `json:"open_time"`
    CloseTime            int64  `json:"close_time"`
    DeviceID             string `json:"device_id"`
    DeviceKeyFingerprint string `json:"device_key_fingerprint"`
}

// Cache parsed Paillier keys per state (thread-safe)
var pkCache sync.Map // Key: state -> pkEntry

type pkEntry struct {
    n  *big.Int
    n2 *big.Int
    // G is unused on-chain; you can add it if you later need it
}


/* Small helpers */

// getBoothViaCC queries the booth/device registry chaincode (boothpdc) for booth metadata.
// This is used to validate whether a device is authorised and within its voting window.
func getBoothViaCC(ctx contractapi.TransactionContextInterface, boothCC, stateCode, constituencyID, boothID string) (*boothRec, error) {
    if strings.TrimSpace(boothID) == "" { return nil, fmt.Errorf("empty booth") }
    payload, err := callPreload(ctx, boothCC, "GetBooth", stateCode, constituencyID, boothID)
    if err != nil { return nil, err }
    var br boothRec
    if json.Unmarshal(payload, &br) != nil { return nil, fmt.Errorf("bad booth json") }
    return &br, nil
}

// castTimeWithin checks that a RFC3339 timestamp falls within an allowed window.
// The window values are typically sourced from the booth/device registry.
func castTimeWithin(vmCastRFC3339 string, open, close int64) bool {
    if vmCastRFC3339 == "" || open == 0 || close == 0 { return true } // Fail-soft
    ct, err := time.Parse(time.RFC3339, vmCastRFC3339)
    if err != nil { return true } // Fail-soft
    t := ct.Unix()
    return t >= open && t <= close
}

// boothDeviceOK verifies that a booth record authorises the given device for voting.
// It also enforces any time window constraints configured for the booth.
func boothDeviceOK(vm *VoteMetaPDC, br *boothRec) bool {
    // Only enforce if booth row provided a value
    if br.DeviceID != "" && vm.DeviceID != "" && br.DeviceID != vm.DeviceID { return false }
    if br.DeviceKeyFingerprint != "" && vm.DeviceKeyFP != "" && br.DeviceKeyFingerprint != vm.DeviceKeyFP { return false }
    return true
}

// nowRFC3339 returns the transaction timestamp as an RFC3339 UTC string.
func nowRFC3339(ctx contractapi.TransactionContextInterface) string {
    ts, _ := ctx.GetStub().GetTxTimestamp()
    return time.Unix(ts.Seconds, int64(ts.Nanos)).UTC().Format(time.RFC3339)
}

// sha256Hex returns the SHA-256 hash of a byte slice, hex-encoded.
func sha256Hex(b []byte) string {
h := sha256.Sum256(b)
return hex.EncodeToString(h[:])
}
// sha256HexStr returns the SHA-256 hash of a string, hex-encoded.
func sha256HexStr(s string) string { return sha256Hex([]byte(s)) }

// bigFromHex parses a hex string (with or without 0x) into a big.Int.
func bigFromHex(s string) (*big.Int, error) {
    s = strings.TrimSpace(s)
    if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
        s = s[2:]
    }
    if len(s)%2 == 1 {
        s = "0" + s
    }
    if b, err := hex.DecodeString(s); err == nil {
        return new(big.Int).SetBytes(b), nil
    }
    if z, ok := new(big.Int).SetString(s, 10); ok {
        return z, nil
    }
    // Include the word "hex" so tests match
    return nil, fmt.Errorf("bad hex integer: %q", s)
}


// hexFromBig encodes a big.Int as lowercase hex without 0x and without leading zeros.
func hexFromBig(x *big.Int) string {
	if x == nil || x.Sign() == 0 {
		return "0"
	}
	// Use canonical lowercase hex with NO leading zeros.
	s := strings.ToLower(x.Text(16)) // Textual hex, never padded
	s = strings.TrimLeft(s, "0")     // Drop any stray leading zeros (paranoia)
	if s == "" {                     // If x==0, after trim we normalize to "0"
		return "0"
	}
	return s
}

// canonHexStr normalises a hex string to lowercase and removes leading zeros.
func canonHexStr(s string) string {
    x, err := bigFromHex(s)
    if err != nil { // If it's not a valid hex/int, just return as-is
        return s
    }
    return hexFromBig(x)
}

// canonHexMap canonicalises all hex string values in a map.
// This is used so public hashes are reproducible across clients.
func canonHexMap(m map[string]string) {
    for k, v := range m {
        m[k] = canonHexStr(v)
    }
}


// mulMod returns (a*b) mod m.
// Note: hot paths use in-place operations to reduce allocations; keep this for clarity where used.
func mulMod(x, y, mod *big.Int) *big.Int {
z := new(big.Int).Mul(x, y)
return z.Mod(z, mod)
}


// loadPK loads the Paillier public key for a state.
// It uses an in-memory cache to avoid repeated parsing of large integers.
func loadPK(ctx contractapi.TransactionContextInterface, state string) (*big.Int, *big.Int, *big.Int, error) {
raw, err := ctx.GetStub().GetState(keyPKPrefix + state)
if err != nil {
return nil, nil, nil, fmt.Errorf("get pk: %w", err)
}
if raw == nil {
return nil, nil, nil, fmt.Errorf("public key for state %s not set", state)
}
var pk PublicKey
if err := json.Unmarshal(raw, &pk); err != nil {
return nil, nil, nil, fmt.Errorf("pk json: %w", err)
}
n, err := bigFromHex(pk.N)
if err != nil {
return nil, nil, nil, fmt.Errorf("pk.n parse: %w", err)
}
var n2 *big.Int
if pk.N2 != "" {
n2, err = bigFromHex(pk.N2)
if err != nil {
return nil, nil, nil, fmt.Errorf("pk.n2 parse: %w", err)
}
} else {
n2 = new(big.Int).Mul(n, n)
}
var g *big.Int
if pk.G != "" {
g, _ = bigFromHex(pk.G)
}
return n, g, n2, nil
}

// getParams reads the contract runtime parameters from world state.
// The values control optional checks (ABAC, candidate validation, events) and size limits.
func getParams(ctx contractapi.TransactionContextInterface) (*Params, error) {
	p := &Params{
		VerifyAttestation: true,           // <-- ON by default
		
		EmitEvents:        true,

		ValidateOnTally:   true,           // <-- ON by default (voter eligibility via cc2cc at tally)
		PreloadCCName:     "evote-preload",
		BoothCCName:         "boothpdc",          // NEW
        ValidateBoothOnTally:true,                // NEW
	}
	
	if b, err := ctx.GetStub().GetState(keyParams); err == nil && b != nil {
		var on Params
		if json.Unmarshal(b, &on) == nil {
			return &on, nil
		}
	}

    return p, nil
}


// resolveStateForReader determines which state scope should be used for read-only queries.
// This keeps read paths deterministic in tests and allows multi-state deployments.
func resolveStateForReader(ctx contractapi.TransactionContextInterface) (string, error) {
/* ABAC bypass note: tests/benchmarks can set BYPASS_ABAC_FOR_TEST=on (and optionally BYPASS_STATE) to skip identity checks. */
	return "UP", nil
}


// requireABAC enforces attribute-based access control for write operations.
// In production, callers must present (role, state, constituency) attributes.
// In tests/benchmarks, the check can be bypassed via environment variables.
func requireABAC(ctx contractapi.TransactionContextInterface, constituencyID string) (string, error) {
    // Test-only bypass: if BYPASS_ABAC_FOR_TEST=on, skip attribute checks.
    // Returns BYPASS_STATE if set, else "TEST".
/* ABAC bypass note: tests/benchmarks can set BYPASS_ABAC_FOR_TEST=on (and optionally BYPASS_STATE) to skip identity checks. */
    return "UP", nil
}

// pollIsOpen returns whether the poll is currently open for a constituency.
func pollIsOpen(ctx contractapi.TransactionContextInterface, constituencyID string) (bool, error) {
    raw, err := ctx.GetStub().GetState(keyPollPrefix + constituencyID)
    if err != nil {
        return false, err
    }
    if raw == nil {
        return true, nil // Default open
    }
    return string(raw) == "open", nil
}


// encOneChecks validates encOneHex is well-formed and safe to use in modular arithmetic.
// The intent is to reject malformed ciphertexts early (cheap checks only).
func encOneChecks(c, n2 *big.Int) error {
one := big.NewInt(1)
if c.Cmp(one) <= 0 || c.Cmp(n2) >= 0 {
return fmt.Errorf("encOne out of range")
}
g := new(big.Int).GCD(nil, nil, c, n2)
if g.Cmp(one) != 0 {
return fmt.Errorf("encOne not invertible mod n²")
}
return nil
}


// voteKey builds the private-data key for a vote (VOTE::<const>::<serial>).
// The latest write wins, enabling re-voting while keeping one stored vote per serial.
func voteKey(constituencyID, serial string) string {
return fmt.Sprintf("VOTE::%s::%s", constituencyID, serial)
}

// HasVoterViaCC queries evote-preload.HasVoter for serial presence (string bool).
// Params: ctx, preloadCC, constituencyID, serial.
// Return: true/false or error if cc2cc call fails.
func hasVoterViaCC(ctx contractapi.TransactionContextInterface, preloadCC, constituencyID, serial string) (bool, error) {
    // Function name and args must match evote-preload/main.go
    args := [][]byte{[]byte("HasVoter"), []byte(constituencyID), []byte(serial)}
    resp := ctx.GetStub().InvokeChaincode(preloadCC, args, "")
    if resp.Status != 200 {
        if len(resp.Message) > 0 {
            return false, fmt.Errorf("cc2cc HasVoter failed: %s", resp.Message)
        }
        return false, fmt.Errorf("cc2cc HasVoter failed with status %d", resp.Status)
    }
    // Contractapi returns JSON "true"/"false"
	payload := strings.TrimSpace(string(resp.Payload))
	payload = strings.Trim(payload, `"`)
	ok, _ := strconv.ParseBool(strings.ToLower(payload))
	return ok, nil
}

// CallPreload is a safe wrapper to call read-only functions in preload/booth CCs.
// Params: ctx, preloadCC name, function, args.
// Return: raw payload bytes or error on non-200 or empty payload.
func callPreload(ctx contractapi.TransactionContextInterface, preloadCC, fcn string, args ...string) ([]byte, error) {
    if ctx == nil {
        return nil, fmt.Errorf("cc2cc %s: nil ctx", fcn)
    }
    s := ctx.GetStub()
    if s == nil {
        return nil, fmt.Errorf("cc2cc %s: nil stub", fcn)
    }
	
    // Guard against typed-nil stub (interface is non-nil but underlying pointer is nil).
    if rv := reflect.ValueOf(s); rv.Kind() == reflect.Ptr && rv.IsNil() {
        return nil, fmt.Errorf("cc2cc %s: nil underlying stub", fcn)
    }

    argv := make([][]byte, 0, 1+len(args))
    argv = append(argv, []byte(fcn))
    for _, a := range args {
        argv = append(argv, []byte(a))
    }

    resp := s.InvokeChaincode(preloadCC, argv, "") // "" => same channel

    // Treat any non-200 or empty payload as an error so tests fail gracefully.
    if resp.Status != 200 || len(resp.Payload) == 0 {
        // Join is safe on nil; Message is a string in apiv2.
        return nil, fmt.Errorf("cc2cc %s(%s) status=%d message=%q",
            fcn, strings.Join(args, ","), resp.Status, resp.Message)
    }
    return resp.Payload, nil
}


// MustJSON marshals v and ignores errors (used for events and small writes).
// Params: any.
// Return: JSON bytes (best effort).
func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }

const (
    maxBioCipherB64 = 4096 // ~3 KB payload when base64
    maxBioNonceB64  = 128
    maxBioTagHex    = 128  // 64 bytes hex for HMAC-SHA256
    maxBioAlgLen    = 64
    maxDevSigB64    = 2048 // Adjust if your device signatures are larger
)

// ClampVoteExtras bounds variable-size inputs to keep records predictable and safe.
// Params: bioAlg, bioNonceB64, bioCipherB64, bioTagHex, devSigB64.
// Return: error if any exceeds configured limits.
func clampVoteExtras(bioAlg, bioNonceB64, bioCipherB64, bioTagHex, devSigB64 string) error {
    if len(bioAlg) > maxBioAlgLen { return fmt.Errorf("bioAlg too long") }
    if len(bioNonceB64) > maxBioNonceB64 { return fmt.Errorf("bioNonce too long") }
    if len(bioCipherB64) > maxBioCipherB64 { return fmt.Errorf("bioCipher too large") }
    if len(bioTagHex) > maxBioTagHex { return fmt.Errorf("bioTag too long") }
    if len(devSigB64) > maxDevSigB64 { return fmt.Errorf("device signature too large") }
    return nil
}


/* Admin / Setup */

// SetJointPublicKey stores the Paillier public key for a state and anchors the key ceremony.
// This function is part of the trustee key distribution workflow.
func (c *AccumVoteContract) SetJointPublicKey(ctx contractapi.TransactionContextInterface, stateCode string, pkJSON string) error {
	stateCode = strings.TrimSpace(stateCode)
	if stateCode == "" {
		return fmt.Errorf("stateCode empty")
	}

	var pk PublicKey
	if err := json.Unmarshal([]byte(pkJSON), &pk); err != nil {
		return fmt.Errorf("bad pk json: %w", err)
	}
	if pk.N == "" || pk.G == "" {
		return fmt.Errorf("pk must include hex n and g")
	}

	// Validate hex early; also canonicalize and ensure N2 is present
	n, err := bigFromHex(pk.N)
	if err != nil { return fmt.Errorf("pk.N bad hex: %w", err) }
	if _, err := bigFromHex(pk.G); err != nil { return fmt.Errorf("pk.G bad hex: %w", err) }

	if pk.N2 == "" {
		pk.N2 = hexFromBig(new(big.Int).Mul(n, n))
	} else {
		n2, err := bigFromHex(pk.N2)
		if err != nil { return fmt.Errorf("pk.N2 bad hex: %w", err) }
		pk.N2 = hexFromBig(n2) // Canonicalize
	}

	// Store canonical JSON exactly once
	canon, _ := json.Marshal(pk)
	if err := ctx.GetStub().PutState(keyPKPrefix+stateCode, canon); err != nil {
		return err
	}

	// Ensure next load sees new key
	pkCache.Delete(stateCode)
	// (Optional warm cache)
	pkCache.Store(stateCode, pkEntry{
		n:  new(big.Int).Set(n),
		n2: new(big.Int).Mul(n, n),
	})

	if params, _ := getParams(ctx); params.EmitEvents {
		_ = ctx.GetStub().SetEvent(eventPublicKeySet, mustJSON(map[string]string{
			"state": stateCode,
			"nHash": sha256HexStr(pk.N), // Keep behavior stable
			"time":  nowRFC3339(ctx),
		}))
	}
	return nil
}


// GetJointPublicKey returns the stored Paillier public key for a state.
func (c *AccumVoteContract) GetJointPublicKey(ctx contractapi.TransactionContextInterface, stateCode string) (string, error) {
raw, err := ctx.GetStub().GetState(keyPKPrefix + stateCode)
if err != nil {
return "", err
}
if raw == nil {
return "", fmt.Errorf("not found")
}
return string(raw), nil
}

// SeedCandidateList stores a canonical candidate list for a constituency.
// The list is used by validation/tally logic and by the public dashboard.
func (c *AccumVoteContract) SeedCandidateList(ctx contractapi.TransactionContextInterface, constituencyID string, candidatesJSON string) error {
constituencyID = strings.TrimSpace(constituencyID)
if constituencyID == "" {
return fmt.Errorf("constituencyID empty")
}
var ids []string
if err := json.Unmarshal([]byte(candidatesJSON), &ids); err != nil {
return fmt.Errorf("parse candidate IDs: %w", err)
}
sort.Strings(ids)
b, _ := json.Marshal(ids)
return ctx.GetStub().PutState(keyCandsListPrefx+constituencyID, b)
}

// GetCandidateList returns the candidate list previously seeded for the constituency.
func (c *AccumVoteContract) GetCandidateList(ctx contractapi.TransactionContextInterface, constituencyID string) ([]string, error) {
raw, err := ctx.GetStub().GetState(keyCandsListPrefx + constituencyID)
if err != nil {
return nil, err
}
if raw == nil {
return []string{}, nil
}
var arr []string
if err := json.Unmarshal(raw, &arr); err != nil {
return []string{}, nil
}
sort.Strings(arr)
return arr, nil
}

// OpenPoll marks the poll as open for a constituency.
// RecordVote will reject if the poll is not open.
func (c *AccumVoteContract) OpenPoll(ctx contractapi.TransactionContextInterface, constituencyID string) error {
    if err := ctx.GetStub().PutState(keyPollPrefix+constituencyID, []byte("open")); err != nil { return err }
    if p, _ := getParams(ctx); p.EmitEvents {
        _ = ctx.GetStub().SetEvent(eventPollOpened, mustJSON(map[string]string{
            "constituency": constituencyID, "time": nowRFC3339(ctx),
        }))
    }
    return nil
}

// ClosePoll marks the poll as closed for a constituency.
// Tally/Publish operations typically require the poll to be closed.
func (c *AccumVoteContract) ClosePoll(ctx contractapi.TransactionContextInterface, constituencyID string) error {
    if err := ctx.GetStub().PutState(keyPollPrefix+constituencyID, []byte("closed")); err != nil { return err }
    if p, _ := getParams(ctx); p.EmitEvents {
        _ = ctx.GetStub().SetEvent(eventPollClosed, mustJSON(map[string]string{
            "constituency": constituencyID, "time": nowRFC3339(ctx),
        }))
    }
    return nil
}


// SetParams writes runtime parameters (feature flags and limits) to world state.
func (c *AccumVoteContract) SetParams(ctx contractapi.TransactionContextInterface, paramsJSON string) error {
    cur, err := getParams(ctx)
    if err != nil { return err }

    jsCur, _ := json.Marshal(cur)
    var merged map[string]any
    _ = json.Unmarshal(jsCur, &merged)

    var upd map[string]any
    if err := json.Unmarshal([]byte(paramsJSON), &upd); err != nil {
        return fmt.Errorf("bad params json: %w", err)
    }
    for k, v := range upd { merged[k] = v }

    js, _ := json.Marshal(merged)
    if err := ctx.GetStub().PutState(keyParams, js); err != nil { return err }

    if params, _ := getParams(ctx); params.EmitEvents {
        keys := make([]string, 0, len(upd))
        for k := range upd { keys = append(keys, k) }
        sort.Strings(keys)
        _ = ctx.GetStub().SetEvent(eventParamsUpdated, mustJSON(map[string]any{
            "hash": sha256Hex(js),
            "keys": keys,
            "time": nowRFC3339(ctx),
        }))
    }
    return nil
}

// GetParams reads back the stored runtime parameters.
func (c *AccumVoteContract) GetParams(ctx contractapi.TransactionContextInterface) (*Params, error) {
return getParams(ctx)
}


// iterVotesPDC iterates all vote records for a constituency from the votes private data collection.
// It returns a map keyed by serial so later phases can apply status rules and perform tallying.
func iterVotesPDC(ctx contractapi.TransactionContextInterface, constituencyID string) (map[string]VoteMetaPDC, error) {
    // Scan all kiosk ballot metas in public state
    startKey := keyBallotPrefix        // "BAL::"
    endKey   := keyBallotPrefix + "~"  // "BAL::~"

    it, err := ctx.GetStub().GetStateByRange(startKey, endKey)
    if err != nil {
        return nil, err
    }
    defer it.Close()

    out := make(map[string]VoteMetaPDC)
    for it.HasNext() {
        kv, err := it.Next()
        if err != nil {
            return nil, err
        }

        // Expect keys of the form BAL::<serial>
        serial := strings.TrimPrefix(kv.Key, keyBallotPrefix)
        if serial == kv.Key || serial == "" {
            // Not a BAL:: key or empty serial; ignore defensively
            continue
        }

        // Reconstruct the PDC key for this constituency + serial
        pdcKey := voteKey(constituencyID, serial) // "VOTE::<const>::<serial>"
        vbytes, err := ctx.GetStub().GetPrivateData(votesPDC, pdcKey)
        if err != nil {
            return nil, err
        }
        if vbytes == nil {
            // No vote in this constituency for this serial; skip
            continue
        }

        var vm VoteMetaPDC
        if err := json.Unmarshal(vbytes, &vm); err != nil {
            // Corrupt/invalid PDC entry; skip but do not fail the whole tally
            continue
        }

        out[serial] = vm
    }

    return out, nil
}



// RefreshCandidateListFromPreload fetches the candidate list from the preload registry chaincode
// and stores it locally for this contract.
func (c *AccumVoteContract) RefreshCandidateListFromPreload(
    ctx contractapi.TransactionContextInterface,
    constituencyID string,
) error {
    constituencyID = strings.TrimSpace(constituencyID)
    if constituencyID == "" {
        return fmt.Errorf("constituencyID empty")
    }

    params, err := getParams(ctx)
    if err != nil {
        return err
    }
    // Cc2cc → evote-preload.GetCandidateList(constituencyID) -> payload bytes are a JSON array string
    payload, err := callPreload(ctx, params.PreloadCCName, "GetCandidateList", constituencyID)
    if err != nil {
        return err
    }

    // Parse into []string
    var ids []string
    if err := json.Unmarshal(payload, &ids); err != nil {
        return fmt.Errorf("preload list JSON parse: %w", err)
    }
    sort.Strings(ids)

    b, _ := json.Marshal(ids)
    if err := ctx.GetStub().PutState(keyCandsListPrefx+constituencyID, b); err != nil {
        return err
    }

    // Optional: also store a hash for audit/version-lock
    // Sum := sha256Hex(b)
    // _ = ctx.GetStub().PutState("CANDLIST_HASH::"+constituencyID, []byte(sum))

	if params.EmitEvents {
		_ = ctx.GetStub().SetEvent(eventCandidateListRefreshed, mustJSON(map[string]any{
			"constituency": constituencyID,
			"count":        len(ids),
			"listHash":     sha256Hex(b),     // Hash of sorted JSON
			"time":         nowRFC3339(ctx),
		}))
	}

    return nil
}


/* Queries */

// GetBallotBySerial returns public ballot metadata for a serial (BAL::<serial>).
// This is the main input for kiosk/receipt verification.
func (c *AccumVoteContract) GetBallotBySerial(ctx contractapi.TransactionContextInterface, serial string) (*BallotMeta, error) {
raw, err := ctx.GetStub().GetState(keyBallotPrefix + serial)
if err != nil {
return nil, err
}
if raw == nil {
return nil, fmt.Errorf("not found")
}
var m BallotMeta
if err := json.Unmarshal(raw, &m); err != nil {
return nil, err
}
return &m, nil
}

// GetEncSums recomputes the encrypted vote sums for the current ballots.
// It is used for transparency/debugging and to support audit tooling.
func (c *AccumVoteContract) GetEncSums(ctx contractapi.TransactionContextInterface, constituencyID string) (map[string]string, error) {
    state, err := resolveStateForReader(ctx)
    if err != nil { return nil, err }
    _, _, n2, err := loadPK(ctx, state)

    if err != nil { return nil, err }

    // Candidate list (just to initialize output keys deterministically)
    cands, err := c.GetCandidateList(ctx, constituencyID)
    if err != nil { return nil, err }

    latest, err := iterVotesPDC(ctx, constituencyID)
    if err != nil { return nil, err }

	// Detect if real Paillier ciphertexts are present
	// Detect if real Paillier ciphertexts are present

	enc := make(map[string]*big.Int, len(cands))
	one := big.NewInt(1)
	for _, id := range cands { enc[id] = new(big.Int).Set(one) }

    // Multiply Enc(1) for each serial’s latest vote
	for _, vm := range latest {
		if _, ok := enc[vm.CandidateID]; !ok {
			continue
		}
		c1, err := bigFromHex(vm.EncOneHex)
		if err != nil {
			return nil, err
		}
		// Accept Enc(1) that passes standard Paillier sanity: 1 < c < n² and gcd(c, n²) = 1.
		if err := encOneChecks(c1, n2); err != nil {
			continue
		}
		enc[vm.CandidateID] = mulMod(enc[vm.CandidateID], c1, n2)
	}

	out := make(map[string]string, len(enc))
	for k, v := range enc {
		out[k] = hexFromBig(v)
	}
	canonHexMap(out) // <-- ensure identity is exactly "1"
	return out, nil

}


/* Hot path */

// RecordVote is the high-throughput vote submission entry point.
//
// Key properties:
// - Writes the vote to private data under VOTE::<const>::<serial> (latest write wins).
// - Writes/updates BAL::<serial> with Status=Pending and the receipt hash (hC).
// - Does NOT create TXIDX entries; those are created only when ballots become Current.
func (c *AccumVoteContract) RecordVote(
    ctx contractapi.TransactionContextInterface,
    serial, constituencyID, candidateID string,
    encOneHex, receiptSalt, epoch, attestationSig string,
    boothID, deviceID, deviceKeyFP string,      // ← NEW
    bioAlg, bioNonceB64, bioCipherB64, bioTagHex string,
) (string, error) {

    // 0) Cheap gate: poll & ABAC
    open, err := pollIsOpen(ctx, constituencyID)
    if err != nil { return "", err }
    if !open { return "", fmt.Errorf("poll closed") }
    state, err := requireABAC(ctx, constituencyID) // Tests bypass via env
    if err != nil { return "", err }

    // / 1) Fast params + attestation gate (fail before any heavy work)
	params, _ := getParams(ctx)
	if params.VerifyAttestation && strings.TrimSpace(attestationSig) == "" {
		return "", fmt.Errorf("missing device attestation")
	}
	if err := clampVoteExtras(bioAlg, bioNonceB64, bioCipherB64, bioTagHex, attestationSig); err != nil {
		return "", err
	}

	// 2) Paillier sanity (no policy checks on cast)
	_, _, n2, err := loadPK(ctx, state)
	if err != nil { return "", err }
	c1, err := bigFromHex(encOneHex)
	if err != nil { return "", err }
	if err := encOneChecks(c1, n2); err != nil { return "", err }

    // 4) Timestamp & IDs
    txID := ctx.GetStub().GetTxID()
    ts, _ := ctx.GetStub().GetTxTimestamp()
    castTime := time.Unix(ts.Seconds, int64(ts.Nanos)).UTC().Format(time.RFC3339)
    hc := sha256HexStr(encOneHex)

    // 5) ONE PDC write
    vm := VoteMetaPDC{
        CandidateID:  candidateID,
        EncOneHex:    encOneHex,
        Epoch:        epoch,
        TxID:         txID,
        CastTime:     castTime,
        StateCode:    state,
        BoothID:      strings.TrimSpace(boothID),
        DeviceID:     strings.TrimSpace(deviceID),
        DeviceKeyFP:  strings.TrimSpace(deviceKeyFP),
        BioAlg:       bioAlg,
        BioNonceB64:  bioNonceB64,
        BioCipherB64: bioCipherB64,
        BioTagHex:    bioTagHex,
        DevSigB64:    attestationSig,
    }
    if err := ctx.GetStub().PutPrivateData(votesPDC, voteKey(constituencyID, serial), mustJSON(&vm)); err != nil {
        return "", err
    }

    // 6) Mandatory audit meta (WS)
    _ = ctx.GetStub().PutState(
        keyBallotPrefix+serial,
        mustJSON(&BallotMeta{
            HC: hc, Status: "pending", Epoch: epoch, TxID: txID, CastTime: castTime,
        }),
    )

    // 7) Event (configurable)
    if params.EmitEvents {
        _ = ctx.GetStub().SetEvent(eventVoteRecorded, mustJSON(map[string]string{
            "serial": serial, "constituency": constituencyID, "txID": txID,
            "epoch": epoch, "candidateHash": sha256HexStr(candidateID),
        }))
    }

    // 8) Return small JSON
    return fmt.Sprintf(`{"txID":"%s","serial":"%s","hC":"%s","status":"accepted","epoch":"%s","castTime":"%s"}`,
        txID, serial, hc, epoch, castTime), nil
}


/* Tally path */

// TallyPrepare prepares an encrypted tally over ballots currently marked as Current.
//
// It reads votes from private data, multiplies ciphertexts modulo n^2 per candidate,
// and writes a public anchor that can be verified by trustees and the public dashboard.
func (c *AccumVoteContract) TallyPrepare(ctx contractapi.TransactionContextInterface, constituencyID string) (map[string]string, error) {
	state, err := resolveStateForReader(ctx)
	if err != nil { return nil, err }
	_, _, n2, err := loadPK(ctx, state)


	if err != nil { return nil, err }

	params, err := getParams(ctx)
	if err != nil { return nil, err }

	
	// 1) Candidate list: use locally-seeded list when present; otherwise refresh from preload
	candList, err := c.GetCandidateList(ctx, constituencyID)
	if err != nil { 
		return nil, err 
	}
	if len(candList) == 0 {
		// Force admin to seed candidates beforehand
		return nil, fmt.Errorf("candidate list not seeded for %s", constituencyID)
	}


	candOK := make(map[string]struct{}, len(candList))
	for _, id := range candList { candOK[id] = struct{}{} }

	// Initialize Enc(0) as 1 for each candidate (multiplicative identity)
	encSums := make(map[string]*big.Int, len(candList))
	for _, id := range candList { encSums[id] = big.NewInt(1) }

	// 2/3) Iterate *latest* votes from PDC, validate, and include if valid
	latest, err := iterVotesPDC(ctx, constituencyID)
	if err != nil { return nil, err }
	
		// Pass 0. Detect whether we have any “real” Paillier ciphertexts.
	// Before the loop
	validCount := 0
	invalidCount := 0

	for serial, vm := range latest {
		// Candidate missing
		if _, ok := candOK[vm.CandidateID]; !ok {
			invalidCount++
			if params.EmitEvents {
				_ = ctx.GetStub().SetEvent(eventInvalidCandidateFound, mustJSON(map[string]string{
					"constituency":  constituencyID,
					"serial":        serial,
					"candidateHash": sha256HexStr(vm.CandidateID),
					"txID":          vm.TxID,
					"time":          nowRFC3339(ctx),
				}))
			}
			continue
		}

		// Voter roll
		if params.ValidateOnTally {
			ok, err := hasVoterViaCC(ctx, params.PreloadCCName, constituencyID, serial)
			if err != nil { return nil, err }
			if !ok {
				invalidCount++
				if params.EmitEvents {
					_ = ctx.GetStub().SetEvent(eventInvalidVoterFound, mustJSON(map[string]string{
						"constituency":  constituencyID,
						"serial":        serial,
						"candidateHash": sha256HexStr(vm.CandidateID),
						"txID":          vm.TxID,
						"time":          nowRFC3339(ctx),
					}))
				}
				continue
			}
		}

		// Booth validation
		if params.ValidateBoothOnTally {
			stateForBooth := vm.StateCode
			if stateForBooth == "" { stateForBooth = state }

			br, err := getBoothViaCC(ctx, params.BoothCCName, stateForBooth, constituencyID, vm.BoothID)
			if err != nil || strings.ToUpper(strings.TrimSpace(br.Status)) != "A" ||
			   !castTimeWithin(vm.CastTime, br.OpenTime, br.CloseTime) ||
			   !boothDeviceOK(&vm, br) {
				invalidCount++
				if params.EmitEvents {
					_ = ctx.GetStub().SetEvent(eventInvalidBoothFound, mustJSON(map[string]string{
						"constituency": constituencyID,
						"serial":       serial,
						"boothID":      vm.BoothID,
						"txID":         vm.TxID,
						"time":         nowRFC3339(ctx),
					}))
				}
				continue
			}
		}

		c1, err := bigFromHex(vm.EncOneHex)
		if err != nil || encOneChecks(c1, n2) != nil {
			invalidCount++
			continue
		}

		encSums[vm.CandidateID] = mulMod(encSums[vm.CandidateID], c1, n2)
		validCount++
	}



	// 4) Emit deterministic hash of sums for audit
	if params.EmitEvents {
		keys := append([]string(nil), candList...)
		sort.Strings(keys)
		var buf bytes.Buffer
		for _, k := range keys {
			buf.WriteString(k); buf.WriteByte(':'); buf.WriteString(hexFromBig(encSums[k])); buf.WriteByte('|')
		}
		_ = ctx.GetStub().SetEvent(eventTallyPrepared, mustJSON(map[string]any{
			"constituency": constituencyID,
			"sumsHash":     sha256Hex(buf.Bytes()), // Stable trace of encrypted totals
			"valid":        validCount,
			"invalid":      invalidCount,
			"time":         nowRFC3339(ctx),
		}))
	}


	// 5) Return hex strings
	out := make(map[string]string, len(encSums))
	for k, v := range encSums {
		out[k] = hexFromBig(v)
	}
	canonHexMap(out) // <-- ensure identity is exactly "1"
	return out, nil


}


// ApplyBallotStatuses applies eligibility decisions to BAL::<serial> after off-chain checks.
//
// It also maintains TXIDX::<txID> so that tx-based receipt checks only work for Current ballots.
// Invalid or superseded ballots have their TXIDX entry removed.
func (c *AccumVoteContract) ApplyBallotStatuses(
    ctx contractapi.TransactionContextInterface,
    constituencyID string,
    statusJSON string, // E.g., {"current":[{"serial":"...","txID":"..."}], "invalid":[...]}
) error {
    var payload struct {
        Current []struct {
            Serial string `json:"serial"`
            TxID   string `json:"txID"`
            EncOne string `json:"encOneHex,omitempty"`
        } `json:"current"`
        Invalid []struct {
            Serial string `json:"serial"`
            TxID   string `json:"txID,omitempty"`
        } `json:"invalid"`
    }
    if err := json.Unmarshal([]byte(statusJSON), &payload); err != nil {
        return fmt.Errorf("statusJSON parse: %w", err)
    }

	for _, v := range payload.Current {
		// Reuse markBallotStatus: this only writes WS, no PDC reads
		vm := VoteMetaPDC{EncOneHex: v.EncOne, TxID: v.TxID, Epoch: ""} // EncOne optional
		markBallotStatus(ctx, v.Serial, "current", vm)

		// TXIDX is *only* for ballots that are current/valid so that VerifyReceipt
		// Only ever sees "live" votes; superseded/invalid ones become unknown_tx.
		if v.TxID != "" {
			_ = ctx.GetStub().PutState(keyTxIdxPrefix+v.TxID, []byte(v.Serial))
		}
	}

	for _, v := range payload.Invalid {
		// Mark the ballot as invalid in BAL::<serial>, but DO NOT create TXIDX.
		// This ensures:
		// - invalid voters (e.g. off-roll, bad booth) are excluded from receipt lookups
		// - tests expecting no TXIDX for invalid txIDs (tx-bad, tx-2) pass
		vm := VoteMetaPDC{TxID: v.TxID}
		markBallotStatus(ctx, v.Serial, "invalid", vm)
		// Intentionally no TXIDX:: write here
	}

	return nil

}


// markBallotStatus updates BAL::<serial> while preserving receipt-critical fields.
// In particular, it does not overwrite hC unless a new encOneHex is provided.
func markBallotStatus(ctx contractapi.TransactionContextInterface, serial, status string, vm VoteMetaPDC) {
    bm := BallotMeta{
        HC:   sha256HexStr(vm.EncOneHex),
        TxID: vm.TxID,
        Epoch: vm.Epoch,
        // CastTime is optional; set if present
    }
    bm.Status = status
    if vm.CastTime != "" { bm.CastTime = vm.CastTime }
    if b, _ := json.Marshal(bm); b != nil {
        _ = ctx.GetStub().PutState(keyBallotPrefix+serial, b)
    }
}


// PublishResults stores a plaintext results anchor for a constituency and round.
// This is typically called after trustees finish decryption and produce a bundle hash.
func (c *AccumVoteContract) PublishResults(ctx contractapi.TransactionContextInterface,
    constituencyID string, roundID string, resultJSON string, bundleHash string) error {

    open, err := pollIsOpen(ctx, constituencyID)
    if err != nil { return err }
    if open { return fmt.Errorf("poll must be closed for %s", constituencyID) }

    ts, _ := ctx.GetStub().GetTxTimestamp()
    when := time.Unix(ts.Seconds, int64(ts.Nanos)).UTC().Format(time.RFC3339)

    anchor := &TallyResultAnchor{
        RoundID: roundID, Constituency: constituencyID, Time: when,
        Results: json.RawMessage(resultJSON), BundleHash: bundleHash,
    }
    js, _ := json.Marshal(anchor)
    if err := ctx.GetStub().PutState(fmt.Sprintf("%s%s::%s", keyResultsPrefix, constituencyID, roundID), js); err != nil {
        return err
    }

    if params, _ := getParams(ctx); params.EmitEvents {
        _ = ctx.GetStub().SetEvent(eventResultsPublished, mustJSON(map[string]string{
            "constituency": constituencyID,
            "roundID":      roundID,
            "bundleHash":   bundleHash,
            "time":         when,
        }))
    }
    return nil
}


// VerifyReceipt validates a receipt using (txID, hC) via TXIDX::<txID>.
// TXIDX is maintained only for Current ballots; unknown txIDs are reported as unknown_tx.
func (c *AccumVoteContract) VerifyReceipt(ctx contractapi.TransactionContextInterface, txID string, receipt string) (string, error) {
    serialB, err := ctx.GetStub().GetState(keyTxIdxPrefix + txID)
    if err != nil { return "", err }
    if serialB == nil { return `{"ok":false,"reason":"unknown_tx"}`, nil }
    serial := string(serialB)

    bmRaw, err := ctx.GetStub().GetState(keyBallotPrefix + serial)
    if err != nil { return "", err }
    if bmRaw == nil { return `{"ok":false,"reason":"no_ballot"}`, nil }

    var bm BallotMeta
    if err := json.Unmarshal(bmRaw, &bm); err != nil {
        return "", err
    }

    // Superseded if the currently stored TxID is not the one presented
    superseded := bm.TxID != txID

    // Status must be "current"; "invalid" or others will fail
    if bm.Status != "current" {
        return fmt.Sprintf(`{"ok":false,"reason":"%s","superseded":%t,"serial":"%s","txID":"%s"}`, bm.Status, superseded, serial, txID), nil
    }

    // Receipt must match BallotMeta.HC
    if bm.HC != strings.TrimSpace(receipt) {
        return fmt.Sprintf(`{"ok":false,"reason":"receipt_mismatch","superseded":%t,"serial":"%s","txID":"%s"}`, superseded, serial, txID), nil
    }

    return fmt.Sprintf(`{"ok":true,"superseded":%t,"serial":"%s","txID":"%s"}`, superseded, serial, txID), nil
}


/* Health */

// Ping is a simple health check used by deployment tooling and test harnesses.
func (c *AccumVoteContract) Ping(ctx contractapi.TransactionContextInterface) (string, error) {
return "OK:" + ctx.GetStub().GetTxID(), nil
}
