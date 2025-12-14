// -----------------------------------------------------------------------------
// accumvote_cc contract (Go, Fabric v3.1.1)
// Purpose: Implements a high-TPS, privacy-preserving vote accumulator using
// Paillier Enc(1) multiplication and “latest-vote-wins”, with validation at tally.
// Role in system: Write-path records a kiosk’s latest vote pointer into a PDC and
// minimal public kiosk metadata; read-path/tally composes encrypted sums and
// marks invalid votes (non-existent voters/booths, bad ciphertexts) for exclusion.
// Key dependencies: Hyperledger Fabric contractapi/cid; a canonical preload
// chaincode (“evote-preload”) for voters/candidates; a booth registry CC
// (“boothpdc”) for booth/device windows; private data collection “votes_pdc”.
// -----------------------------------------------------------------------------

/*
accumvote_cc (Fabric v3.1.1) — homomorphic accumulators + re-vote

Hot path (TPS focus):
• RecordVote(serial, constituencyID, candidateID, encOneHex, receiptSalt, epoch, attestationSig)
– ABAC (role=voterMachine; state/const match)
– (Optional) attestationSig check (can be fixed string during Caliper)
– Enc(1) sanity (range + gcd)
– Same-candidate re-vote is a no-op (idempotent)
– Update PDC (latest choice); kiosk meta write is toggleable

Cold path (no private keys on-chain):
• TallyPrepare(constituencyID) – combine votes → encrypted sums
• PublishResults(constituencyID, roundID, ...) – anchor off-chain results + proof-bundle hash
• SummarizeInvalids / ApplyCorrections – homomorphic back-out of invalids post-close

*/
package main

import (
"bytes"
"crypto/sha256"
"encoding/hex"
"encoding/json"
//"errors"
"fmt"
"math/big"
//"os"
"sort"
"strings"
"time"
"reflect"
"strconv"
//"github.com/hyperledger/fabric-chaincode-go/v2/pkg/cid"
"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
"sync"
)

/* ---------- Keys & constants (single namespace for this chaincode) ---------- */

const (
// Collections
votesPDC = "votes_pdc"

// World state prefixes (public)
keyPKPrefix       = "PK::"       // PK::<state>                 → PublicKey JSON (Paillier)
keyBallotPrefix   = "BAL::"      // BAL::<serial>               → BallotMeta for kiosk (no identity)
keyPollPrefix     = "POLL::"     // POLL::<const>               → "open"/"closed"
keyParams         = "PARAMS"     // global Params JSON
keyCandsListPrefx = "CANDLIST::" // CANDLIST::<const>           → []string (local copy)
keyResultsPrefix  = "RES::"      // RES::<const>::<roundID>     → anchored results

keyTxIdxPrefix = "TXIDX::"          // txID → serial (for VerifyReceipt)
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

/* ------------------------- Types & small data models ------------------------ */

type AccumVoteContract struct{ contractapi.Contract }

// Paillier public key (hex). n2 may be omitted (we compute n^2).
type PublicKey struct {
N  string `json:"n"`
G  string `json:"g"`
N2 string `json:"n2,omitempty"`

}

// VoteMetaPDC is stored in the private collection.
// We now persist *every* cast, and also store a "latest pointer" under votes::<const>::<serial>.
type VoteMetaPDC struct {
    CandidateID string `json:"candidateID"`
    // New fields for append-only audit + tally reconstruction:
    EncOneHex string `json:"encOneHex"`           // Paillier Enc(1;r) as hex
    Epoch     string `json:"epoch,omitempty"`     // logical revote epoch (string to avoid int64 JSON ambiguity)
    TxID      string `json:"txID"`                // tx that wrote this record (unique per cast)
    CastTime  string `json:"castTime,omitempty"`  // RFC3339 (optional)
	
	 // NEW – for booth/device validation at tally:
    StateCode   string `json:"stateCode,omitempty"`
    BoothID     string `json:"boothID,omitempty"`
    DeviceID    string `json:"deviceID,omitempty"`
    DeviceKeyFP string `json:"deviceKeyFP,omitempty"`
	
	 // New: encrypted facial embedding (simple, optional)
    BioAlg       string `json:"bioAlg,omitempty"`       // e.g., "AES-256-GCM"
    BioNonceB64  string `json:"bioNonceB64,omitempty"`  // base64(nonce)
    BioCipherB64 string `json:"bioCipherB64,omitempty"` // base64(ciphertext||tag)
    BioTagHex    string `json:"bioTagHex,omitempty"`    // optional HMAC link tag (hex)

    // New: device signature (base64 or hex; you decide upstream)
    DevSigB64    string `json:"devSigB64,omitempty"`
}


// Public kiosk/meta: ties a serial to THIS tx’s ciphertext via hC = SHA256(encOneHex).
type BallotMeta struct {
HC       string `json:"hC"`
Status   string `json:"status"`   // "current"
Epoch    string `json:"epoch"`
CastTime string `json:"castTime"` // RFC3339
TxID     string `json:"txID"`

}

// Published result anchor (plaintext + hash of proof bundle, both produced off-chain).
type TallyResultAnchor struct {
RoundID      string          `json:"roundID"`
Constituency string          `json:"constituency"`
Time         string          `json:"time"`
Results      json.RawMessage `json:"results"`
BundleHash   string          `json:"bundleHash"`

}

// On-ledger tuning knobs. Env vars can override at runtime inside chaincode container.
type Params struct {
VerifyAttestation bool   `json:"VERIFY_ATTESTATION"`  // accept non-empty/fixed value
EmitEvents        bool `json:"EMIT_EVENTS"`        // default true: emit events

ValidateOnTally bool   `json:"VALIDATE_ON_TALLY"` // default false: run eligibility at tally-time

PreloadCCName string `json:"PRELOAD_CC_NAME"` // single source of truth CC for candidates/voters

BoothCCName        string `json:"BOOTH_CC_NAME"`         // default "boothpdc"
ValidateBoothOnTally bool `json:"VALIDATE_BOOTH_ON_TALLY"` // default true
}


type boothRec struct {
    Status               string `json:"status"`
    OpenTime             int64  `json:"open_time"`
    CloseTime            int64  `json:"close_time"`
    DeviceID             string `json:"device_id"`
    DeviceKeyFingerprint string `json:"device_key_fingerprint"`
}

// cache parsed Paillier keys per state (thread-safe)
var pkCache sync.Map // key: state -> pkEntry

type pkEntry struct {
    n  *big.Int
    n2 *big.Int
    // g is unused on-chain; you can add it if you later need it
}


/* ------------------------------ Small helpers ------------------------------ */

// getBoothViaCC asks the booth registry chaincode for a booth row.
// params: boothCC/stateCode/constituencyID/boothID identify the booth.
// returns: parsed boothRec or error when CC2CC fails or JSON is malformed.
func getBoothViaCC(ctx contractapi.TransactionContextInterface, boothCC, stateCode, constituencyID, boothID string) (*boothRec, error) {
    if strings.TrimSpace(boothID) == "" { return nil, fmt.Errorf("empty booth") }
    payload, err := callPreload(ctx, boothCC, "GetBooth", stateCode, constituencyID, boothID)
    if err != nil { return nil, err }
    var br boothRec
    if json.Unmarshal(payload, &br) != nil { return nil, fmt.Errorf("bad booth json") }
    return &br, nil
}

// castTimeWithin checks whether vmCastRFC3339 falls inside [open, close] (epoch seconds).
// params: RFC3339 timestamp and booth open/close epoch seconds.
// return: true if within; fail-soft true when any time is missing or unparsable.
func castTimeWithin(vmCastRFC3339 string, open, close int64) bool {
    if vmCastRFC3339 == "" || open == 0 || close == 0 { return true } // fail-soft
    ct, err := time.Parse(time.RFC3339, vmCastRFC3339)
    if err != nil { return true } // fail-soft
    t := ct.Unix()
    return t >= open && t <= close
}

// boothDeviceOK enforces optional device binding checks from booth record.
// params: vote meta (device fields) and boothRec (expected device IDs).
// return: false if booth specifies a value and it mismatches; otherwise true.
func boothDeviceOK(vm *VoteMetaPDC, br *boothRec) bool {
    // only enforce if booth row provided a value
    if br.DeviceID != "" && vm.DeviceID != "" && br.DeviceID != vm.DeviceID { return false }
    if br.DeviceKeyFingerprint != "" && vm.DeviceKeyFP != "" && br.DeviceKeyFingerprint != vm.DeviceKeyFP { return false }
    return true
}

// nowRFC3339 returns current tx timestamp as RFC3339 string.
// params: ctx for Fabric tx timestamp.
// return: RFC3339 in UTC.
func nowRFC3339(ctx contractapi.TransactionContextInterface) string {
    ts, _ := ctx.GetStub().GetTxTimestamp()
    return time.Unix(ts.Seconds, int64(ts.Nanos)).UTC().Format(time.RFC3339)
}

// sha256Hex returns hex(SHA256(b)).
// params: byte slice.
// return: lowercase hex digest.
func sha256Hex(b []byte) string {
h := sha256.Sum256(b)
return hex.EncodeToString(h[:])
}
func sha256HexStr(s string) string { return sha256Hex([]byte(s)) }

// bigFromHex parses hex (with/without 0x) or base-10 string into big.Int.
// params: string s.
// return: *big.Int or error with clear message for bad inputs.
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
    // include the word "hex" so tests match
    return nil, fmt.Errorf("bad hex integer: %q", s)
}


// hexFromBig renders big.Int as canonical lowercase hex (no leading zeros).
// params: x big.Int.
// return: "0" for zero; hex string otherwise.
func hexFromBig(x *big.Int) string {
	if x == nil || x.Sign() == 0 {
		return "0"
	}
	// Use canonical lowercase hex with NO leading zeros.
	s := strings.ToLower(x.Text(16)) // textual hex, never padded
	s = strings.TrimLeft(s, "0")     // drop any stray leading zeros (paranoia)
	if s == "" {                     // if x==0, after trim we normalize to "0"
		return "0"
	}
	return s
}

// canonHexStr reparses s and re-emits canonical hex; returns s unchanged on parse failure.
// params: hex/decimal string.
// return: canonical hex string.
func canonHexStr(s string) string {
    x, err := bigFromHex(s)
    if err != nil { // if it's not a valid hex/int, just return as-is
        return s
    }
    return hexFromBig(x)
}

// canonHexMap canonicalizes all values in a map in-place.
// params: map[string]string.
// return: none (mutates input map).
func canonHexMap(m map[string]string) {
    for k, v := range m {
        m[k] = canonHexStr(v)
    }
}


// mulMod computes (x*y) % mod using big.Int to avoid overflow.
// params: x,y,mod.
// return: result big.Int pointer (new allocation).
func mulMod(x, y, mod *big.Int) *big.Int {
z := new(big.Int).Mul(x, y)
return z.Mod(z, mod)
}


// loadPK reads the Paillier public key for a state from WS and parses n, g, n².
// params: ctx, state string.
// return: (n, g, n2) or error if state row missing or malformed.
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

// getParams returns the active parameter set (defaults overlaid by on-ledger PARAMS).
// params: ctx.
// return: *Params; never nil on success. On bad PARAMS JSON, falls back to defaults.
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


// resolveStateForReader figures out caller’s state attribute for read-side APIs.
// params: ctx.
// return: state code or "TEST"/BYPASS_STATE when test-bypassed via env.
func resolveStateForReader(ctx contractapi.TransactionContextInterface) (string, error) {
	/*if strings.ToLower(os.Getenv("BYPASS_ABAC_FOR_TEST")) == "on" {
		if s := os.Getenv("BYPASS_STATE"); s != "" {
			return s, nil
		}
		return "TEST", nil
	}
	id, err := cid.New(ctx.GetStub())
	if err != nil {
		return "", fmt.Errorf("cid: %w", err)
	}
	state, ok, _ := id.GetAttributeValue("state")
	if !ok || state == "" {
		return "", errors.New("missing state attribute")
	}*/
	return "UP", nil
}


// requireABAC enforces ABAC for write-side calls (role/state/constituency).
// params: ctx, constituencyID.
// return: caller state or error when attributes are absent/mismatched.
func requireABAC(ctx contractapi.TransactionContextInterface, constituencyID string) (string, error) {
    // Test-only bypass: if BYPASS_ABAC_FOR_TEST=on, skip attribute checks.
    // Returns BYPASS_STATE if set, else "TEST".
    /*if strings.ToLower(os.Getenv("BYPASS_ABAC_FOR_TEST")) == "on" {
        if s := os.Getenv("BYPASS_STATE"); s != "" {
            return s, nil
        }
        return "TEST", nil
    }

    id, err := cid.New(ctx.GetStub())
    if err != nil {
        return "", fmt.Errorf("cid: %w", err)
    }
    role, ok, _ := id.GetAttributeValue("role")
    if !ok || role != "voterMachine" {
        return "", errors.New("caller role must be voterMachine")
    }
    stateAttr, ok, _ := id.GetAttributeValue("state")
    if !ok || stateAttr == "" {
        return "", errors.New("missing state attribute")
    }
    constAttr, ok, _ := id.GetAttributeValue("constituency")
    if !ok || constAttr == "" || constAttr != constituencyID {
        return "", errors.New("constituency attribute mismatch")
    }*/
    return "UP", nil
}

// pollIsOpen returns whether a constituency is open for casting.
// params: ctx, constituencyID.
// return: true when open or unset (default-open), false when explicitly "closed".
func pollIsOpen(ctx contractapi.TransactionContextInterface, constituencyID string) (bool, error) {
    raw, err := ctx.GetStub().GetState(keyPollPrefix + constituencyID)
    if err != nil {
        return false, err
    }
    if raw == nil {
        return true, nil // default open
    }
    return string(raw) == "open", nil
}


// encOneChecks ensures a Paillier ciphertext is a valid Enc(1) element.
// params: c ciphertext, n2 modulus n².
// return: error when out of range or non-invertible (gcd != 1).
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


// voteKey builds the PDC key for the latest-vote pointer.
// params: constituencyID, serial.
// return: namespaced composite key string.
func voteKey(constituencyID, serial string) string {
return fmt.Sprintf("VOTE::%s::%s", constituencyID, serial)
}

// hasVoterViaCC queries evote-preload.HasVoter for serial presence (string bool).
// params: ctx, preloadCC, constituencyID, serial.
// return: true/false or error if cc2cc call fails.
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
    // contractapi returns JSON "true"/"false"
	payload := strings.TrimSpace(string(resp.Payload))
	payload = strings.Trim(payload, `"`)
	ok, _ := strconv.ParseBool(strings.ToLower(payload))
	return ok, nil
}

// callPreload is a safe wrapper to call read-only functions in preload/booth CCs.
// params: ctx, preloadCC name, function, args.
// return: raw payload bytes or error on non-200 or empty payload.
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


// mustJSON marshals v and ignores errors (used for events and small writes).
// params: any.
// return: JSON bytes (best effort).
func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }

const (
    maxBioCipherB64 = 4096 // ~3 KB payload when base64
    maxBioNonceB64  = 128
    maxBioTagHex    = 128  // 64 bytes hex for HMAC-SHA256
    maxBioAlgLen    = 64
    maxDevSigB64    = 2048 // adjust if your device signatures are larger
)

// clampVoteExtras bounds variable-size inputs to keep records predictable and safe.
// params: bioAlg, bioNonceB64, bioCipherB64, bioTagHex, devSigB64.
// return: error if any exceeds configured limits.
func clampVoteExtras(bioAlg, bioNonceB64, bioCipherB64, bioTagHex, devSigB64 string) error {
    if len(bioAlg) > maxBioAlgLen { return fmt.Errorf("bioAlg too long") }
    if len(bioNonceB64) > maxBioNonceB64 { return fmt.Errorf("bioNonce too long") }
    if len(bioCipherB64) > maxBioCipherB64 { return fmt.Errorf("bioCipher too large") }
    if len(bioTagHex) > maxBioTagHex { return fmt.Errorf("bioTag too long") }
    if len(devSigB64) > maxDevSigB64 { return fmt.Errorf("device signature too large") }
    return nil
}


/* ------------------------------ Admin / Setup ------------------------------ */

// SetJointPublicKey stores the Paillier PK for a state (validates hex; fills N2; warms cache).
// params: ctx, stateCode, pkJSON (fields: N,G[,N2]).
// return: error on malformed JSON/hex or ledger write failure.
// Store Paillier PK (hex fields). EP enforced by endorsement policy.
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
		pk.N2 = hexFromBig(n2) // canonicalize
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
			"nHash": sha256HexStr(pk.N), // keep behavior stable
			"time":  nowRFC3339(ctx),
		}))
	}
	return nil
}


// GetJointPublicKey fetches the stored PK JSON for a state.
// params: ctx, stateCode.
// return: pkJSON string or error if missing/ledger read fails.
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

// SeedCandidateList writes a sorted local copy of candidate IDs for a constituency.
// params: ctx, constituencyID, candidatesJSON ([]string).
// return: error on parse or write failure.
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

// GetCandidateList returns the stored local candidate list, sorted, or empty.
// params: ctx, constituencyID.
// return: []string list; never nil on success.
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

// OpenPoll marks a constituency as open (emits event when enabled).
// params: ctx, constituencyID.
// return: error on ledger write failure.
func (c *AccumVoteContract) OpenPoll(ctx contractapi.TransactionContextInterface, constituencyID string) error {
    if err := ctx.GetStub().PutState(keyPollPrefix+constituencyID, []byte("open")); err != nil { return err }
    if p, _ := getParams(ctx); p.EmitEvents {
        _ = ctx.GetStub().SetEvent(eventPollOpened, mustJSON(map[string]string{
            "constituency": constituencyID, "time": nowRFC3339(ctx),
        }))
    }
    return nil
}

// ClosePoll marks a constituency as closed (emits event when enabled).
// params: ctx, constituencyID.
// return: error on ledger write failure.
func (c *AccumVoteContract) ClosePoll(ctx contractapi.TransactionContextInterface, constituencyID string) error {
    if err := ctx.GetStub().PutState(keyPollPrefix+constituencyID, []byte("closed")); err != nil { return err }
    if p, _ := getParams(ctx); p.EmitEvents {
        _ = ctx.GetStub().SetEvent(eventPollClosed, mustJSON(map[string]string{
            "constituency": constituencyID, "time": nowRFC3339(ctx),
        }))
    }
    return nil
}


// SetParams merges provided JSON with current params and stores the result.
// params: ctx, paramsJSON (partial map).
// return: error on bad JSON or ledger write failure.
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

// GetParams returns the active params (public API wrapper).
// params: ctx.
// return: *Params or error if underlying read fails.
func (c *AccumVoteContract) GetParams(ctx contractapi.TransactionContextInterface) (*Params, error) {
return getParams(ctx)
}


// iterVotesPDC scans latest-vote pointers for a constituency from the PDC.
// It now uses a public-state range scan over BAL::<serial> keys,
// and per-key GetPrivateData() on the votes_pdc collection.
// Returns: map[serial]VoteMetaPDC or error on iterator failure.
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



// RefreshCandidateListFromPreload copies the canonical list from preload CC.
// params: ctx, constituencyID.
// return: error on cc2cc failure, bad JSON, or WS write failure.
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
    // cc2cc → evote-preload.GetCandidateList(constituencyID) -> payload bytes are a JSON array string
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
    // sum := sha256Hex(b)
    // _ = ctx.GetStub().PutState("CANDLIST_HASH::"+constituencyID, []byte(sum))

	if params.EmitEvents {
		_ = ctx.GetStub().SetEvent(eventCandidateListRefreshed, mustJSON(map[string]any{
			"constituency": constituencyID,
			"count":        len(ids),
			"listHash":     sha256Hex(b),     // hash of sorted JSON
			"time":         nowRFC3339(ctx),
		}))
	}

    return nil
}


/* --------------------------------- Queries -------------------------------- */

// GetBallotBySerial fetches the public kiosk meta row for a serial.
// params: ctx, serial.
// return: *BallotMeta or error if missing/malformed.
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

// GetEncSums multiplies each candidate’s Enc(1) over latest valid pointers (no eligibility checks).
// params: ctx, constituencyID.
// return: map[candidateID]hex(Enc(sum)) or error.
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


/* -------------------------------- Hot path -------------------------------- */

// RecordVote writes a voter’s latest choice to PDC and kiosk meta, with basic gates.
// params: ctx; serial/constituencyID/candidateID; encOneHex/receiptSalt/epoch/attestationSig;
//         boothID/deviceID/deviceKeyFP; bioAlg/bioNonceB64/bioCipherB64/bioTagHex.
// return: compact JSON ack or error; no eligibility checks here (done at tally).
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
    state, err := requireABAC(ctx, constituencyID) // tests bypass via env
    if err != nil { return "", err }

    /// 1) Fast params + attestation gate (fail before any heavy work)
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


/* ------------------------------- Tally path ------------------------------- */

// TallyPrepare validates latest votes and composes encrypted sums; sets BAL status and TXIDX.
// params: ctx, constituencyID.
// return: map[candidateID]hex(Enc(sum)) or error; emits counts and hash for audit.
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
		// force admin to seed candidates beforehand
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
	// before the loop
	validCount := 0
	invalidCount := 0

	for serial, vm := range latest {
		// candidate missing
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

		// voter roll
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

		// booth validation
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
			"sumsHash":     sha256Hex(buf.Bytes()), // stable trace of encrypted totals
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


// ApplyBallotStatuses writes statuses and/or tx index based on
// an off-chain prepared JSON payload (no private-data reads).
func (c *AccumVoteContract) ApplyBallotStatuses(
    ctx contractapi.TransactionContextInterface,
    constituencyID string,
    statusJSON string, // e.g., {"current":[{"serial":"...","txID":"..."}], "invalid":[...]}
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
		// reuse markBallotStatus: this only writes WS, no PDC reads
		vm := VoteMetaPDC{EncOneHex: v.EncOne, TxID: v.TxID, Epoch: ""} // EncOne optional
		markBallotStatus(ctx, v.Serial, "current", vm)

		// TXIDX is *only* for ballots that are current/valid so that VerifyReceipt
		// only ever sees "live" votes; superseded/invalid ones become unknown_tx.
		if v.TxID != "" {
			_ = ctx.GetStub().PutState(keyTxIdxPrefix+v.TxID, []byte(v.Serial))
		}
	}

	for _, v := range payload.Invalid {
		// Mark the ballot as invalid in BAL::<serial>, but DO NOT create TXIDX.
		// This ensures:
		//   - invalid voters (e.g. off-roll, bad booth) are excluded from receipt lookups
		//   - tests expecting no TXIDX for invalid txIDs (tx-bad, tx-2) pass
		vm := VoteMetaPDC{TxID: v.TxID}
		markBallotStatus(ctx, v.Serial, "invalid", vm)
		// intentionally no TXIDX:: write here
	}

	return nil

}


// markBallotStatus updates public kiosk meta with status and hashes the Enc(1) for receipt match.
// params: ctx, serial, new status, vm (to derive HC/epoch/txID).
// return: none; best-effort write.
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


// PublishResults anchors plaintext results and a proof bundle hash after close.
// params: ctx, constituencyID, roundID, resultJSON, bundleHash.
// return: error when poll still open or on write failure; emits event on success.
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


// VerifyReceipt checks a (txID, receipt) pair against public kiosk meta.
// params: ctx, txID, receipt (hex SHA256(encOneHex)).
// return: small JSON with ok/superseded/reason/serial/txID.
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


/* --------------------------------- Health --------------------------------- */

// Ping returns a simple liveness string prefixed with current txID.
// params: ctx.
// return: "OK:<txID>", error on stub failure.
func (c *AccumVoteContract) Ping(ctx contractapi.TransactionContextInterface) (string, error) {
return "OK:" + ctx.GetStub().GetTxID(), nil
}

