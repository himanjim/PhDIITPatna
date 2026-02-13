// Harness_test.go
//
// Purpose: Minimal, deterministic test harness for the AccumVote chaincode.
// Role: Provides an in-memory world-state/private-data “ledger”, a mocked Fabric
// ChaincodeStub (via gomock), and focused helpers to stub cross-chaincode
// Calls (evote-preload, boothpdc). It lets tests drive the contract
// Without real peers, orderers, or crypto material.
// Key deps:
// - Hyperledger Fabric Go SDKs: chaincode-go/shim, contractapi, protos (peer, msp, queryresult)
// - gomock for stub expectations and return paths
// - Google protobuf/timestamppb for stable TxTimestamp values
// - Local fakes package: github.com/yourorg/accumvote_cc/fakes (mock stub interface)
// Notes:
// - This harness makes defensive copies of byte slices to avoid aliasing between
// The test code and the “ledger” maps.
// - It purposely keeps behavior small and predictable; only the code paths used
// By the tests are mocked.

package main

import (
    "crypto/rand"
    "crypto/rsa"
    "crypto/x509"
    "encoding/pem"
    "encoding/json"
    "fmt"
    "math/big"
    "strings"
    testing "testing"
    "time"
    "sort"

    msp "github.com/hyperledger/fabric-protos-go-apiv2/msp"
    pb "github.com/hyperledger/fabric-protos-go-apiv2/peer"
    queryresult "github.com/hyperledger/fabric-protos-go-apiv2/ledger/queryresult"

    "github.com/golang/mock/gomock"
    "github.com/hyperledger/fabric-chaincode-go/v2/shim"
    "github.com/hyperledger/fabric-chaincode-go/v2/pkg/cid"
    contractapi "github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
    "google.golang.org/protobuf/proto"
    "google.golang.org/protobuf/types/known/timestamppb"
	f "github.com/yourorg/accumvote_cc/fakes"
)

const (
    testStateUP   = "UP"
    testConst     = "C-001"
	testBoothID      = "B-0001"
    testDeviceID     = "D0010001000"
    testDeviceFP     = "5eeea7e7483b89aa2b79a8050747cff9"
    testOpenTime  int64 = 1763173800
    testCloseTime int64 = 1763206200
    testCand1     = "cand-000001"
    testCand2     = "cand-000002"
    testCand3     = "cand-000003"   // <- add
    testCand4     = "cand-000004"   // <- add
    testEpoch     = "E1"
    testSalt      = "deadbeef"
    testAttestOK  = "FIXED"
    hexN          = "0xca1"
    hexG          = "0xca2"
    hexEncOneGood = "0x0bb9"
)


/* in-memory WS/PDC harness */

// MemWorld is a tiny in-memory ledger used by the mock stub.
// It tracks world state (ws), private data (pdc), emitted events, and op counts.
type memWorld struct {
    ws  map[string][]byte
    pdc map[string]map[string][]byte
    events []struct{ name string; payload []byte }
    opsCounts struct {
        getState, putState int
        getPDC, putPDC     int
        setEvent           int
    }
}

// NewMemWorld allocates an empty memWorld.
// Params: none.
// Returns: pointer to a zeroed, ready-to-use memWorld.
func newMemWorld() *memWorld {
    return &memWorld{ws: make(map[string][]byte), pdc: make(map[string]map[string][]byte)}
}

// GetState simulates GetState on the in-mem world state.
// Copies the value before returning to avoid aliasing in tests.
// Params: key (string).
// Returns: value ([]byte) or nil, error (always nil here).
func (m *memWorld) getState(key string) ([]byte, error) {
    m.opsCounts.getState++
    if v, ok := m.ws[key]; ok {
        return append([]byte(nil), v...), nil // Copy for safety
    }
    return nil, nil
}

// PutState simulates PutState on the in-mem world state.
// Params: key, value.
// Returns: error (always nil here).
func (m *memWorld) putState(key string, val []byte) error {
    m.opsCounts.putState++
    m.ws[key] = append([]byte(nil), val...) // Copy for safety
    return nil
}

// GetPDC simulates GetPrivateData from a named collection.
// Params: coll, key.
// Returns: value or nil, error (always nil here).
func (m *memWorld) getPDC(coll, key string) ([]byte, error) {
    m.opsCounts.getPDC++
    if c, ok := m.pdc[coll]; ok {
        if v, ok2 := c[key]; ok2 {
            return append([]byte(nil), v...), nil // Copy for safety
        }
    }
    return nil, nil
}

// PutPDC simulates PutPrivateData into a named collection.
// Lazily creates the collection map if needed.
// Params: coll, key, value.
// Returns: error (always nil here).
func (m *memWorld) putPDC(coll, key string, val []byte) error {
    m.opsCounts.putPDC++
    c := m.pdc[coll]
    if c == nil {
        c = make(map[string][]byte)
        m.pdc[coll] = c
    }
    c[key] = append([]byte(nil), val...) // Copy for safety
    return nil
}

// SetEvent records a chaincode event into the in-mem log.
// Params: name, payload.
// Returns: error (always nil here).
func (m *memWorld) setEvent(name string, payload []byte) error {
    m.opsCounts.setEvent++
    m.events = append(m.events, struct {
        name string; payload []byte
    }{name: name, payload: append([]byte(nil), payload...)}) // Copy for safety
    return nil
}

// MemPDCIter is a simple iterator over a pre-materialized slice of PDC keys/values.
// It implements the subset of shim.StateQueryIteratorInterface used by tests.
type memPDCIter struct {
    keys []string
    vals [][]byte
    i    int
}

// HasNext tells whether another KV is available.
// Params: none.
// Returns: bool.
func (it *memPDCIter) HasNext() bool { return it.i < len(it.keys) }

// Next returns the current KV and advances the iterator.
// Params: none.
// Returns: *queryresult.KV, error when past the end.
func (it *memPDCIter) Next() (*queryresult.KV, error) {
    if !it.HasNext() { return nil, fmt.Errorf("iterator exhausted") }
    kv := &queryresult.KV{Key: it.keys[it.i], Value: it.vals[it.i]}
    it.i++
    return kv, nil
}

// Close is a no-op to satisfy the interface.
// Params: none.
// Returns: error (always nil here).
func (it *memPDCIter) Close() error { return nil }

// IterPDCRange materializes a range scan over a PDC collection.
// It honors [start, end) lexicographic bounds and sorts keys for deterministic order.
// Params: coll, start, end.
// Returns: an iterator over the selected KV slice.
func (m *memWorld) iterPDCRange(coll, start, end string) *memPDCIter {
    c := m.pdc[coll]
    if c == nil { return &memPDCIter{} }
    var keys []string
    for k := range c {
        // Range semantics are inclusive of start and exclusive of end, matching Fabric behavior.
        if (start == "" || k >= start) && (end == "" || k < end) { keys = append(keys, k) }
    }
    sort.Strings(keys) // Keep scans stable across runs
    vals := make([][]byte, len(keys))
    for i, k := range keys { vals[i] = append([]byte(nil), c[k]...) } // Copy for safety
    return &memPDCIter{keys: keys, vals: vals}
}


// IterWSRange materializes a range scan over world state (ws).
// It honors [start, end) lexicographic bounds and sorts keys for deterministic order.
// Params: start, end.
// Returns: an iterator over the selected KV slice.
func (m *memWorld) iterWSRange(start, end string) *memPDCIter {
    if m.ws == nil {
        return &memPDCIter{}
    }
    var keys []string
    for k := range m.ws {
        if (start == "" || k >= start) && (end == "" || k < end) {
            keys = append(keys, k)
        }
    }
    sort.Strings(keys)
    vals := make([][]byte, len(keys))
    for i, k := range keys {
        vals[i] = append([]byte(nil), m.ws[k]...) // Copy for safety
    }
    return &memPDCIter{keys: keys, vals: vals}
}


// --- booth cc stubbing ---

// BoothStubRec describes the shape of a booth record when mocking boothpdc.
// It is here for clarity; the test harness constructs JSON directly when returning.
type boothStubRec struct {
    StateCode, ConstituencyID, BoothID string
    Status string
    OpenTime, CloseTime int64
    DeviceID, DeviceKeyFingerprint string
}

// StubBoothOK wires gomock expectations to answer boothpdc calls for a single booth.
// It handles HasBooth and GetBooth for the provided state/constituency/booth, returning
// An active row with the given device and time window.
// Params: state, cid, booth, devID, devFP, open, close.
// Returns: none.
func (h *testHarness) stubBoothOK(state, cid, booth, devID, devFP string, open, close int64) {
    h.stub.EXPECT().
        InvokeChaincode(
            gomock.Eq("boothpdc"),                       // Cc name
            gomock.AssignableToTypeOf([][]byte{}), // Args
            gomock.Any(),                          // Channel
        ).
        AnyTimes().
        DoAndReturn(func(cc string, args [][]byte, ch string) *pb.Response {
            if cc != "boothpdc" {
                return &pb.Response{Status: 404, Message: "InvokeChaincode: not boothpdc"}
            }
            if len(args) == 0 {
                return &pb.Response{Status: int32(shim.ERROR), Message: "missing fcn"}
            }
            fcn := string(args[0])
            switch fcn {
            case "HasBooth":
                if len(args) < 4 { return &pb.Response{Status: int32(shim.ERROR), Message: "bad args"} }
                ok := string(args[1]) == state && string(args[2]) == cid && string(args[3]) == booth
                if ok { return &pb.Response{Status: int32(shim.OK), Payload: []byte(`"true"`)} }
                return &pb.Response{Status: int32(shim.OK), Payload: []byte(`"false"`)}
            case "GetBooth":
                if len(args) < 4 { return &pb.Response{Status: int32(shim.ERROR), Message: "bad args"} }
                match := string(args[1]) == state && string(args[2]) == cid && string(args[3]) == booth
                if !match { return &pb.Response{Status: 404, Message: "booth not found"} }
                b, _ := json.Marshal(map[string]any{
                    "status": "A",
                    "open_time":  open,
                    "close_time": close,
                    "device_id": devID,
                    "device_key_fingerprint": devFP,
                })
                return &pb.Response{Status: int32(shim.OK), Payload: b}
            default:
                return &pb.Response{Status: 404, Message: "unknown fcn for boothpdc"}
            }
        })
}

// ReadVM is a test helper that fetches the latest VoteMetaPDC for a serial from the in-mem PDC.
// It fails the test if the key is missing or JSON is malformed.
// Params: t (testing.T), h (harness), constituencyID, serial.
// Returns: VoteMetaPDC value.
func readVM(t *testing.T, h *testHarness, constituencyID, serial string) VoteMetaPDC {
	t.Helper()
	coll := votesPDC
	key := voteKey(constituencyID, serial)
	cm := h.mem.pdc[coll]
	if cm == nil {
		t.Fatalf("PDC %s empty (want %s)", coll, key)
	}
	raw, ok := cm[key]
	if !ok {
		t.Fatalf("missing PDC key %s", key)
	}
	var vm VoteMetaPDC
	if err := json.Unmarshal(raw, &vm); err != nil {
		t.Fatalf("bad PDC json for %s: %v", key, err)
	}
	return vm
}


/* tx context w/ real stub (no gomock ctx) */

// SimpleTxCtx adapts a raw shim.ChaincodeStubInterface to a contractapi TransactionContext.
// It keeps the shape tiny because tests only need GetStub.
// Params: none (constructed with a stub field).
// Returns: methods to satisfy contractapi.TransactionContextInterface.
type simpleTxCtx struct{ s shim.ChaincodeStubInterface }

// GetStub returns the underlying ChaincodeStubInterface.
func (c *simpleTxCtx) GetStub() shim.ChaincodeStubInterface { return c.s }

// GetClientIdentity is not used by the tests; it returns nil to satisfy the interface.
func (c *simpleTxCtx) GetClientIdentity() cid.ClientIdentity { return nil }

/* test harness (single definition) */

// TestHarness bundles the mock controller, stub, in-mem ledger, and the contract under test.
// It also tracks a mutable txID to let tests simulate different transactions.
type testHarness struct {
    ctrl *gomock.Controller
    ctx  contractapi.TransactionContextInterface
    stub *f.MockChaincodeStubInterface
    mem  *memWorld
    cc   *AccumVoteContract
    t    *testing.T
    txID string
}

// newHarness builds a mocked Fabric transaction context for unit tests.
// It wires world state and private data collections to in-memory maps.
// The harness resets global caches (e.g., pkCache) to keep tests isolated.
func newHarness(t *testing.T) *testHarness {
    t.Helper()

    ctrl := gomock.NewController(t)
    stub := f.NewMockChaincodeStubInterface(ctrl)
    txctx := &simpleTxCtx{s: stub}
    mem := newMemWorld()

    h := &testHarness{
        ctrl: ctrl, ctx: txctx, stub: stub, mem: mem,
        cc: new(AccumVoteContract), t: t, txID: "tx-0001",
    }
	
	// Provide a valid creator so contract code can parse MSP/attributes if it wants.
    stub.EXPECT().GetCreator().AnyTimes().Return(devSerializedIdentity("ECIMSP"), nil)

    // Return the current harness txID; tests may override it per case.
    stub.EXPECT().GetTxID().AnyTimes().DoAndReturn(func() string { return h.txID })
    
	// Pick a cast time inside the booth window so time checks pass by default.
	const testCastTime int64 = testOpenTime + 60 // Any value within [open, close]

	stub.EXPECT().
		GetTxTimestamp().
		AnyTimes().
		Return(&timestamppb.Timestamp{Seconds: testCastTime}, nil)

	// Stable channel ID used by the contract.
    stub.EXPECT().GetChannelID().AnyTimes().Return("statechan-01")

    // Wire world state and PDC to the in-mem maps.
    stub.EXPECT().GetState(gomock.Any()).AnyTimes().DoAndReturn(mem.getState)
    stub.EXPECT().PutState(gomock.Any(), gomock.Any()).AnyTimes().DoAndReturn(mem.putState)
    stub.EXPECT().GetPrivateData(gomock.Any(), gomock.Any()).AnyTimes().DoAndReturn(mem.getPDC)
    stub.EXPECT().PutPrivateData(gomock.Any(), gomock.Any(), gomock.Any()).AnyTimes().DoAndReturn(mem.putPDC)

    // Range queries return a lightweight iterator backed by a prebuilt slice.
    stub.EXPECT().
        GetPrivateDataByRange(gomock.Any(), gomock.Any(), gomock.Any()).
        AnyTimes().
        DoAndReturn(func(coll, start, end string) (shim.StateQueryIteratorInterface, error) {
            return mem.iterPDCRange(coll, start, end), nil
        })
		
	// World-state range queries (used by iterVotesPDC over BAL:: keys).
	stub.EXPECT().
		GetStateByRange(gomock.Any(), gomock.Any()).
		AnyTimes().
		DoAndReturn(func(start, end string) (shim.StateQueryIteratorInterface, error) {
			return mem.iterWSRange(start, end), nil
		})


    // Capture events into the in-mem log for assertions.
    stub.EXPECT().SetEvent(gomock.Any(), gomock.Any()).AnyTimes().DoAndReturn(mem.setEvent)
	
	// Default booth row: B-0001 is active and matches device/time window.
	// Keep this near the end so it takes effect for all tests unless they override.
	h.stubBoothOK(testStateUP, testConst, testBoothID, testDeviceID, testDeviceFP, testOpenTime, testCloseTime)

    return h
}

/* cc2cc stub (pointer return matches your shim) */

// StubPreloadCC mocks evote-preload for both candidate list and voter eligibility.
// It responds to multiple function names to tolerate minor variations in upstream code.
// Params: cands (candidate IDs), eligible map[serial]bool (nil => everyone eligible).
// Returns: none.
func (h *testHarness) stubPreloadCC(cands []string, eligible map[string]bool) {
    payloadCands := toJSONBytes(cands)

    h.stub.EXPECT().
        InvokeChaincode(
            gomock.Eq("evote-preload"), 
            gomock.AssignableToTypeOf([][]byte{}),
            gomock.Any(),
        ).
        AnyTimes().
        DoAndReturn(func(cc string, args [][]byte, ch string) *pb.Response {
            if len(args) == 0 {
                return &pb.Response{Status: int32(shim.ERROR), Message: "missing fcn"}
            }
            fcn := string(args[0])
            switch fcn {
            case "GetCandidateList", "GetCandidateIDs", "GetCandidateListJSON",
                 "GetCandidatesForConstituency", "GetCandidates":
                return &pb.Response{Status: int32(shim.OK), Payload: payloadCands}
            case "HasVoter", "IsVoterEligible", "VerifyVoter", "CheckVoterEligible", "HasVoterPDC":
                // Accept either arg positions [constituency, serial] or just [serial].
                var serial string
                if len(args) >= 3 { serial = string(args[2]) } else if len(args) >= 2 { serial = string(args[1]) } else {
                    return &pb.Response{Status: int32(shim.ERROR), Message: "bad args for voter check"}
                }
                ok := true
                if eligible != nil {
                    if v, exists := eligible[serial]; exists { ok = v }
                }
                if ok { return &pb.Response{Status: int32(shim.OK), Payload: []byte("true")} }
                return &pb.Response{Status: int32(shim.OK), Payload: []byte("false")}
            default:
                return &pb.Response{Status: 404, Message: "not mocked: " + fcn}
            }
        })
}

// StubPreloadCandidatesOnly is a shorthand to mock only candidate enumeration.
// Params: cands.
// Returns: none.
func (h *testHarness) stubPreloadCandidatesOnly(cands []string) { h.stubPreloadCC(cands, nil) }

/* small helpers */

// SetTxID overrides the txID seen by the contract for the next operations.
// Params: id.
// Returns: none.
func (h *testHarness) setTxID(id string) { h.txID = id }

// SetPK_UP writes a simple public key JSON for the test state and calls SetJointPublicKey.
// It uses the contract under test so code paths remain realistic.
// Params: none.
// Returns: error from SetJointPublicKey.
func (h *testHarness) setPK_UP() error {
    pkJSON := `{"n":"` + hexN + `","g":"` + hexG + `"}`
    return h.cc.SetJointPublicKey(h.ctx, testStateUP, pkJSON)
}

// OpenPoll calls the contract’s OpenPoll for the test constituency.
// Params: none.
// Returns: error from the contract.
func (h *testHarness) openPoll() error  { return h.cc.OpenPoll(h.ctx, testConst) }

// ClosePoll calls the contract’s ClosePoll for the test constituency.
// Params: none.
// Returns: error from the contract.
func (h *testHarness) closePoll() error { return h.cc.ClosePoll(h.ctx, testConst) }

// SeedCandidates persists a candidate list into world state using the contract API.
// Params: ids (candidate IDs).
// Returns: error from the contract.
func (h *testHarness) seedCandidates(ids []string) error {
    b := jsonMarshal(ids)
    return h.cc.SeedCandidateList(h.ctx, testConst, string(b))
}

// RecordVote submits a RecordVote on the contract with sensible defaults for booth/device fields.
// Params: serial, cand, enc1 (Enc(1) hex).
// Returns: contract’s JSON string response and error.
func (h *testHarness) recordVote(serial, cand, enc1 string) (string, error) {
    return h.cc.RecordVote(
        h.ctx, serial, testConst, cand, enc1, "salt", testEpoch, testAttestOK,
        testBoothID, testDeviceID, testDeviceFP,  // ← use constants
        "", "", "", "",
    )
}

// RequireNoErr fails the test immediately if err != nil, labeling it unexpected.
// Params: t, err.
// Returns: none.
func requireNoErr(t *testing.T, err error) {
    t.Helper()
    if err != nil { t.Fatalf("unexpected error: %v", err) }
}

// RequireErrContains asserts that err is non-nil and its message contains wantSubstr (case-insensitive).
// Params: t, err, wantSubstr (may be empty to assert only non-nil).
// Returns: none.
func requireErrContains(t *testing.T, err error, wantSubstr string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected error containing %q, got nil", wantSubstr)
	}
	if wantSubstr != "" && !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(wantSubstr)) {
		t.Fatalf("error %q does not contain %q", err.Error(), wantSubstr)
	}
}


/* tiny JSON & identity helpers */

// JsonMarshal is a minimal, allocation-aware encoder for []string used by tests.
// It exists to avoid pulling in more JSON wrangling and to keep output stable.
// Params: v (only []string is supported; anything else returns "null").
// Returns: raw JSON bytes.
func jsonMarshal(v any) []byte {
    switch s := v.(type) {
    case []string:
        if len(s) == 0 { return []byte("[]") }
        var b strings.Builder
        b.WriteByte('[')
        for i, e := range s {
            if i > 0 { b.WriteByte(',') }
            b.WriteByte('"')
            b.WriteString(strings.ReplaceAll(e, `"`, `\"`))
            b.WriteByte('"')
        }
        b.WriteByte(']')
        return []byte(b.String())
    default:
        return []byte("null")
    }
}

// ToJSONBytes marshals any Go value using encoding/json with errors ignored for tests.
// Params: v.
// Returns: JSON bytes (best effort).
func toJSONBytes(v any) []byte { b, _ := json.Marshal(v); return b }

// DevSerializedIdentity generates a minimal SerializedIdentity with a self-signed cert.
// It’s good enough for GetCreator parsing in contract code.
// Params: ms (MSP ID).
// Returns: raw serialized identity bytes.
func devSerializedIdentity(ms string) []byte {
    key, _ := rsa.GenerateKey(rand.Reader, 1024)
    tpl := &x509.Certificate{SerialNumber: big.NewInt(1), NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour)}
    der, _ := x509.CreateCertificate(rand.Reader, tpl, tpl, &key.PublicKey, key)
    pemCert := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
    sid := &msp.SerializedIdentity{Mspid: ms, IdBytes: pemCert}
    b, _ := proto.Marshal(sid)
    return b
}

/* env presets used by your tests */

// SetDefaultEnv applies production-like toggles and enables ABAC bypass for unit tests.
// It sets BYPASS_STATE to ensure deterministic state resolution.
// Params: t.
// Returns: none.
func setDefaultEnv(t *testing.T) {
    setProdEnv(t)
    t.Setenv("BYPASS_ABAC_FOR_TEST", "on")
    t.Setenv("BYPASS_STATE", testStateUP)
}

// SetProdEnv sets sane defaults mirroring the intended production posture,
// While still allowing unit tests to run deterministically.
// Params: t.
// Returns: none.
func setProdEnv(t *testing.T) {
    t.Setenv("VALIDATE_CANDIDATE", "on")
    t.Setenv("VERIFY_ATTESTATION", "on")
    t.Setenv("VALIDATE_ON_TALLY", "on")
    t.Setenv("WRITE_BALLOT_META", "on")
    t.Setenv("EMIT_EVENTS", "on")
    t.Setenv("PRELOAD_CC_NAME", "evote-preload")
    t.Setenv("VOTER_ROLL_PDC", "voter_roll_pdc")
    t.Setenv("ACCUM_SHARDS", "64")
    t.Setenv("SHARD_SALT", "bench-salt")
    t.Setenv("BYPASS_ABAC_FOR_TEST", "on")
    t.Setenv("BYPASS_STATE", testStateUP)
}
