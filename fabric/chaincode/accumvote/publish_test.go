// publish_test.go
//
// Purpose: End-to-end and integration-style tests for the AccumVote contract’s
//          publishing/tally flow. These tests exercise: Paillier Enc(1) combine,
//          re-vote “latest wins”, voter eligibility via cc2cc, booth/device/time
//          validation, public result anchoring, and receipt verification.
// Role:    Runs against a lightweight harness (mocked ChaincodeStub + in-memory
//          WS/PDC) to validate the contract’s behavior without a real Fabric network.
// Key dependencies:
//   • Contract under test: AccumVoteContract (same package)
//   • Test harness: newHarness, memWorld, fakes.MockChaincodeStubInterface
//   • Fabric protos (peer/msp/queryresult) and chaincode-go shim/contractapi
//   • gomock for stubbing cc2cc and ledger calls
//   • Crypto: small Paillier helpers (for tests only), HMAC-SHA256, SHA256
//
// Notes:
//   • All helpers are intentionally tiny and deterministic.
//   • No production crypto here—Paillier bits are minimal and for testing only.

package main

import (
	"bytes"
	"crypto/hmac"
    "crypto/sha256"
	"crypto/rand"
    "encoding/json"
    "fmt"
	"strings"
    "math/big"
    "testing"
	"encoding/base64"
)

// cast holds one synthetic vote used by tests.
type cast struct {
    serial string
    cand   string
    r      int64
    salt   string
    txID   string
    enc1   string // filled after we compute Enc(1;r)
	hc     string // <- store RecordVote's HC to verify later
}


// -------------------- Minimal Paillier helpers for tests --------------------

// paillierTestKey is a toy keypair with just enough fields for unit tests.
type paillierTestKey struct {
	n, n2, lambda, mu *big.Int
}

// newPaillierTestKey builds a tiny (insecure) Paillier key from two primes.
// Params: p, q – small primes for test use.
// Returns: *paillierTestKey with n, n², lambda (lcm(p−1,q−1)), and mu precomputed.
func newPaillierTestKey(p, q int64) *paillierTestKey {
	P := big.NewInt(p)
	Q := big.NewInt(q)
	n := new(big.Int).Mul(P, Q)
	n2 := new(big.Int).Mul(n, n)
	l1 := new(big.Int).Sub(P, big.NewInt(1))
	l2 := new(big.Int).Sub(Q, big.NewInt(1))
	gcd := new(big.Int).GCD(nil, nil, l1, l2)
	l1Div := new(big.Int).Div(l1, gcd)
	lambda := new(big.Int).Mul(l1Div, l2) // lcm(p−1,q−1) without allocating an extra big.Int

	// mu = (L(g^lambda mod n^2))^{-1} mod n, with g=n+1 and L(u)=(u-1)/n
	g := new(big.Int).Add(n, big.NewInt(1))
	u := new(big.Int).Exp(g, lambda, n2)
	Lu := new(big.Int).Sub(u, big.NewInt(1))
	Lu.Div(Lu, n)
	mu := new(big.Int).ModInverse(Lu, n) // test-only; assume inverse exists

	return &paillierTestKey{n: n, n2: n2, lambda: lambda, mu: mu}
}

// encOneHex returns Enc(1; r) = (n+1) * r^n mod n^2 as lowercase hex (no 0x).
// Params: r – randomizer as int64.
// Returns: hex string of ciphertext.
func (k *paillierTestKey) encOneHex(r int64) string {
	R := big.NewInt(r)
	g := new(big.Int).Add(k.n, big.NewInt(1)) // standard g = n+1
	rn := new(big.Int).Exp(R, k.n, k.n2)
	c := new(big.Int).Mul(g, rn)
	c.Mod(c, k.n2)
	return fmt.Sprintf("%x", c)
}

// decCountT decrypts a ciphertext and returns the integer count.
// Accepts hex/0x/decimal via bigFromPossiblyHex (provided by another test file).
// Params: t (testing helper), cStr (ciphertext string).
// Returns: *big.Int plaintext in Z_n.
func (k *paillierTestKey) decCountT(t *testing.T, cStr string) *big.Int {
	t.Helper()
	c, err := bigFromPossiblyHex(cStr) // helper from tally_test.go
	requireNoErr(t, err)
	u := new(big.Int).Exp(c, k.lambda, k.n2)
	Lu := new(big.Int).Sub(u, big.NewInt(1))
	Lu.Div(Lu, k.n)
	m := new(big.Int).Mul(Lu, k.mu)
	m.Mod(m, k.n)
	return m
}

// ------------------------------ Test utilities ------------------------------

// setPK_Custom sets a public key on-ledger with g=n+1 for the test state.
// Params: t (testing helper), h (harness), n (modulus).
// Returns: none; fails the test on error.
func setPK_Custom(t *testing.T, h *testHarness, n *big.Int) {
	t.Helper()
	g := new(big.Int).Add(n, big.NewInt(1))
	pkJSON := fmt.Sprintf(`{"n":"0x%x","g":"0x%x"}`, n, g)
	requireNoErr(t, h.cc.SetJointPublicKey(h.ctx, testStateUP, pkJSON))
}

// fetchPublishedAnchor loads the stored TallyResultAnchor for (constituencyID, roundID).
// Params: t, h, constituencyID, roundID.
// Returns: parsed TallyResultAnchor; fails the test if missing or malformed.
func fetchPublishedAnchor(t *testing.T, h *testHarness, constituencyID, roundID string) TallyResultAnchor {
	t.Helper()
	key := fmt.Sprintf("%s%s::%s", keyResultsPrefix, constituencyID, roundID)
	raw, ok := h.mem.ws[key]
	if !ok || len(raw) == 0 {
		t.Fatalf("published anchor not found, key=%s", key)
	}
	var anchor TallyResultAnchor
	if err := json.Unmarshal(raw, &anchor); err != nil {
		t.Fatalf("bad anchor json: %v", err)
	}
	return anchor
}

// toJSON marshals any value to a JSON string and fails the test on error.
// Params: t, v.
// Returns: JSON string.
func toJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	requireNoErr(t, err)
	return string(b)
}

// makeEmbedding creates a synthetic 512-D float embedding with small integer pattern.
// Params: seed – shifts the pattern.
// Returns: []float32 of length 512.
func makeEmbedding(seed int) []float32 {
	emb := make([]float32, 512)
	for i := range emb {
		v := ((i+seed)%21 - 10) // [-10..+10]
		emb[i] = float32(v) / 10.0
	}
	return emb
}

// quantizeEmbeddingToU8 clamps an embedding to [-1,1] and quantizes to uint8.
// Params: emb []float32.
// Returns: []byte of same length in [0,255].
func quantizeEmbeddingToU8(emb []float32) []byte {
	q := make([]byte, len(emb))
	for i, v := range emb {
		if v < -1 { v = -1 }           // guard outliers
		if v > 1  { v = 1 }
		u := int((v+1.0)*0.5*255.0 + 0.5) // midpoint rounding
		if u < 0 { u = 0 }
		if u > 255 { u = 255 }
		q[i] = byte(u)
	}
	return q
}

// makeBioCipherB64 fakes a nonce||ciphertext blob and returns base64.
// Params: qbytes – payload to append after a random 12-byte nonce.
// Returns: base64 string.
func makeBioCipherB64(qbytes []byte) string {
	nonce := make([]byte, 12)
	_, _ = rand.Read(nonce) // randomness not asserted in tests
	buf := append(nonce, qbytes...)
	return base64.StdEncoding.EncodeToString(buf)
}

// computeBioTagHex builds HMAC-SHA256 over qbytes|serial|txID|const and returns hex.
// Params: qbytes, serial, txID, constituency, ktag (HMAC key).
// Returns: lowercase hex string of tag.
func computeBioTagHex(qbytes []byte, serial, txID, constituency string, ktag []byte) string {
	mac := hmac.New(sha256.New, ktag)
	mac.Write(qbytes)
	mac.Write([]byte("|")) // explicit separators to avoid accidental collisions
	mac.Write([]byte(serial))
	mac.Write([]byte("|"))
	mac.Write([]byte(txID))
	mac.Write([]byte("|"))
	mac.Write([]byte(constituency))
	return fmt.Sprintf("%x", mac.Sum(nil))
}

// ----------------------------- Tests 12 to 16 -------------------------------

// TestPublish_RequiresClosedPoll ensures PublishResults refuses when the poll is open.
// Params: t.
// Returns: none; assertions inside.
func TestPublish_RequiresClosedPoll(t *testing.T) {
    setProdEnv(t)
    h := newHarness(t); 
	defer h.ctrl.Finish()
	
	key := newPaillierTestKey(61, 53)    // n = 3233 (small test modulus)
    setPK_Custom(t, h, key.n)

    requireNoErr(t, h.setPK_UP())
    requireNoErr(t, h.seedCandidates([]string{testCand1}))

    // OPEN the poll so publish must reject.
    requireNoErr(t, h.openPoll())

    err := h.cc.PublishResults(h.ctx, testConst, "R0", `{"cand-000001":1}`, "BUNDLE")
    requireErrContains(t, err, "poll must be closed")
}

// TestPublish_HappyPath verifies anchor fields when publishing a simple result map.
// Params: t.
// Returns: none.
func TestPublish_HappyPath(t *testing.T) {
	setProdEnv(t)
	h := newHarness(t)

	// Intercept BOTH preload calls: GetCandidateList + HasVoter
	c1, c2, c3, c4 := "cand-000001", "cand-000002", "cand-000003", "cand-000004"
	h.stubPreloadCC([]string{c1, c2, c3, c4}, map[string]bool{
		"S-001": true, "S-002": true, "S-003": true, "S-004": true,
	})

	// Set the PK once (state=UP as per BYPASS_STATE)
	requireNoErr(t, h.setPK_UP())

	// (Optional) Seed a local list — TallyPrepare will refresh from preload anyway
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2}))

	requireNoErr(t, h.openPoll())

	// Minimal single vote to produce a non-identity for cand1
	h.setTxID("tx-1")
	_, err := h.recordVote("S-001", testCand1, hexEncOneGood)
	requireNoErr(t, err)

	requireNoErr(t, h.closePoll())

	_, err = h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	plain := map[string]uint64{testCand1: 1, testCand2: 0}
	resJSON := toJSON(t, plain) // assuming your helper exists
	round := "R1"
	bundle := "BUNDLE_ABC123"
	requireNoErr(t, h.cc.PublishResults(h.ctx, testConst, round, resJSON, bundle))

	anchor := fetchPublishedAnchor(t, h, testConst, round)
	if anchor.BundleHash != bundle {
		t.Fatalf("bundle mismatch: got %s want %s", anchor.BundleHash, bundle)
	}
	if string(anchor.Results) != resJSON {
		t.Fatalf("results mismatch: got %s want %s", string(anchor.Results), resJSON)
	}
}

// TestPublish_ConsistencyWithDecryption checks plaintext map matches decrypted sums.
// Params: t.
// Returns: none.
func TestPublish_ConsistencyWithDecryption(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	key := newPaillierTestKey(53, 59) // n = 3127
	setPK_Custom(t, h, key.n)
	
	h.stubPreloadCandidatesOnly([]string{testCand1, testCand2})

	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2}))
	requireNoErr(t, h.openPoll())
	h.setTxID("tx-1"); requireNoErr(t, must2(h.recordVote("S-001", testCand1, key.encOneHex(2))))
	h.setTxID("tx-2"); requireNoErr(t, must2(h.recordVote("S-002", testCand1, key.encOneHex(3))))
	h.setTxID("tx-3"); requireNoErr(t, must2(h.recordVote("S-003", testCand2, key.encOneHex(5))))
	requireNoErr(t, h.closePoll())

	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	count1 := key.decCountT(t, sums[testCand1]).Uint64()
	count2 := key.decCountT(t, sums[testCand2]).Uint64()

	got := map[string]uint64{testCand1: count1, testCand2: count2}
	round := "R3"
	payload := toJSON(t, got)
	requireNoErr(t, h.cc.PublishResults(h.ctx, testConst, round, payload, "HASH-R3"))

	anchor := fetchPublishedAnchor(t, h, testConst, round)
	var pub map[string]uint64
	requireNoErr(t, json.Unmarshal(anchor.Results, &pub))
	if pub[testCand1] != count1 || pub[testCand2] != count2 {
		t.Fatalf("published plaintext does not match decrypted tally: got=%v want=%v", pub, got)
	}
}

// TestPublish_EndToEnd_DecryptAndVerifyWinner exercises re-votes and winner detection.
// Params: t.
// Returns: none.
func TestPublish_EndToEnd_DecryptAndVerifyWinner(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	key := newPaillierTestKey(61, 53) // n = 3233
	setPK_Custom(t, h, key.n)

	c1 := "cand-000001"
	c2 := "cand-000002"
	c3 := "cand-000003"
	c4 := "cand-000004"
	requireNoErr(t, h.seedCandidates([]string{c1, c2, c3, c4}))
	
	h.stubPreloadCandidatesOnly([]string{c1, c2, c3, c4})

	requireNoErr(t, h.openPoll())

	// S1: c1 -> c3 ; S2: c2 -> c3 ; S3: c3 ; S4: c4 -> c3  ⇒ c3:4, others:0
	h.setTxID("tx-01"); requireNoErr(t, must2(h.recordVote("S1", c1, key.encOneHex(2))))
	h.setTxID("tx-02"); requireNoErr(t, must2(h.recordVote("S2", c2, key.encOneHex(3))))
	h.setTxID("tx-03"); requireNoErr(t, must2(h.recordVote("S3", c3, key.encOneHex(5))))
	h.setTxID("tx-04"); requireNoErr(t, must2(h.recordVote("S4", c4, key.encOneHex(7))))
	h.setTxID("tx-05"); requireNoErr(t, must2(h.recordVote("S1", c3, key.encOneHex(11))))
	h.setTxID("tx-06"); requireNoErr(t, must2(h.recordVote("S2", c3, key.encOneHex(13))))
	h.setTxID("tx-07"); requireNoErr(t, must2(h.recordVote("S4", c3, key.encOneHex(17))))
	requireNoErr(t, h.closePoll())

	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	cnt1 := key.decCountT(t, sums[c1]).Uint64()

	cnt2 := key.decCountT(t, sums[c2]).Uint64()
	cnt3 := key.decCountT(t, sums[c3]).Uint64()
	cnt4 := key.decCountT(t, sums[c4]).Uint64()

	if !(cnt1 == 0 && cnt2 == 0 && cnt4 == 0 && cnt3 == 4) {
		t.Fatalf("unexpected plaintext counts: c1=%d c2=%d c3=%d c4=%d", cnt1, cnt2, cnt3, cnt4)
	}

	round := "R4"
	plain := map[string]uint64{c1: cnt1, c2: cnt2, c3: cnt3, c4: cnt4}
	requireNoErr(t, h.cc.PublishResults(h.ctx, testConst, round, toJSON(t, plain), "HASH-R4"))
	anchor := fetchPublishedAnchor(t, h, testConst, round)

	var pub map[string]uint64
	requireNoErr(t, json.Unmarshal(anchor.Results, &pub))
	if pub[c3] != 4 || pub[c1] != 0 || pub[c2] != 0 || pub[c4] != 0 {
		t.Fatalf("published results do not match decrypted truth: %v", pub)
	}
}

// TestPublish_EndToEnd_RealScenario_WithQR simulates a realistic flow including
// cc2cc stubs, re-votes, publish, and saving per-tx receipt hashes (HC).
// Params: t.
// Returns: none.
func TestPublish_EndToEnd_RealScenario_WithQR(t *testing.T) {
    // Production-ish flags
    setProdEnv(t)

    h := newHarness(t)
    defer h.ctrl.Finish()

    // Paillier test key (same small primes we’ve used earlier)
    key := newPaillierTestKey(61, 53) // n=3233, n2=10465289
    setPK_Custom(t, h, key.n)

    // 1) Preload candidates (cc2cc stub → local refresh)
	c1, c2, c3, c4 := "cand-000001", "cand-000002", "cand-000003", "cand-000004"
	h.stubPreloadCC(
    []string{c1, c2, c3, c4},
    map[string]bool{
        "S1": true, "S2": true, "S3": true, "S4": true,
    },
    )

    //requireNoErr(t, h.cc.RefreshCandidateListFromPreload(h.ctx, testConst))
	requireNoErr(t, h.seedCandidates([]string{c1, c2, c3, c4}))
	
    requireNoErr(t, h.openPoll())

    // Voting pattern with re-votes; c3 should end up winning 3–1.
    // We’ll capture txIDs to verify QR receipts via chaincode.
	casts := []cast{
		{serial: "S1", cand: c1, r: 2,  salt: "salt-01", txID: "tx-01"},
		{serial: "S2", cand: c2, r: 3,  salt: "salt-02", txID: "tx-02"},
		{serial: "S3", cand: c3, r: 5,  salt: "salt-03", txID: "tx-03"},
		{serial: "S4", cand: c4, r: 7,  salt: "salt-04", txID: "tx-04"},
		{serial: "S1", cand: c3, r: 11, salt: "salt-05", txID: "tx-05"},
		{serial: "S2", cand: c3, r: 13, salt: "salt-06", txID: "tx-06"},
		{serial: "S4", cand: c3, r: 17, salt: "salt-07", txID: "tx-07"},
	}

    // Cast all ballots; capture HC for each txID. Do NOT verify receipts yet.
	for i := range casts {
		casts[i].enc1 = key.encOneHex(casts[i].r)
		h.setTxID(casts[i].txID)

		respJSON, err := h.cc.RecordVote(
			h.ctx,
			casts[i].serial, testConst, casts[i].cand,
			casts[i].enc1, casts[i].salt,
			testEpoch, testAttestOK,
			testBoothID, testDeviceID, testDeviceFP, // ← added
			"", "", "", "",                          // bio fields
		)
		requireNoErr(t, err)

		// Capture the hash-of-ciphertext (HC) returned by the contract.
		type voteResp struct {
			TxID     string `json:"txID"`
			Serial   string `json:"serial"`
			HC       string `json:"hC"`
			Status   string `json:"status"`
			Epoch    string `json:"epoch"`
			CastTime string `json:"castTime"`
		}
		var vr voteResp
		requireNoErr(t, json.Unmarshal([]byte(respJSON), &vr))

		casts[i].hc = vr.HC

		// Sanity: txID echoed back should match what we set
		if vr.TxID != casts[i].txID {
			t.Fatalf("txID mismatch: got=%s want=%s", vr.TxID, casts[i].txID)
		}
	}

    requireNoErr(t, h.closePoll())

    // Tally (accumulators → enc sums)
    sums, err := h.cc.TallyPrepare(h.ctx, testConst)
    requireNoErr(t, err)

    // Decrypt using test key; check winner c3
    cnt1 := key.decCountT(t, sums[c1]).Uint64()
    cnt2 := key.decCountT(t, sums[c2]).Uint64()
    cnt3 := key.decCountT(t, sums[c3]).Uint64()
    cnt4 := key.decCountT(t, sums[c4]).Uint64()
    if !(cnt3 == 4 && cnt1 == 0 && cnt2 == 0 && cnt4 == 0) {
    t.Fatalf("unexpected tally counts: c1=%d c2=%d c3=%d c4=%d", cnt1, cnt2, cnt3, cnt4)
    }

    // Publish and verify published map matches decrypted truth
    round := "R-real"
    plain := map[string]uint64{c1: cnt1, c2: cnt2, c3: cnt3, c4: cnt4}
    requireNoErr(t, h.cc.PublishResults(h.ctx, testConst, round, toJSON(t, plain), "HASH-R-REAL"))
    anchor := fetchPublishedAnchor(t, h, testConst, round)

    var pub map[string]uint64
    requireNoErr(t, json.Unmarshal(anchor.Results, &pub))
    if pub[c3] != 4 || pub[c1] != 0 || pub[c2] != 0 || pub[c4] != 0 {
    t.Fatalf("published results do not match decrypted truth: %v", pub)
    }
}

// Test_VerifyReceipt_Revote_OnlyLatestIsValid validates that only the latest
// vote survives VerifyReceipt after tally (old tx becomes unknown).
// Params: t.
// Returns: none.
func Test_VerifyReceipt_Revote_OnlyLatestIsValid(t *testing.T) {
	setProdEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	h.stubPreloadCC([]string{testCand1, testCand2}, map[string]bool{"S-001": true})
	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2}))

	requireNoErr(t, h.openPoll())

	// First vote -> cand1
	h.setTxID("tx-old")
	_, err := h.cc.RecordVote(
		h.ctx, "S-001", testConst, testCand1, hexEncOneGood, "salt1", testEpoch, testAttestOK,
		testBoothID, testDeviceID, testDeviceFP,
		"", "", "", "",
	)
	requireNoErr(t, err)

	// Second vote (latest-wins) -> cand2
	h.setTxID("tx-new")
	_, err = h.cc.RecordVote(
		h.ctx, "S-001", testConst, testCand2, hexEncOneGood, "salt2", testEpoch, testAttestOK,
		testBoothID, testDeviceID, testDeviceFP,
		"", "", "", "",
	)
	requireNoErr(t, err)

	requireNoErr(t, h.closePoll())

	// Read-only tally
	_, err = h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	// NEW: off-chain tally engine would now decide which tx is current and
	// apply the status plan. Only tx-new is marked "current"; tx-old is not
	// referenced at all, so VerifyReceipt should treat it as unknown_tx.
	statusJSON := fmt.Sprintf(
		`{"current":[{"serial":"%s","txID":"%s","encOneHex":"%s"}],"invalid":[]}`,
		"S-001", "tx-new", hexEncOneGood,
	)
	requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, statusJSON))

	// Receipt string is sha256(encOneHex) stored as BallotMeta.HC
	hashHex := func(s string) string {
		sum := sha256.Sum256([]byte(s))
		return fmt.Sprintf("%x", sum[:])
	}
	r := hashHex(hexEncOneGood)

	// Old tx must be unknown
	js, err := h.cc.VerifyReceipt(h.ctx, "tx-old", r)
	requireNoErr(t, err)
	if !strings.Contains(js, `"ok":false`) || !strings.Contains(js, `"unknown_tx"`) {
		t.Fatalf("old tx should be unknown, got %s", js)
	}

	// New tx must verify OK and not be marked superseded
	js2, err := h.cc.VerifyReceipt(h.ctx, "tx-new", r)
	requireNoErr(t, err)
	if !strings.Contains(js2, `"ok":true`) || strings.Contains(js2, `"superseded":true`) {
		t.Fatalf("new tx should be OK and not superseded, got %s", js2)
	}
}



// Test_EndToEnd_InvalidVoter_BioTag_Linkage checks that ineligible voters are
// excluded at tally and that BioTagHex values round-trip correctly in PDC.
// Params: t.
// Returns: none.
func Test_EndToEnd_InvalidVoter_BioTag_Linkage(t *testing.T) {
	setProdEnv(t) // VERIFY_ATTESTATION=on; VALIDATE_ON_TALLY=on; ValidateBoothOnTally=on
	h := newHarness(t)
	defer h.ctrl.Finish()

	// Preload: two candidates; GOOD is eligible, BAD is ineligible
	h.stubPreloadCC([]string{testCand1, testCand2}, map[string]bool{
		"S-GOOD": true,
		"S-BAD":  false, // <-- ineligible, must be excluded at tally
	})
	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2}))
	requireNoErr(t, h.openPoll())

	// Deterministic txIDs for inspection
	h.setTxID("tx-good")
	const embGood = "EMBEDDING-ABC"
	const saltGood = "salt-good"

	// Define a recomputable bioTag = sha256(embedding | constituency | txID | salt) hex
	mkBioTag := func(embedding, constituency, txID, salt string) string {
		var buf bytes.Buffer
		buf.WriteString(embedding)
		buf.WriteByte('|')
		buf.WriteString(constituency)
		buf.WriteByte('|')
		buf.WriteString(txID)
		buf.WriteByte('|')
		buf.WriteString(salt)
		sum := sha256.Sum256(buf.Bytes())
		return fmt.Sprintf("%x", sum[:])
	}
	tagGood := mkBioTag(embGood, testConst, "tx-good", saltGood)

	// GOOD vote (valid booth/device)
	_, err := h.cc.RecordVote(
		h.ctx, "S-GOOD", testConst, testCand1, hexEncOneGood, saltGood, testEpoch, testAttestOK,
		testBoothID, testDeviceID, testDeviceFP,
		"AES-256-GCM", "nonce", "cipher", tagGood,
	)
	requireNoErr(t, err)

	// BAD vote (same booth/device, but voter is ineligible)
	h.setTxID("tx-bad")
	const embBad = "EMBEDDING-XYZ"
	const saltBad = "salt-bad"
	tagBad := mkBioTag(embBad, testConst, "tx-bad", saltBad)
	_, err = h.cc.RecordVote(
		h.ctx, "S-BAD", testConst, testCand2, hexEncOneGood, saltBad, testEpoch, testAttestOK,
		testBoothID, testDeviceID, testDeviceFP,
		"AES-256-GCM", "nonce2", "cipher2", tagBad,
	)
	requireNoErr(t, err)

	requireNoErr(t, h.closePoll())

	// Read-only tally: only S-GOOD must count; S-BAD must be excluded from sums.
	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	// Expect enc sum for cand2 == identity "1" (only invalid voter chose cand2)
	if sums[testCand2] != "1" {
		t.Fatalf("cand2 should be identity (no valid votes), got %q", sums[testCand2])
	}

	// NEW: off-chain tally marks GOOD as current and BAD as invalid.
	statusJSON := fmt.Sprintf(
		`{"current":[{"serial":"%s","txID":"%s","encOneHex":"%s"}],`+
			`"invalid":[{"serial":"%s","txID":"%s"}]}`,
		"S-GOOD", "tx-good", hexEncOneGood,
		"S-BAD", "tx-bad",
	)
	requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, statusJSON))

	// BAD ballot meta must be invalid; GOOD must be current
	bmGood, err := h.cc.GetBallotBySerial(h.ctx, "S-GOOD")
	requireNoErr(t, err)
	if bmGood.Status != "current" {
		t.Fatalf("S-GOOD status = %s want current", bmGood.Status)
	}
	bmBad, err := h.cc.GetBallotBySerial(h.ctx, "S-BAD")
	requireNoErr(t, err)
	if bmBad.Status != "invalid" {
		t.Fatalf("S-BAD status = %s want invalid", bmBad.Status)
	}

	// TX index only for valid vote
	if _, ok := h.mem.ws[keyTxIdxPrefix+"tx-good"]; !ok {
		t.Fatalf("missing TXIDX for tx-good")
	}
	if _, badOK := h.mem.ws[keyTxIdxPrefix+"tx-bad"]; badOK {
		t.Fatalf("unexpected TXIDX for tx-bad (ineligible voter should not be indexed)")
	}

	// Recompute and verify BioTagHex persisted correctly
	vmGood := readVM(t, h, testConst, "S-GOOD")
	if vmGood.BioTagHex != tagGood {
		t.Fatalf("bio tag mismatch for GOOD: got %s want %s", vmGood.BioTagHex, tagGood)
	}
	vmBad := readVM(t, h, testConst, "S-BAD")
	if vmBad.BioTagHex != tagBad {
		t.Fatalf("bio tag mismatch for BAD: got %s want %s", vmBad.BioTagHex, tagBad)
	}
}


// Test_EndToEnd_InvalidVoter_Excluded_And_BioTag_Linkage is a variant that uses
// a deterministic 512-byte “embedding” and HMAC-based BioTag to assert linkage.
// Params: t.
// Returns: none.
func Test_EndToEnd_InvalidVoter_Excluded_And_BioTag_Linkage(t *testing.T) {
	setProdEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	// Preload: 2 candidates; S-002 is NOT on the voter roll (invalid).
	h.stubPreloadCC(
		[]string{testCand1, testCand2},
		map[string]bool{"S-001": true, "S-002": false, "S-003": true},
	)

	// PK + open poll
	requireNoErr(t, h.setPK_UP())
	requireNoErr(t, h.seedCandidates([]string{testCand1, testCand2}))
	requireNoErr(t, h.openPoll())

	// Fixed dummy 512-D embedding (pretend InsightFace), BioTag via HMAC-SHA256
	Ktag := []byte("test-ktag-secret-32bytes-pad-test-ktag!") // 32 bytes for HMAC key in tests
	emb := dummyEmbed512()                                    // deterministic 512 bytes

	// Vote 1: valid voter S-001 -> cand1
	h.setTxID("tx-1")
	tag1 := bioTagHexFor(Ktag, emb, "S-001", "tx-1", testConst)
	_, err := h.cc.RecordVote(h.ctx, "S-001", testConst, testCand1, hexEncOneGood, "salt", testEpoch, testAttestOK,
		testBoothID, testDeviceID, testDeviceFP,
		"AES-256-GCM", "NONCE-1", "CIPHER-1", tag1)
	requireNoErr(t, err)

	// Vote 2: INVALID voter S-002 -> cand2 (should be excluded at tally)
	h.setTxID("tx-2")
	tag2 := bioTagHexFor(Ktag, emb, "S-002", "tx-2", testConst)
	_, err = h.cc.RecordVote(h.ctx, "S-002", testConst, testCand2, hexEncOneGood, "salt", testEpoch, testAttestOK,
		testBoothID, testDeviceID, testDeviceFP,
		"AES-256-GCM", "NONCE-2", "CIPHER-2", tag2)
	requireNoErr(t, err)

	// Vote 3: valid voter S-003 -> cand1
	h.setTxID("tx-3")
	tag3 := bioTagHexFor(Ktag, emb, "S-003", "tx-3", testConst)
	_, err = h.cc.RecordVote(h.ctx, "S-003", testConst, testCand1, hexEncOneGood, "salt", testEpoch, testAttestOK,
		testBoothID, testDeviceID, testDeviceFP,
		"AES-256-GCM", "NONCE-3", "CIPHER-3", tag3)
	requireNoErr(t, err)

	// Close poll and tally
	requireNoErr(t, h.closePoll())
	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)

	// cand2 must be identity (its only vote was from an invalid voter)
	if got := sums[testCand2]; got != "1" {
		t.Fatalf("cand2 should be identity (no valid votes), got %q", got)
	}
	// cand1 must be non-identity (two valid votes multiplied)
	if got := sums[testCand1]; got == "1" {
		t.Fatalf("cand1 should NOT be identity (has valid votes), got %q", got)
	}

	// NEW: off-chain tally plan — S-001 and S-003 current, S-002 invalid.
	statusJSON := fmt.Sprintf(
		`{"current":[`+
			`{"serial":"%s","txID":"%s","encOneHex":"%s"},`+
			`{"serial":"%s","txID":"%s","encOneHex":"%s"}],`+
			`"invalid":[{"serial":"%s","txID":"%s"}]}`,
		"S-001", "tx-1", hexEncOneGood,
		"S-003", "tx-3", hexEncOneGood,
		"S-002", "tx-2",
	)
	requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, statusJSON))

	// Ballot meta for S-002 must be marked invalid
	bm, err := h.cc.GetBallotBySerial(h.ctx, "S-002")
	requireNoErr(t, err)
	if bm.Status != "invalid" {
		t.Fatalf("S-002 ballot status = %q, want invalid", bm.Status)
	}

	// BioTag linkage: the tag we computed must be what landed in PDC for each serial
	latest, err := iterVotesPDC(h.ctx, testConst) // same package, so we can call it
	requireNoErr(t, err)

	if latest["S-001"].BioTagHex != tag1 {
		t.Fatalf("tag1 mismatch")
	}
	if latest["S-002"].BioTagHex != tag2 {
		t.Fatalf("tag2 mismatch")
	}
	if latest["S-003"].BioTagHex != tag3 {
		t.Fatalf("tag3 mismatch")
	}

	// Optional: invalid voter should NOT have a TXIDX entry
	if v, _ := h.mem.getState("TXIDX::tx-2"); v != nil {
		t.Fatalf("invalid voter should not create TXIDX, but found one")
	}
}


/* ---- test-only helpers (place in the same _test.go file) ---- */

// dummyEmbed512 returns a deterministic 512-byte vector (not real biometrics).
// Params: none.
// Returns: []byte length 512.
func dummyEmbed512() []byte {
    // Deterministic 512-byte vector so the HMAC is stable across runs
    b := make([]byte, 512)
    for i := 0; i < 512; i++ { b[i] = byte((i*31 + 7) & 0xff) }
    return b
}

// bioTagHexFor computes base64(HMAC-SHA256(Ktag, embed|serial|txID|const)).
// Params: Ktag key, embed bytes, serial, txID, constituency.
// Returns: base64 string of tag.
func bioTagHexFor(Ktag, embed []byte, serial, txID, constituency string) string {
    // HMAC-SHA256(Ktag, embed || "|" || serial || "|" || txID || "|" || constituency)
    mac := sha256.New
    h := hmac.New(mac, Ktag)
    h.Write(embed)
    h.Write([]byte("|"))
    h.Write([]byte(serial))
    h.Write([]byte("|"))
    h.Write([]byte(txID))
    h.Write([]byte("|"))
    h.Write([]byte(constituency))
    return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

// computeQRTest reproduces the off-chain QR logic used by clients.
// Formula: QR = base64url( SHA256( const || txID || salt || SHA256(encOneHex) ) )
// Params: constituencyID, txID, encOneHex, receiptSalt.
// Returns: base64url string.
func computeQRTest(constituencyID, txID, encOneHex, receiptSalt string) string {
    inner := sha256.Sum256([]byte(encOneHex))
    buf := bytes.NewBuffer(nil)
    buf.WriteString(constituencyID)
    buf.WriteString(txID)
    buf.WriteString(receiptSalt)
    buf.Write(inner[:]) // feed the 32 bytes, not the hex string
    sum := sha256.Sum256(buf.Bytes())
    return b64url(sum[:])
}

// b64url encodes bytes with URL-safe base64 and no padding.
// Params: b.
// Returns: string.
func b64url(b []byte) string {
    s := base64.URLEncoding.WithPadding(base64.NoPadding).EncodeToString(b)
    return s
}
