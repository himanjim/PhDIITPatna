// voterview_test.go covers the kiosk view of a voter's latest ballot.
//
// The feature needs one thing from the contract and one check at the kiosk.
//
//  1. ApplyBallotStatuses must be able to run while the poll is open (rolling
//     reconciliation), so that VerifyReceipt can confirm during polling that a
//     receipt belongs to the voter's latest ballot. A payload prepared before a
//     re-vote must not overwrite the newer ballot's record.
//  2. The kiosk opens the view envelope ve carried in the printed receipt,
//     re-derives the ciphertext from the opening (j, r_s), and shows option j
//     only when SHA256 of the re-derived ciphertext equals the hC that the
//     ledger holds for the voter's current ballot. The display is therefore
//     checked against the ledger and is not a repetition of the voting
//     terminal's own claim.
//
// The envelope is sealed to an election view key with an HPKE-style construction
// (X25519, HKDF-SHA256, AES-256-GCM), so the terminal that creates it holds no
// secret. The contract never receives the envelope.
package main

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdh"
	"crypto/hkdf"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math/big"
	"strings"
	"testing"
)

const viewEnvLabel = "VIEW_ENV_V1"

// voterReceipt is the printed receipt: the RecordVote acknowledgement plus ve.
type voterReceipt struct {
	TxID   string
	Serial string
	HC     string
	VE     string
}

type viewOpening struct {
	J int    `json:"j"`
	R string `json:"r"` // Paillier randomness r_s, hex
}

// viewAEAD derives the envelope cipher from the X25519 shared secret.
func viewAEAD(t *testing.T, shared, ephPub, recipPub []byte) cipher.AEAD {
	t.Helper()
	salt := append(append([]byte{}, ephPub...), recipPub...)
	k, err := hkdf.Key(sha256.New, shared, salt, viewEnvLabel, 32)
	requireNoErr(t, err)
	block, err := aes.NewCipher(k)
	requireNoErr(t, err)
	gcm, err := cipher.NewGCM(block)
	requireNoErr(t, err)
	return gcm
}

// sealView is what the voting terminal does after encrypting the ballot. The
// associated data binds the envelope to hC, so it cannot be moved to another
// receipt.
func sealView(t *testing.T, viewPub *ecdh.PublicKey, hC string, j int, r int64) string {
	t.Helper()
	eph, err := ecdh.X25519().GenerateKey(rand.Reader)
	requireNoErr(t, err)
	shared, err := eph.ECDH(viewPub)
	requireNoErr(t, err)
	gcm := viewAEAD(t, shared, eph.PublicKey().Bytes(), viewPub.Bytes())
	nonce := make([]byte, gcm.NonceSize())
	_, err = rand.Read(nonce)
	requireNoErr(t, err)
	pt, _ := json.Marshal(viewOpening{J: j, R: fmt.Sprintf("%x", r)})
	ct := gcm.Seal(nil, nonce, pt, []byte(viewEnvLabel+"|"+hC))
	out := append(append(eph.PublicKey().Bytes(), nonce...), ct...)
	return base64.RawURLEncoding.EncodeToString(out)
}

// openView is what the view service does with the view key.
func openView(t *testing.T, viewPriv *ecdh.PrivateKey, hC, ve string) (viewOpening, error) {
	t.Helper()
	raw, err := base64.RawURLEncoding.DecodeString(ve)
	if err != nil || len(raw) < 32+12 {
		return viewOpening{}, fmt.Errorf("malformed envelope")
	}
	ephPub, err := ecdh.X25519().NewPublicKey(raw[:32])
	if err != nil {
		return viewOpening{}, err
	}
	shared, err := viewPriv.ECDH(ephPub)
	if err != nil {
		return viewOpening{}, err
	}
	gcm := viewAEAD(t, shared, raw[:32], viewPriv.PublicKey().Bytes())
	pt, err := gcm.Open(nil, raw[32:32+12], raw[32+12:], []byte(viewEnvLabel+"|"+hC))
	if err != nil {
		return viewOpening{}, fmt.Errorf("envelope does not open for this hC")
	}
	var o viewOpening
	if err := json.Unmarshal(pt, &o); err != nil {
		return viewOpening{}, err
	}
	return o, nil
}

// kioskView is the reference kiosk check. It returns the option index to
// display, or a reason for not displaying one. Order matters: the receipt must
// be the voter's current ballot before the envelope is opened at all.
func kioskView(t *testing.T, h *testHarness, key *paillierTestKey, viewPriv *ecdh.PrivateKey, slotBits int, rc voterReceipt) (int, string) {
	t.Helper()

	vr, err := h.cc.VerifyReceipt(h.ctx, rc.TxID, rc.HC)
	requireNoErr(t, err)
	var res struct {
		OK         bool   `json:"ok"`
		Reason     string `json:"reason"`
		Superseded bool   `json:"superseded"`
	}
	requireNoErr(t, json.Unmarshal([]byte(vr), &res))
	if res.Superseded {
		return -1, "superseded"
	}
	if !res.OK {
		if res.Reason == "unknown_tx" {
			// Not indexed yet, or never current. The kiosk tells the two apart
			// through the serial, as for exclusions.
			bm, err := h.cc.GetBallotBySerial(h.ctx, rc.Serial)
			if err == nil && bm.TxID != rc.TxID {
				return -1, "superseded"
			}
			if err == nil && bm.Status == "pending" {
				return -1, "pending"
			}
		}
		return -1, res.Reason
	}

	o, err := openView(t, viewPriv, rc.HC, rc.VE)
	if err != nil {
		return -1, "no_view"
	}
	r, ok := new(big.Int).SetString(o.R, 16)
	if !ok || !r.IsInt64() {
		return -1, "no_view"
	}
	v, err := PackedSlotValue(o.J, slotBits)
	if err != nil {
		return -1, "no_view"
	}
	if sha256HexOf(key.encValueHex(v, r.Int64())) != strings.ToLower(rc.HC) {
		// The opening does not re-derive the recorded ballot: mismatch report.
		return -1, "mismatch"
	}
	return o.J, "ok"
}

// castWithView casts a packed ballot for encryptedJ and returns the printed
// receipt whose envelope claims sealedJ. An honest terminal passes the same
// value twice.
func castWithView(t *testing.T, h *testHarness, key *paillierTestKey, viewPub *ecdh.PublicKey,
	serial, txID string, encryptedJ, sealedJ int, r int64, slotBits int) voterReceipt {
	t.Helper()
	h.setTxID(txID)
	ack, err := h.recordVote(serial, PackedSentinel, key.packedBallotHex(t, encryptedJ, slotBits, r))
	requireNoErr(t, err)
	hC := hcFromRecordVote(t, ack)
	return voterReceipt{TxID: txID, Serial: serial, HC: hC, VE: sealView(t, viewPub, hC, sealedJ, r)}
}

func newViewKey(t *testing.T) *ecdh.PrivateKey {
	t.Helper()
	k, err := ecdh.X25519().GenerateKey(rand.Reader)
	requireNoErr(t, err)
	return k
}

func setupPackedPoll(t *testing.T) (*testHarness, *paillierTestKey) {
	t.Helper()
	setDefaultEnv(t)
	h := newHarness(t)
	key := newPackedTestKey(t)
	h.stubPreloadCandidatesOnly([]string{PackedSentinel})
	requireNoErr(t, h.seedCandidates([]string{PackedSentinel}))
	setPK_Custom(t, h, key.n)
	requireNoErr(t, h.openPoll())
	return h, key
}

func applyCurrent(t *testing.T, h *testHarness, serial, txID string) error {
	t.Helper()
	return h.cc.ApplyBallotStatuses(h.ctx, testConst,
		fmt.Sprintf(`{"current":[{"serial":%q,"txID":%q}],"invalid":[]}`, serial, txID))
}

/* ---------- rolling ApplyBallotStatuses ---------- */

// TestRollingStatus_SkipsEntryOvertakenByRevote prepares a status payload for a
// ballot, records a re-vote, and only then applies the payload. Before the
// change the payload wrote the old txID over the new ballot's record, so the
// voter's latest receipt could not be verified. It must now be skipped.
func TestRollingStatus_SkipsEntryOvertakenByRevote(t *testing.T) {
	h, key := setupPackedPoll(t)
	defer h.ctrl.Finish()
	const slotBits = 8

	h.setTxID("tx-1")
	ack1, err := h.recordVote("S-001", PackedSentinel, key.packedBallotHex(t, 1, slotBits, 37))
	requireNoErr(t, err)
	hc1 := hcFromRecordVote(t, ack1)

	h.setTxID("tx-2")
	ack2, err := h.recordVote("S-001", PackedSentinel, key.packedBallotHex(t, 2, slotBits, 41))
	requireNoErr(t, err)
	hc2 := hcFromRecordVote(t, ack2)

	// The payload naming tx-1 arrives after tx-2 was recorded.
	requireNoErr(t, applyCurrent(t, h, "S-001", "tx-1"))

	bm, err := h.cc.GetBallotBySerial(h.ctx, "S-001")
	requireNoErr(t, err)
	if bm.TxID != "tx-2" || bm.HC != hc2 || bm.Status != "pending" {
		t.Fatalf("stale entry altered the latest ballot: %+v", bm)
	}
	if _, ok := h.mem.ws[keyTxIdxPrefix+"tx-1"]; ok {
		t.Fatalf("stale entry was indexed in TXIDX")
	}

	// The next round covers tx-2, and the latest receipt then verifies.
	requireNoErr(t, applyCurrent(t, h, "S-001", "tx-2"))
	vr, err := h.cc.VerifyReceipt(h.ctx, "tx-2", hc2)
	requireNoErr(t, err)
	if !strings.Contains(vr, `"ok":true`) || !strings.Contains(vr, `"superseded":false`) {
		t.Fatalf("latest receipt should verify during polling; got %s", vr)
	}
	vr, err = h.cc.VerifyReceipt(h.ctx, "tx-1", hc1)
	requireNoErr(t, err)
	if strings.Contains(vr, `"ok":true`) {
		t.Fatalf("earlier receipt must not verify; got %s", vr)
	}
}

// TestRollingStatus_RefusesCiphertextAndMissingTxID checks the two payload
// rules that apply only while the poll is open, and that the final run after
// close still publishes the ciphertext as before.
func TestRollingStatus_RefusesCiphertextAndMissingTxID(t *testing.T) {
	h, key := setupPackedPoll(t)
	defer h.ctrl.Finish()
	const slotBits = 8

	h.setTxID("tx-1")
	ballot := key.packedBallotHex(t, 1, slotBits, 37)
	_, err := h.recordVote("S-001", PackedSentinel, ballot)
	requireNoErr(t, err)

	withCipher := fmt.Sprintf(`{"current":[{"serial":"S-001","txID":"tx-1","encOneHex":%q}],"invalid":[]}`, ballot)
	requireErrContains(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, withCipher), "only after the poll closes")

	noTx := `{"current":[],"invalid":[{"serial":"S-001","reason":"booth"}]}`
	requireErrContains(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, noTx), "must carry txID")

	bm, err := h.cc.GetBallotBySerial(h.ctx, "S-001")
	requireNoErr(t, err)
	if bm.EncOneHex != "" || bm.Status != "pending" {
		t.Fatalf("refused payloads changed the record: %+v", bm)
	}

	requireNoErr(t, h.closePoll())
	requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, withCipher))
	bm, err = h.cc.GetBallotBySerial(h.ctx, "S-001")
	requireNoErr(t, err)
	if bm.Status != "current" || bm.EncOneHex == "" {
		t.Fatalf("final run after close should publish the ciphertext: %+v", bm)
	}
}

/* ---------- kiosk view ---------- */

// TestKioskView_ShowsLatestBallotOnly follows one voter through a vote, a
// re-vote and two kiosk visits during polling.
func TestKioskView_ShowsLatestBallotOnly(t *testing.T) {
	h, key := setupPackedPoll(t)
	defer h.ctrl.Finish()
	const slotBits = 8
	viewPriv := newViewKey(t)
	viewPub := viewPriv.PublicKey()

	rc1 := castWithView(t, h, key, viewPub, "S-001", "tx-1", 1, 1, 37, slotBits)

	// Before reconciliation the kiosk says "not final yet" and shows nothing.
	if j, why := kioskView(t, h, key, viewPriv, slotBits, rc1); why != "pending" {
		t.Fatalf("before reconciliation: got (%d, %s), want pending", j, why)
	}

	requireNoErr(t, applyCurrent(t, h, "S-001", "tx-1"))
	if j, why := kioskView(t, h, key, viewPriv, slotBits, rc1); why != "ok" || j != 1 {
		t.Fatalf("first visit: got (%d, %s), want option 1", j, why)
	}

	// Re-vote for option 3. The earlier receipt no longer shows anything.
	rc2 := castWithView(t, h, key, viewPub, "S-001", "tx-2", 3, 3, 41, slotBits)
	if _, why := kioskView(t, h, key, viewPriv, slotBits, rc1); why != "superseded" {
		t.Fatalf("earlier receipt after re-vote: got %s, want superseded", why)
	}
	if _, why := kioskView(t, h, key, viewPriv, slotBits, rc2); why != "pending" {
		t.Fatalf("new receipt before reconciliation: got %s, want pending", why)
	}

	requireNoErr(t, applyCurrent(t, h, "S-001", "tx-2"))
	if j, why := kioskView(t, h, key, viewPriv, slotBits, rc2); why != "ok" || j != 3 {
		t.Fatalf("second visit: got (%d, %s), want option 3", j, why)
	}
	if _, why := kioskView(t, h, key, viewPriv, slotBits, rc1); why != "superseded" {
		t.Fatalf("earlier receipt after reconciliation: got %s, want superseded", why)
	}

	// An envelope copied onto another receipt does not open.
	moved := rc2
	moved.VE = rc1.VE
	if _, why := kioskView(t, h, key, viewPriv, slotBits, moved); why != "no_view" {
		t.Fatalf("envelope moved between receipts: got %s, want no_view", why)
	}

	// The envelope never reaches ledger state.
	for k, v := range h.mem.ws {
		if strings.Contains(string(v), rc2.VE) {
			t.Fatalf("view envelope found in world state under %s", k)
		}
	}
	for k, v := range h.mem.pdc[votesPDC] {
		if strings.Contains(string(v), rc2.VE) {
			t.Fatalf("view envelope found in votes_pdc under %s", k)
		}
	}
}

// TestKioskView_DetectsTerminalThatSealsAnotherOption covers a voting terminal
// that encrypts option 2 for a voter who chose option 1. If it seals the
// voter's option, the re-derivation fails and the kiosk reports a mismatch. If
// it seals the option it encrypted, the kiosk shows option 2 and the voter sees
// that it is not the candidate chosen.
func TestKioskView_DetectsTerminalThatSealsAnotherOption(t *testing.T) {
	h, key := setupPackedPoll(t)
	defer h.ctrl.Finish()
	const slotBits = 8
	viewPriv := newViewKey(t)

	lying := castWithView(t, h, key, viewPriv.PublicKey(), "S-001", "tx-1", 2, 1, 37, slotBits)
	requireNoErr(t, applyCurrent(t, h, "S-001", "tx-1"))
	if _, why := kioskView(t, h, key, viewPriv, slotBits, lying); why != "mismatch" {
		t.Fatalf("terminal sealing the voter's option: got %s, want mismatch", why)
	}

	candid := castWithView(t, h, key, viewPriv.PublicKey(), "S-002", "tx-2", 2, 2, 41, slotBits)
	requireNoErr(t, applyCurrent(t, h, "S-002", "tx-2"))
	if j, why := kioskView(t, h, key, viewPriv, slotBits, candid); why != "ok" || j != 2 {
		t.Fatalf("terminal sealing the encrypted option: got (%d, %s), want option 2 shown", j, why)
	}
}
