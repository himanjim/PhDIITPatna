// verifiability_test.go exercises the three end-to-end verifiability links at
// the contract level.
//
//	Link 1 - cast as intended  : RecordAuditOpening / ExportAuditOpenings plus the
//	                             cast-or-audit coin, reconciled against the cast set.
//	Link 2 - recorded as cast  : the receipt anchor hC survives status transitions,
//	                             and a substituted ciphertext is refused.
//	Link 3 - tallied as recorded: the encrypted tally is recomputable from PUBLIC
//	                             world state alone, with no votes_pdc access.
//
// The third test is the important one. Before these changes the ballot ciphertext
// lived only in the private data collection and public state held just its hash,
// so no member of the public could reproduce the aggregate; verification was
// available to authorised auditors only. TestVerifiability_PublicRecomputationOfTally
// demonstrates that this is no longer the case.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"strings"
	"testing"
)

/* ---------- toy Paillier helpers sized for packed encoding ---------- */

// nextPrimeAbove returns the first probable prime strictly greater than start.
// It keeps the packed-encoding tests deterministic without hard-coding magic
// primes whose primality a later reader would have to take on trust.
func nextPrimeAbove(start int64) *big.Int {
	p := big.NewInt(start)
	one := big.NewInt(1)
	for {
		p.Add(p, one)
		if p.ProbablyPrime(32) {
			return new(big.Int).Set(p)
		}
	}
}

// newPackedTestKey builds a toy Paillier key whose modulus is wide enough to
// carry several packed slots. It is cryptographically insecure and exists only
// so that the packed aggregation path can be exercised end to end.
//
// Two ~21-bit primes give a ~42-bit modulus, which at 8-bit slots leaves room
// for four option slots after the reserved headroom slot.
func newPackedTestKey(t *testing.T) *paillierTestKey {
	t.Helper()
	p := nextPrimeAbove(2_000_000)
	q := nextPrimeAbove(2_100_000)
	k := newPaillierTestKey(p.Int64(), q.Int64())
	if k.mu == nil {
		t.Fatalf("toy key construction failed: no modular inverse for mu")
	}
	return k
}

// encValueHex returns Enc(m; r) = g^m * r^n mod n^2 as hex, for the toy key.
// This generalises the existing encOneHex helper, which is fixed at m = 1.
func (k *paillierTestKey) encValueHex(m *big.Int, r int64) string {
	g := new(big.Int).Add(k.n, big.NewInt(1))
	gm := new(big.Int).Exp(g, m, k.n2)
	rn := new(big.Int).Exp(big.NewInt(r), k.n, k.n2)
	c := new(big.Int).Mul(gm, rn)
	c.Mod(c, k.n2)
	return fmt.Sprintf("%x", c)
}

// packedBallotHex encrypts a vote for optionIndex under the packed encoding.
func (k *paillierTestKey) packedBallotHex(t *testing.T, optionIndex, slotBits int, r int64) string {
	t.Helper()
	v, err := PackedSlotValue(optionIndex, slotBits)
	requireNoErr(t, err)
	return k.encValueHex(v, r)
}

func sha256HexOf(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// hcFromRecordVote pulls the receipt commitment out of the RecordVote
// acknowledgement, which is what the QR receipt carries.
func hcFromRecordVote(t *testing.T, ack string) string {
	t.Helper()
	var m map[string]string
	requireNoErr(t, json.Unmarshal([]byte(ack), &m))
	if m["hC"] == "" {
		t.Fatalf("RecordVote acknowledgement carried no hC: %s", ack)
	}
	return m["hC"]
}

/* ---------- Link 3: tallied as recorded, by the public ---------- */

// TestVerifiability_PublicRecomputationOfTally is the universal-verifiability
// test: it recomputes the encrypted tally using nothing but public world state.
//
// The sequence mirrors a real election:
//   - ballots are cast under the packed encoding, so every record carries the
//     same constant candidate label and the choice lives inside the ciphertext;
//   - the tally is prepared from private data, as it is today;
//   - the off-chain pipeline publishes the ciphertext set through
//     ApplyBallotStatuses, which accepts it only if it hashes to the commitment
//     RecordVote already made;
//   - an observer exports the public ballot set, re-derives every commitment,
//     multiplies the ciphertexts, and obtains the same aggregate the contract
//     produced from the private collection.
//
// The final decryption step then reads off the per-option counts, which is what
// a trustee quorum would do with a threshold key.
func TestVerifiability_PublicRecomputationOfTally(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	const slotBits = 8
	const numSlots = 4
	key := newPackedTestKey(t)

	if cap := PackedCapacity(key.n.BitLen(), slotBits); cap < numSlots {
		t.Fatalf("toy modulus of %d bits only carries %d slots, need %d", key.n.BitLen(), cap, numSlots)
	}

	// Under packed encoding the tally has exactly one bucket.
	h.stubPreloadCandidatesOnly([]string{PackedSentinel})
	requireNoErr(t, h.seedCandidates([]string{PackedSentinel}))
	setPK_Custom(t, h, key.n)
	requireNoErr(t, h.cc.SetParams(h.ctx, fmt.Sprintf(
		`{"PACKED_MODE":true,"PACKED_SLOT_BITS":%d,"PACKED_SLOTS":%d}`, slotBits, numSlots)))
	requireNoErr(t, h.openPoll())

	// Cast a known distribution: option 0 x3, option 1 x1, option 2 x2, option 3 x0.
	ballots := []struct {
		serial string
		option int
		r      int64
	}{
		{"S-001", 0, 7}, {"S-002", 0, 11}, {"S-003", 0, 13},
		{"S-004", 1, 17},
		{"S-005", 2, 19}, {"S-006", 2, 23},
	}
	want := []uint64{3, 1, 2, 0}

	type published struct{ serial, hC, enc string }
	var pub []published

	for i, b := range ballots {
		h.setTxID(fmt.Sprintf("tx-packed-%03d", i))
		enc := key.packedBallotHex(t, b.option, slotBits, b.r)
		ack, err := h.recordVote(b.serial, PackedSentinel, enc)
		requireNoErr(t, err)
		pub = append(pub, published{serial: b.serial, hC: hcFromRecordVote(t, ack), enc: enc})
	}
	requireNoErr(t, h.closePoll())

	// Contract-side tally from the private collection, exactly as today.
	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)
	contractAgg := mustBigFromHex(t, sums[PackedSentinel])

	// The off-chain pipeline publishes the ciphertext set into public state.
	var cur []string
	for _, p := range pub {
		cur = append(cur, fmt.Sprintf(`{"serial":%q,"txID":%q,"encOneHex":%q}`,
			p.serial, "tx-final-"+p.serial, p.enc))
	}
	statusJSON := fmt.Sprintf(`{"current":[%s],"invalid":[]}`, strings.Join(cur, ","))
	requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, statusJSON))

	/* ---- everything below uses PUBLIC state only ---- */

	exported, err := h.cc.ExportPublicBallots(h.ctx)
	requireNoErr(t, err)
	var rows []PublicBallot
	requireNoErr(t, json.Unmarshal([]byte(exported), &rows))
	if len(rows) != len(ballots) {
		t.Fatalf("exported %d public ballots, want %d", len(rows), len(ballots))
	}

	// The export must be sorted by serial: this is the ordering the freeze
	// commitment HR relies on, so an observer can hash it directly.
	for i := 1; i < len(rows); i++ {
		if rows[i-1].Serial >= rows[i].Serial {
			t.Fatalf("public ballot export is not sorted by serial: %q then %q",
				rows[i-1].Serial, rows[i].Serial)
		}
	}

	publicAgg := big.NewInt(1)
	for _, row := range rows {
		if row.EncOneHex == "" {
			t.Fatalf("serial %s has no published ciphertext; public tally is impossible", row.Serial)
		}
		// Bind the published ciphertext to the commitment made at cast time.
		if got := sha256HexOf(row.EncOneHex); !strings.EqualFold(got, row.HC) {
			t.Fatalf("serial %s: published ciphertext hashes to %s but hC is %s", row.Serial, got, row.HC)
		}
		c := mustBigFromHex(t, row.EncOneHex)
		publicAgg = mulMod(publicAgg, c, key.n2)
	}

	if publicAgg.Cmp(contractAgg) != 0 {
		t.Fatalf("aggregate recomputed from public state (%s) differs from TallyPrepare (%s)",
			publicAgg.Text(16), contractAgg.Text(16))
	}

	// Read off the per-option counts, as a trustee quorum would after decryption.
	plain := key.decCountT(t, publicAgg.Text(16))
	counts, err := PackedDecodeChecked(plain, slotBits, numSlots, uint64(len(ballots)))
	requireNoErr(t, err)
	for i := range want {
		if counts[i] != want[i] {
			t.Fatalf("option %d = %d, want %d (all counts %v)", i, counts[i], want[i], counts)
		}
	}
	t.Logf("tally recomputed from public state only: %v", counts)
}

// TestVerifiability_PackedCiphertextHidesTheChoice checks that two ballots for
// different options are indistinguishable in public state, which is what makes
// publishing the ciphertext set safe.
func TestVerifiability_PackedCiphertextHidesTheChoice(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	const slotBits = 8
	key := newPackedTestKey(t)

	h.stubPreloadCandidatesOnly([]string{PackedSentinel})
	requireNoErr(t, h.seedCandidates([]string{PackedSentinel}))
	setPK_Custom(t, h, key.n)
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-hide-1")
	encA := key.packedBallotHex(t, 0, slotBits, 29)
	_, err := h.recordVote("S-A", PackedSentinel, encA)
	requireNoErr(t, err)

	h.setTxID("tx-hide-2")
	encB := key.packedBallotHex(t, 3, slotBits, 31)
	_, err = h.recordVote("S-B", PackedSentinel, encB)
	requireNoErr(t, err)

	// Public metadata must not distinguish the two choices in any field.
	for _, s := range []string{"S-A", "S-B"} {
		bm, err := h.cc.GetBallotBySerial(h.ctx, s)
		requireNoErr(t, err)
		blob := string(mustJSON(bm))
		if strings.Contains(blob, "cand-") {
			t.Fatalf("public ballot meta for %s leaks a candidate label: %s", s, blob)
		}
	}

	// The private record must carry only the constant sentinel, not a choice.
	raw, err := h.mem.getPDC(votesPDC, voteKey(testConst, "S-A"))
	requireNoErr(t, err)
	var vm VoteMetaPDC
	requireNoErr(t, json.Unmarshal(raw, &vm))
	if vm.CandidateID != PackedSentinel {
		t.Fatalf("private record candidateID = %q, want the constant %q; "+
			"a per-candidate label would expose the choice to every collection member",
			vm.CandidateID, PackedSentinel)
	}
}

/* ---------- Link 2: recorded as cast ---------- */

// TestVerifiability_RejectsSubstitutedCiphertext is the check that stops the
// publication step from being circular.
//
// The publisher supplies the ciphertext set, so without binding it to the
// commitment written inside the endorsed RecordVote transaction it could publish
// any ciphertexts it liked and the "public" tally would prove nothing.
func TestVerifiability_RejectsSubstitutedCiphertext(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	const slotBits = 8
	key := newPackedTestKey(t)

	h.stubPreloadCandidatesOnly([]string{PackedSentinel})
	requireNoErr(t, h.seedCandidates([]string{PackedSentinel}))
	setPK_Custom(t, h, key.n)
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-sub-1")
	honest := key.packedBallotHex(t, 1, slotBits, 37)
	_, err := h.recordVote("S-001", PackedSentinel, honest)
	requireNoErr(t, err)
	requireNoErr(t, h.closePoll())

	// A ciphertext for a different option, validly formed but never cast.
	substituted := key.packedBallotHex(t, 2, slotBits, 41)
	bad := fmt.Sprintf(`{"current":[{"serial":"S-001","txID":"tx-x","encOneHex":%q}],"invalid":[]}`, substituted)
	err = h.cc.ApplyBallotStatuses(h.ctx, testConst, bad)
	requireErrContains(t, err, "committed hc")

	// The honest ciphertext for the same serial must still be accepted.
	good := fmt.Sprintf(`{"current":[{"serial":"S-001","txID":"tx-x","encOneHex":%q}],"invalid":[]}`, honest)
	requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, good))

	bm, err := h.cc.GetBallotBySerial(h.ctx, "S-001")
	requireNoErr(t, err)
	if !strings.EqualFold(bm.HC, sha256HexOf(honest)) {
		t.Fatalf("receipt anchor = %s, want %s", bm.HC, sha256HexOf(honest))
	}
}

// TestVerifiability_InvalidationPreservesReceiptAnchor covers a latent defect in
// the original markBallotStatus: it recomputed HC = sha256(encOneHex) on every
// status write, so the "invalid" path - which supplies no ciphertext - replaced
// the receipt anchor with sha256("").
//
// A voter checking a perfectly valid receipt for an excluded ballot would then
// see "receipt_mismatch" rather than a truthful invalid status, and would have no
// way to tell an exclusion from a tampered record.
func TestVerifiability_InvalidationPreservesReceiptAnchor(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	const slotBits = 8
	key := newPackedTestKey(t)

	h.stubPreloadCandidatesOnly([]string{PackedSentinel})
	requireNoErr(t, h.seedCandidates([]string{PackedSentinel}))
	setPK_Custom(t, h, key.n)
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-inv-1")
	enc := key.packedBallotHex(t, 0, slotBits, 43)
	ack, err := h.recordVote("S-001", PackedSentinel, enc)
	requireNoErr(t, err)
	receipt := hcFromRecordVote(t, ack)

	// Second ballot is cast while the poll is still open; it is excluded below
	// with no stated reason, to check the default reason is recorded.
	h.setTxID("tx-inv-2")
	enc2 := key.packedBallotHex(t, 1, slotBits, 47)
	_, err = h.recordVote("S-002", PackedSentinel, enc2)
	requireNoErr(t, err)

	requireNoErr(t, h.closePoll())

	// Exclude the ballot, giving a reason, and supplying no ciphertext.
	statusJSON := `{"current":[],"invalid":[{"serial":"S-001","txID":"tx-inv-1","reason":"no valid 1-of-m proof"}]}`
	requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst, statusJSON))

	bm, err := h.cc.GetBallotBySerial(h.ctx, "S-001")
	requireNoErr(t, err)
	if bm.Status != "invalid" {
		t.Fatalf("status = %q, want invalid", bm.Status)
	}
	if !strings.EqualFold(bm.HC, receipt) {
		t.Fatalf("receipt anchor was destroyed by invalidation: got %s, want %s", bm.HC, receipt)
	}
	if bm.Reason != "no valid 1-of-m proof" {
		t.Fatalf("exclusion reason = %q, want it recorded in public state", bm.Reason)
	}

	// An exclusion with no stated reason must still be marked, not left blank.
	requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst,
		`{"current":[],"invalid":[{"serial":"S-002","txID":"tx-inv-2"}]}`))
	bm2, err := h.cc.GetBallotBySerial(h.ctx, "S-002")
	requireNoErr(t, err)
	if bm2.Reason != "unspecified" {
		t.Fatalf("missing reason = %q, want %q", bm2.Reason, "unspecified")
	}
}

/* ---------- packed-mode enforcement ---------- */

// TestVerifiability_PackedModeExcludesLegacyLabel checks that once packed mode is
// enabled, a record still carrying a per-candidate plaintext label is excluded
// from the tally rather than silently counted.
//
// Without this guard a ledger could end up half-migrated, with some ballots
// leaking their choice to collection members and the public aggregate quietly
// missing them.
func TestVerifiability_PackedModeExcludesLegacyLabel(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	const slotBits = 8
	key := newPackedTestKey(t)

	// Both labels are seeded so the exclusion is attributable to packed mode
	// alone and not to an unknown-candidate rejection.
	h.stubPreloadCandidatesOnly([]string{PackedSentinel, testCand1})
	requireNoErr(t, h.seedCandidates([]string{PackedSentinel, testCand1}))
	setPK_Custom(t, h, key.n)
	requireNoErr(t, h.openPoll())

	h.setTxID("tx-legacy-1")
	legacy := key.encValueHex(big.NewInt(1), 53) // old-style Enc(1)
	_, err := h.recordVote("S-legacy", testCand1, legacy)
	requireNoErr(t, err)

	h.setTxID("tx-packed-1")
	packed := key.packedBallotHex(t, 0, slotBits, 59)
	_, err = h.recordVote("S-packed", PackedSentinel, packed)
	requireNoErr(t, err)
	requireNoErr(t, h.closePoll())

	// Packed mode off: the legacy ballot is still counted into its own bucket.
	sums, err := h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)
	if sums[testCand1] == "1" {
		t.Fatal("with packed mode off the legacy ballot should still be aggregated")
	}

	// Packed mode on: the legacy ballot is excluded.
	requireNoErr(t, h.cc.SetParams(h.ctx, `{"PACKED_MODE":true}`))
	sums, err = h.cc.TallyPrepare(h.ctx, testConst)
	requireNoErr(t, err)
	if sums[testCand1] != "1" {
		t.Fatalf("legacy-labelled ballot must be excluded in packed mode, bucket = %q", sums[testCand1])
	}
	if sums[PackedSentinel] == "1" {
		t.Fatal("packed ballot must still be aggregated in packed mode")
	}
}

/* ---------- Link 1: cast as intended ---------- */

// TestVerifiability_AuditOpeningRecordAndReconcile walks the cast-or-audit path.
//
// A ballot the coin selects is opened rather than cast: the client publishes the
// option index and the Paillier randomness, and anyone can recompute the
// ciphertext and confirm it hashes to the commitment the voter already saw. The
// opened ballot never reaches RecordVote, so it never enters the tally.
func TestVerifiability_AuditOpeningRecordAndReconcile(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	const slotBits = 8
	const oneIn = 4
	key := newPackedTestKey(t)
	keyDay := testAuditKey("contract-level")

	h.stubPreloadCandidatesOnly([]string{PackedSentinel})
	requireNoErr(t, h.seedCandidates([]string{PackedSentinel}))
	setPK_Custom(t, h, key.n)
	requireNoErr(t, h.openPoll())

	commit, err := AuditKeyCommitment(keyDay)
	requireNoErr(t, err)
	requireNoErr(t, h.cc.SetParams(h.ctx, fmt.Sprintf(
		`{"PACKED_MODE":true,"PACKED_SLOT_BITS":%d,"AUDIT_ONE_IN":%d,"AUDIT_KEY_COMMIT":%q}`,
		slotBits, oneIn, commit)))

	// The published audit parameters must be readable from the ledger.
	params, err := h.cc.GetParams(h.ctx)
	requireNoErr(t, err)
	if params.AuditOneIn != oneIn || params.AuditKeyCommit != commit {
		t.Fatalf("audit parameters not published: oneIn=%d commit=%q", params.AuditOneIn, params.AuditKeyCommit)
	}

	var castHCs, openedHCs []string
	opened := 0

	for i := 0; i < 40; i++ {
		serial := fmt.Sprintf("S-%03d", i)
		option := i % 4
		r := int64(61 + 2*i)

		// The client commits to the ciphertext, then asks for the coin.
		enc := key.packedBallotHex(t, option, slotBits, r)
		hC := sha256HexOf(enc)
		outcome, err := AuditOutcome(keyDay, hC, uint32(oneIn))
		requireNoErr(t, err)

		if outcome == AuditOutcomeOpen {
			// Opened: published, never cast.
			requireNoErr(t, h.cc.RecordAuditOpening(
				h.ctx, testConst, hC, option, fmt.Sprintf("%x", r), testBoothID, testDeviceID))
			openedHCs = append(openedHCs, hC)
			opened++
			continue
		}

		h.setTxID(fmt.Sprintf("tx-coin-%03d", i))
		ack, err := h.recordVote(serial, PackedSentinel, enc)
		requireNoErr(t, err)
		castHCs = append(castHCs, hcFromRecordVote(t, ack))
	}

	if opened == 0 {
		t.Fatal("fixture produced no audited ballots; the coin never fired")
	}
	t.Logf("%d of 40 ballots opened at a 1-in-%d rate", opened, oneIn)

	// An observer repeats the kiosk's check from the published opening alone.
	exported, err := h.cc.ExportAuditOpenings(h.ctx)
	requireNoErr(t, err)
	var log []AuditLogEntry
	requireNoErr(t, json.Unmarshal([]byte(exported), &log))
	if len(log) != opened {
		t.Fatalf("audit log holds %d entries, want %d", len(log), opened)
	}
	for _, e := range log {
		r := new(big.Int)
		if _, ok := r.SetString(e.RandomnessHex, 16); !ok {
			t.Fatalf("audit entry %s: unparseable randomness %q", e.HC, e.RandomnessHex)
		}
		recomputed := key.packedBallotHex(t, e.OptionIndex, slotBits, r.Int64())
		if !strings.EqualFold(sha256HexOf(recomputed), e.HC) {
			t.Fatalf("audit entry %s does not open to its commitment", e.HC)
		}
	}

	// No opened ballot may appear in the tally universe.
	rows := publicBallotRows(t, h)
	castSet := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		castSet[strings.ToLower(row.HC)] = struct{}{}
	}
	for _, hC := range openedHCs {
		if _, found := castSet[strings.ToLower(hC)]; found {
			t.Fatalf("opened ballot %s also appears in the cast set", hC)
		}
	}

	// Once K_day is released, the whole election reconciles cleanly.
	requireNoErr(t, VerifyAuditKeyCommitment(keyDay, params.AuditKeyCommit))
	shouldHaveOpened, openedWithoutCause, err := ReconcileAuditLog(keyDay, uint32(oneIn), castHCs, openedHCs)
	requireNoErr(t, err)
	if len(shouldHaveOpened) != 0 || len(openedWithoutCause) != 0 {
		t.Fatalf("honest run must reconcile cleanly: ignored=%v unjustified=%v",
			shouldHaveOpened, openedWithoutCause)
	}
}

// publicBallotRows is a small helper that exports and parses the public ballot
// set, used wherever a test needs to reason as an outside observer would.
func publicBallotRows(t *testing.T, h *testHarness) []PublicBallot {
	t.Helper()
	exported, err := h.cc.ExportPublicBallots(h.ctx)
	requireNoErr(t, err)
	var rows []PublicBallot
	requireNoErr(t, json.Unmarshal([]byte(exported), &rows))
	return rows
}

// TestVerifiability_AuditOpeningGuards checks the input validation on the audit
// publication path, including the refusal to record a second opening for the
// same commitment.
func TestVerifiability_AuditOpeningGuards(t *testing.T) {
	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	hC := sha256HexOf("some-ciphertext")

	requireErrContains(t, h.cc.RecordAuditOpening(h.ctx, testConst, "  ", 0, "ff", "", ""), "empty commitment")
	requireErrContains(t, h.cc.RecordAuditOpening(h.ctx, testConst, hC, -1, "ff", "", ""), "option index")
	requireErrContains(t, h.cc.RecordAuditOpening(h.ctx, testConst, hC, MaxPackedSlots, "ff", "", ""), "option index")
	requireErrContains(t, h.cc.RecordAuditOpening(h.ctx, testConst, hC, 0, "  ", "", ""), "randomness")

	requireNoErr(t, h.cc.RecordAuditOpening(h.ctx, testConst, hC, 2, "beef", testBoothID, testDeviceID))
	requireErrContains(t, h.cc.RecordAuditOpening(h.ctx, testConst, hC, 2, "beef", "", ""), "already recorded")

	got, err := h.cc.GetAuditOpening(h.ctx, strings.ToUpper(hC))
	requireNoErr(t, err)
	if got.OptionIndex != 2 || got.BoothID != testBoothID || got.DeviceID != testDeviceID {
		t.Fatalf("audit opening round-trip mismatch: %+v", got)
	}

	if _, err := h.cc.GetAuditOpening(h.ctx, sha256HexOf("never-opened")); err == nil {
		t.Fatal("expected an error for a commitment with no published opening")
	}
}
