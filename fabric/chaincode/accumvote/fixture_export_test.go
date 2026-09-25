// fixture_export_test.go writes the verification-pack fixture used by the
// public verifier tools in fabric/tools.
//
// It is not an assertion test. It runs a small but complete election against the
// contract - packed ballots, a few ballots opened by the cast-or-audit coin, an
// excluded ballot with its reason, the tally and the published ciphertext set -
// and writes the resulting PUBLIC material to disk in the shape that
// tools/export_freeze.js consumes. The point is that the verifier tools are
// exercised against data the contract itself produced, not against JSON written
// by hand to match them.
//
// The test skips unless FIXTURE_OUT names a directory:
//
//	FIXTURE_OUT=../../tools/fixtures go test -run TestFixture_ExportVerificationPack -count=1
package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixtureBallot is one ballot in the scripted election.
type fixtureBallot struct {
	serial string
	option int
	role   string // "cast", "open" or "invalid"
}

// searchRandomness finds a Paillier randomness for which the cast-or-audit coin
// gives the outcome the script asks for. Because the coin is a function of the
// commitment, and the commitment is a function of the ciphertext, this is the
// only way to script an election in which particular ballots are opened.
// The search starts from a different point for every ballot so that no two
// ballots share randomness, as they would not in an election.
func searchRandomness(t *testing.T, k *paillierTestKey, keyDay []byte, optionIndex, slotBits int, oneIn uint32, wantOpen bool, start int64) (string, string) {
	t.Helper()
	for r := start | 1; r < start+50000; r += 2 {
		enc := k.packedBallotHex(t, optionIndex, slotBits, r)
		hc := sha256HexOf(enc)
		open, err := AuditCoin(keyDay, hc, oneIn)
		requireNoErr(t, err)
		if open == wantOpen {
			return enc, fmt.Sprintf("%x", r)
		}
	}
	t.Fatalf("no randomness below the search bound gives coin open=%v", wantOpen)
	return "", ""
}

func TestFixture_ExportVerificationPack(t *testing.T) {
	outDir := os.Getenv("FIXTURE_OUT")
	if outDir == "" {
		t.Skip("set FIXTURE_OUT=<dir> to write the verification-pack fixture")
	}

	setDefaultEnv(t)
	h := newHarness(t)
	defer h.ctrl.Finish()

	const slotBits = 8
	const numSlots = 4
	const oneIn = 4
	key := newPackedTestKey(t)
	keyDay, err := hex.DecodeString(strings.Repeat("a7", 32))
	requireNoErr(t, err)
	commit, err := AuditKeyCommitment(keyDay)
	requireNoErr(t, err)

	h.stubPreloadCandidatesOnly([]string{PackedSentinel})
	requireNoErr(t, h.seedCandidates([]string{PackedSentinel}))
	setPK_Custom(t, h, key.n)
	requireNoErr(t, h.cc.SetParams(h.ctx, fmt.Sprintf(
		`{"PACKED_MODE":true,"PACKED_SLOT_BITS":%d,"PACKED_SLOTS":%d,"AUDIT_ONE_IN":%d,"AUDIT_KEY_COMMIT":%q}`,
		slotBits, numSlots, oneIn, commit)))
	requireNoErr(t, h.openPoll())

	script := []fixtureBallot{
		{"S-0001", 0, "cast"}, {"S-0002", 0, "cast"}, {"S-0003", 0, "cast"},
		{"S-0004", 1, "cast"}, {"S-0005", 1, "cast"},
		{"S-0006", 2, "cast"}, {"S-0007", 2, "cast"}, {"S-0008", 2, "cast"},
		{"S-0009", 3, "cast"}, {"S-0010", 3, "cast"},
		{"S-0011", 1, "invalid"},
		{"A-0001", 0, "open"}, {"A-0002", 2, "open"}, {"A-0003", 3, "open"},
	}

	type castRow struct{ serial, txID, enc, hC string }
	var cast []castRow
	var invalid []castRow
	var queryLog []string
	counts := make([]uint64, numSlots)

	for i, b := range script {
		wantOpen := b.role == "open"
		enc, rHex := searchRandomness(t, key, keyDay, b.option, slotBits, oneIn, wantOpen, int64(101+i*997))
		hc := sha256HexOf(enc)
		queryLog = append(queryLog, hc)

		if wantOpen {
			// An opened ballot is never submitted to RecordVote. Its opening is
			// published instead, and the voter votes again.
			requireNoErr(t, h.cc.RecordAuditOpening(h.ctx, testConst, hc, b.option, rHex, testBoothID, testDeviceID))
			continue
		}

		txID := fmt.Sprintf("tx-fixture-%03d", i)
		h.setTxID(txID)
		ack, err := h.recordVote(b.serial, PackedSentinel, enc)
		requireNoErr(t, err)
		if got := hcFromRecordVote(t, ack); !strings.EqualFold(got, hc) {
			t.Fatalf("contract commitment %s differs from the locally computed %s", got, hc)
		}
		row := castRow{serial: b.serial, txID: txID, enc: enc, hC: hc}
		if b.role == "invalid" {
			invalid = append(invalid, row)
			continue
		}
		cast = append(cast, row)
		counts[b.option]++
	}
	requireNoErr(t, h.closePoll())

	// Publish the ciphertext set and the exclusion, exactly as the off-chain
	// pipeline does after the freeze.
	var cur, bad []string
	for _, r := range cast {
		cur = append(cur, fmt.Sprintf(`{"serial":%q,"txID":%q,"encOneHex":%q}`, r.serial, r.txID, r.enc))
	}
	for _, r := range invalid {
		bad = append(bad, fmt.Sprintf(`{"serial":%q,"txID":%q,"reason":%q}`, r.serial, r.txID, "voter-roll: serial not present"))
	}
	requireNoErr(t, h.cc.ApplyBallotStatuses(h.ctx, testConst,
		fmt.Sprintf(`{"current":[%s],"invalid":[%s]}`, strings.Join(cur, ","), strings.Join(bad, ","))))

	// The aggregate over the published ciphertext set, and the plaintext a
	// trustee quorum would obtain from it.
	aggregate := big.NewInt(1)
	for _, r := range cast {
		aggregate = mulMod(aggregate, mustBigFromHex(t, r.enc), key.n2)
	}
	total := key.decCountT(t, aggregate.Text(16))
	decoded, err := PackedDecodeChecked(total, slotBits, numSlots, uint64(len(cast)))
	requireNoErr(t, err)
	for j := range counts {
		if decoded[j] != counts[j] {
			t.Fatalf("option %d decodes to %d, the script cast %d", j, decoded[j], counts[j])
		}
	}

	ballotsJSON, err := h.cc.ExportPublicBallots(h.ctx)
	requireNoErr(t, err)
	openingsJSON, err := h.cc.ExportAuditOpenings(h.ctx)
	requireNoErr(t, err)
	params, err := h.cc.GetParams(h.ctx)
	requireNoErr(t, err)

	var ballots []PublicBallot
	requireNoErr(t, json.Unmarshal([]byte(ballotsJSON), &ballots))
	var openings []AuditLogEntry
	requireNoErr(t, json.Unmarshal([]byte(openingsJSON), &openings))

	options := make([]string, numSlots)
	for j := range options {
		options[j] = fmt.Sprintf("option-%d", j)
	}
	queries := make([]map[string]string, 0, len(queryLog))
	for _, q := range queryLog {
		queries = append(queries, map[string]string{"hC": q})
	}

	snapshot := map[string]any{
		"meta": map[string]any{
			"eid":            "EID-FIXTURE-2026",
			"constituencyID": testConst,
			"blockHeight":    "1042",
			"txID":           "tx-freeze-fixture",
			"generatedAt":    "2026-01-01T00:00:00Z",
		},
		"ballots":   ballots,
		"openings":  openings,
		"querylog":  queries,
		"params":    params,
		"publicKey": map[string]string{"n": fmt.Sprintf("%x", key.n), "g": fmt.Sprintf("%x", new(big.Int).Add(key.n, big.NewInt(1))), "n2": fmt.Sprintf("%x", key.n2)},
		"options":   options,
		"tally":     map[string]string{"cTally": aggregate.Text(16)},
		"results":   map[string]any{"counts": decoded},
	}

	requireNoErr(t, os.MkdirAll(outDir, 0o755))
	write := func(name string, body []byte) {
		requireNoErr(t, os.WriteFile(filepath.Join(outDir, name), body, 0o644))
	}
	blob, err := json.MarshalIndent(snapshot, "", " ")
	requireNoErr(t, err)
	write("snapshot.json", append(blob, '\n'))
	write("keyday.txt", []byte(hex.EncodeToString(keyDay)+"\n"))
	write("total.txt", []byte(total.String()+"\n"))

	t.Logf("fixture written to %s: %d ballots (%d current, %d excluded), %d openings, counts %v",
		outDir, len(ballots), len(cast), len(invalid), len(openings), decoded)
}
