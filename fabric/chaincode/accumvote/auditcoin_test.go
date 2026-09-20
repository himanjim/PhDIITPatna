// auditcoin_test.go covers the cast-or-audit coin that supplies cast-as-intended
// verification without asking the voter to make a security decision.
//
// The properties that matter, and that these tests pin down:
//   - the coin is a deterministic function of (K_day, hC), so the verifier kiosk
//     has no discretion and the whole election can be re-checked afterwards;
//   - the outcome is unpredictable without K_day, so the voting client cannot
//     know at commit time whether the ballot it is building will be opened;
//   - K_day is committed to before polling and verified on release, so it cannot
//     be chosen after the fact to avoid opening the ballots a device tampered with;
//   - a device that ignores the coin and casts a ballot that should have been
//     opened is detectable by reconciliation.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"testing"
)

func testAuditKey(seed string) []byte {
	sum := sha256.Sum256([]byte("audit-test-key/" + seed))
	return sum[:]
}

func testHC(i int) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("ciphertext-%d", i)))
	return hex.EncodeToString(sum[:])
}

// TestAuditCoin_DeterministicAndCaseInsensitive checks that the same inputs
// always yield the same outcome, and that presentation differences in the
// commitment string (case, surrounding whitespace) cannot change it.
//
// This matters operationally: the gateway, the kiosk and the public verifier kit
// each receive hC through a different path, and a disagreement between them
// would look identical to tampering.
func TestAuditCoin_DeterministicAndCaseInsensitive(t *testing.T) {
	key := testAuditKey("determinism")
	h := testHC(1)

	first, err := AuditCoin(key, h, 20)
	requireNoErr(t, err)
	for i := 0; i < 50; i++ {
		again, err := AuditCoin(key, h, 20)
		requireNoErr(t, err)
		if again != first {
			t.Fatal("audit coin is not deterministic")
		}
	}

	upper, err := AuditCoin(key, "  "+hexUpper(h)+"  ", 20)
	requireNoErr(t, err)
	if upper != first {
		t.Fatal("audit coin must ignore case and surrounding whitespace in hC")
	}
}

func hexUpper(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'a' && b[i] <= 'f' {
			b[i] -= 'a' - 'A'
		}
	}
	return string(b)
}

// TestAuditCoin_DependsOnKey checks that the outcome cannot be predicted without
// K_day: a different key produces a different pattern of decisions.
//
// This is the property that stops a compromised voting client from cheating only
// on ballots it knows will never be opened.
func TestAuditCoin_DependsOnKey(t *testing.T) {
	k1 := testAuditKey("alpha")
	k2 := testAuditKey("beta")

	differences := 0
	const n = 500
	for i := 0; i < n; i++ {
		h := testHC(i)
		a, err := AuditCoin(k1, h, 2)
		requireNoErr(t, err)
		b, err := AuditCoin(k2, h, 2)
		requireNoErr(t, err)
		if a != b {
			differences++
		}
	}
	// With oneIn=2 two independent keys disagree about half the time. Anything
	// close to zero would mean the key is not actually mixed into the decision.
	if differences < n/5 {
		t.Fatalf("only %d/%d decisions differ between keys; coin does not depend on the key", differences, n)
	}
}

// TestAuditCoin_RateIsCloseToConfigured checks that roughly one ballot in
// `oneIn` is selected, so the published audit rate is honest.
//
// The bound is deliberately loose: this is a sanity check on the distribution,
// not a statistical test of HMAC-SHA256.
func TestAuditCoin_RateIsCloseToConfigured(t *testing.T) {
	key := testAuditKey("rate")
	const oneIn = 20
	const n = 20000

	opened := 0
	for i := 0; i < n; i++ {
		open, err := AuditCoin(key, testHC(i), oneIn)
		requireNoErr(t, err)
		if open {
			opened++
		}
	}
	expected := n / oneIn
	low, high := expected*7/10, expected*13/10
	if opened < low || opened > high {
		t.Fatalf("opened %d of %d ballots, expected about %d (accepted range %d..%d)",
			opened, n, expected, low, high)
	}
	t.Logf("audit rate: %d of %d ballots opened (target 1 in %d)", opened, n, oneIn)
}

// TestAuditCoin_RejectsBadParameters checks the guards that stop a deployment
// from silently disabling the audit.
func TestAuditCoin_RejectsBadParameters(t *testing.T) {
	if _, err := AuditCoin([]byte("short"), testHC(1), 20); err == nil {
		t.Fatal("expected rejection of an undersized audit key")
	}
	if _, err := AuditCoin(testAuditKey("x"), testHC(1), 0); err == nil {
		t.Fatal("expected rejection of oneIn = 0")
	}
	if _, err := AuditCoin(testAuditKey("x"), "   ", 20); err == nil {
		t.Fatal("expected rejection of an empty commitment")
	}
}

// TestAuditCoin_OutcomeStrings checks the string form used by the gateway API
// and the published audit log.
func TestAuditCoin_OutcomeStrings(t *testing.T) {
	key := testAuditKey("outcome")
	for i := 0; i < 100; i++ {
		h := testHC(i)
		open, err := AuditCoin(key, h, 3)
		requireNoErr(t, err)
		got, err := AuditOutcome(key, h, 3)
		requireNoErr(t, err)
		want := AuditOutcomeCast
		if open {
			want = AuditOutcomeOpen
		}
		if got != want {
			t.Fatalf("AuditOutcome = %q, want %q", got, want)
		}
	}
}

// TestAuditCoin_KeyCommitmentRoundTrip checks the pre-poll commitment to K_day
// and its verification on release. Without this, the authority could pick a key
// after the fact that happens to open no tampered ballot.
func TestAuditCoin_KeyCommitmentRoundTrip(t *testing.T) {
	key := testAuditKey("commit")

	commit, err := AuditKeyCommitment(key)
	requireNoErr(t, err)
	if len(commit) != 64 {
		t.Fatalf("commitment length = %d, want 64 hex chars", len(commit))
	}
	requireNoErr(t, VerifyAuditKeyCommitment(key, commit))
	// Case and whitespace tolerance on the published value.
	requireNoErr(t, VerifyAuditKeyCommitment(key, " "+hexUpper(commit)+" "))

	// A different key must not verify against the published commitment.
	if err := VerifyAuditKeyCommitment(testAuditKey("other"), commit); err == nil {
		t.Fatal("expected rejection of a substituted audit key")
	}
	if _, err := AuditKeyCommitment([]byte("short")); err == nil {
		t.Fatal("expected rejection of an undersized audit key")
	}
}

// TestAuditCoin_ReconcileDetectsIgnoredCoin is the end-of-election check.
//
// A compromised terminal that receives "open" and casts the ballot anyway leaves
// its commitment in the cast set with no matching opening. Once K_day is
// released, reconciliation names exactly those ballots.
func TestAuditCoin_ReconcileDetectsIgnoredCoin(t *testing.T) {
	key := testAuditKey("reconcile")
	const oneIn = 5

	var cast, opened, ignored []string
	for i := 0; i < 300; i++ {
		h := testHC(i)
		open, err := AuditCoin(key, h, oneIn)
		requireNoErr(t, err)
		switch {
		case !open:
			cast = append(cast, h)
		case i%2 == 0:
			// Honest terminal: opened and published.
			opened = append(opened, h)
		default:
			// Dishonest terminal: coin said open, but it cast the ballot.
			cast = append(cast, h)
			ignored = append(ignored, h)
		}
	}
	if len(ignored) == 0 {
		t.Fatal("test fixture produced no ignored-coin cases")
	}

	shouldHaveOpened, openedWithoutCause, err := ReconcileAuditLog(key, oneIn, cast, opened)
	requireNoErr(t, err)

	if len(openedWithoutCause) != 0 {
		t.Fatalf("unexpected openings without cause: %v", openedWithoutCause)
	}
	if len(shouldHaveOpened) != len(ignored) {
		t.Fatalf("reconciliation found %d ignored-coin ballots, want %d",
			len(shouldHaveOpened), len(ignored))
	}
	found := make(map[string]struct{}, len(shouldHaveOpened))
	for _, h := range shouldHaveOpened {
		found[h] = struct{}{}
	}
	for _, h := range ignored {
		if _, ok := found[h]; !ok {
			t.Fatalf("reconciliation missed ignored-coin ballot %s", h)
		}
	}
	t.Logf("reconciliation flagged %d ballots cast despite an 'open' coin", len(shouldHaveOpened))
}

// TestAuditCoin_ReconcileDetectsUnjustifiedOpening checks the opposite abuse: a
// terminal discarding a ballot by claiming it was audited when the coin did not
// call for it.
func TestAuditCoin_ReconcileDetectsUnjustifiedOpening(t *testing.T) {
	key := testAuditKey("unjustified")
	const oneIn = 5

	var cast, opened []string
	var bogus string
	for i := 0; i < 200; i++ {
		h := testHC(i)
		open, err := AuditCoin(key, h, oneIn)
		requireNoErr(t, err)
		if open {
			opened = append(opened, h)
			continue
		}
		if bogus == "" {
			// Terminal discards a ballot the coin said to cast.
			bogus = h
			opened = append(opened, h)
			continue
		}
		cast = append(cast, h)
	}
	if bogus == "" {
		t.Fatal("test fixture produced no unjustified opening")
	}

	shouldHaveOpened, openedWithoutCause, err := ReconcileAuditLog(key, oneIn, cast, opened)
	requireNoErr(t, err)
	if len(shouldHaveOpened) != 0 {
		t.Fatalf("unexpected ignored-coin findings: %v", shouldHaveOpened)
	}
	if len(openedWithoutCause) != 1 || openedWithoutCause[0] != bogus {
		t.Fatalf("expected exactly the unjustified opening %s, got %v", bogus, openedWithoutCause)
	}
}
