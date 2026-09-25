# AccumVote chaincode

This directory contains the main voting contract for the prototype together with
its Go test suite, its module definition and the generated mocks the tests run
against. The contract records packed Paillier ballots, maintains the encrypted
aggregate, implements the cast-or-audit coin, and publishes the material that the
public verification tools in `fabric/tools` consume.

## Contract sources

| File | Purpose |
|---|---|
| `accumvote.go` | The contract itself. Voter and candidate preload, ballot recording and supersession, poll state, the encrypted aggregate, the publication and export transactions, and the audit log. |
| `packed.go` | The packed ballot encoding. One Paillier ciphertext carries every contest slot as a base `2^b` digit, so a single homomorphic product yields all option counts. Encoding, checked decoding and the overflow bound live here. |
| `auditcoin.go` | The cast-or-audit coin. `AuditCoin` decides whether a ballot commitment is selected for opening, and `AuditKeyCommitment` produces the commitment to the day key that is published before the poll opens. |
| `main.go` | The chaincode entry point used when the contract is packaged for a peer. |
| `collections_config.json` | The private-data collection definition used when the contract is deployed. |

## Test suite

The suite is written against mocked Fabric interfaces, so it runs without a
network, a peer or Docker. It contains 56 test functions. Fifty-five run in an
ordinary `go test ./...` invocation and one is the fixture generator described
below, which skips unless an environment variable is set.

| File | Tests | What it covers |
|---|---|---|
| `voting_test.go` | 8 | Ballot recording and revoting. A first vote, a repeated vote for the same candidate, a change of candidate, refusal when the public key is absent or the poll is closed, rejection of a non-hexadecimal or out-of-range ciphertext, and the state-operation budget of a revote. |
| `voting_critical_test.go` | 9 | The same paths under strict validation, adding a non-invertible ciphertext and an unknown candidate when candidate validation is enabled. |
| `tally_test.go` | 5 | The homomorphic aggregate. The identity element before any vote, a simple distribution, the effect of a revote, agreement between `GetEncSums` and the tally, and exclusion on a booth or device mismatch. |
| `publish_test.go` | 9 | Publication of the result. Refusal while the poll is open, agreement between the published figures and a decryption of the aggregate, receipt verification across a revote including supersession, and the linkage recorded when a voter is later found ineligible. |
| `packed_test.go` | 7 | The packed encoding. Slot round trips, rejection of malformed input, the overflow bound, detection of an inflated ballot, the documented case a checked decode cannot catch, parameter validation and the result mapping. |
| `auditcoin_test.go` | 8 | The cast-or-audit coin. Determinism and case insensitivity, dependence on the day key, the observed selection rate against the configured one, the key-commitment round trip, and reconciliation catching both an ignored coin and an unjustified opening. |
| `verifiability_test.go` | 7 | The verifiability properties. Public recomputation of the tally, that a packed ciphertext hides the choice, rejection of a substituted ciphertext, preservation of the receipt anchor when a ballot is invalidated, and the recording and reconciliation of audit openings. |
| `contract_ping_test.go` | 2 | Construction of the contract and the liveness transaction the benchmark harnesses call. |
| `harness_test.go` | 0 | Not a test file in itself. It holds the shared harness, the environment defaults and the helper assertions used by every other file. |
| `fixture_export_test.go` | 1 | The fixture generator. See below. |

`fakes/mock_stub.go` and `fakes/mock_txctx.go` are GoMock output for
`shim.ChaincodeStubInterface` and `contractapi.TransactionContextInterface`. They
are committed rather than regenerated on each run so that the suite does not
depend on `mockgen` being installed. Regenerate them only when the Fabric
interfaces change.

## Building and testing

The module is `github.com/yourorg/accumvote_cc`. Its `go.mod` declares Go 1.25,
so a toolchain at least that recent is required, or a Go 1.21 or later toolchain
that is permitted to fetch the declared version automatically.

```bash
cd fabric/chaincode/accumvote
go mod vendor          # see the note below, needed once per checkout
go vet ./...
go test ./...
go test -race ./...
```

A successful run reports `ok github.com/yourorg/accumvote_cc` and
`? github.com/yourorg/accumvote_cc/fakes [no test files]`. The second line is
expected, because the mocks carry no tests of their own.

## The vendor directory

`vendor/` is deliberately not committed. It holds 981 files and about 20 MB of
third-party source, which is disproportionate in a repository whose purpose is to
record the research artefacts. `go.mod` and `go.sum` are committed instead, and
they pin every dependency by version and by hash, so the vendor tree can be
rebuilt exactly.

Restore it with:

```bash
cd fabric/chaincode/accumvote
go mod download        # fetches the pinned modules into the local module cache
go mod verify          # confirms each one against the hashes in go.sum
go mod vendor          # writes vendor/ from the cache
```

`go mod download` is the only step that needs network access. Once `vendor/`
exists the build is fully offline, and Go uses it automatically because a
`vendor/modules.txt` consistent with `go.mod` is present. To be explicit, add
`-mod=vendor`, and to force the opposite behaviour use `-mod=mod`.

If the machine has no route to `proxy.golang.org`, run the three commands on a
machine that does and copy the resulting `vendor/` directory across. Nothing else
is needed, because the tests import no packages beyond those already pinned.

## The verification fixture

`fixture_export_test.go` is not an assertion test. It runs a small but complete
election against the contract and writes the resulting public material to disk in
the shape that `fabric/tools/export_freeze.js` consumes. Fourteen ballots are
scripted: ten recorded and counted, one recorded and then excluded with a
published reason, and three selected by the cast-or-audit coin and opened instead
of cast. The test skips unless `FIXTURE_OUT` names a directory.

```bash
cd fabric/chaincode/accumvote
FIXTURE_OUT=../../tools/fixtures go test -run TestFixture_ExportVerificationPack -count=1
```

The output is deterministic. Regenerating the fixture over an unchanged contract
reproduces `fabric/tools/fixtures/snapshot.json`,
`fabric/tools/fixtures/keyday.txt` and `fabric/tools/fixtures/total.txt` byte for
byte. That is the point of the arrangement: the public verifier is
exercised against data the contract itself produced, so a change in the contract's
export format breaks the tool tests rather than surfacing later in a deployment.

## Related directories

- `fabric/tools/` holds the public verification tools and their tests, and
  consumes the fixture written here.
- `fabric/docs/testing/` holds the environment notes for running Go tests.
- `fabric/network/` holds the Raft and SmartBFT configurations used when the
  contract is deployed rather than mocked.
