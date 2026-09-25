# Public verification tools

Two tools turn what the ledger publishes into something a third party can check
without any privileged access.

| File | What it does |
|---|---|
| `export_freeze.js` | Builds the verification pack for one constituency at the freeze point: the freeze list S, the published ciphertext set, the audit log, the query log, the parameters, the option list and the election public key, with a manifest that carries the freeze commitment HR and a digest of every file. |
| `verify_public.py` | Runs the Tier A checks over a pack and prints which of them passed, which failed, and which could not be run on the present prototype. |
| `test_export_freeze.js` | Tests for the exporter. |
| `test_verify_public.py` | Tests for the verifier, including agreement with the Go implementation of the audit coin. |
| `fixtures/` | A small election exported by the contract itself (see below). |

## Running them

```bash
# a pack from files the chaincode produced
node export_freeze.js --out pack \
     --ballots ballots.json --openings openings.json \
     --params params.json --pubkey publickey.json --options options.json

# or straight from a peer
node export_freeze.js --out pack --gateway --profile gateway.json --state UP --constituency C-001

# the public checks; K_day is published after the poll closes
python3 verify_public.py --pack pack --key-day <K_day hex> --total <decrypted packed total>
```

`verify_public.py` exits 0 when every check that could be run passed, 1 when a
check failed, and 2 on a usage error. Add `--json` for machine-readable output.

Requirements: Node 18 or later and Python 3.9 or later, both with the standard
library only. The `--gateway` path additionally needs `@hyperledger/fabric-gateway`
and `@grpc/grpc-js`; it is loaded lazily, so a verifier who works from published
files never installs them. That path talks to a peer and is therefore not
covered by the offline tests.

## The checks

| Ref | Check | Status on the present prototype |
|---|---|---|
| 0 | every file in the pack matches its digest in the manifest | runs |
| i | HR recomputed from S equals the published HR | runs |
| ii | SHA256(C_s) equals the recorded hC, for every current ballot | runs |
| iii | the product of the current ciphertexts mod n² equals the published C_tally | runs |
| iv-a | trustee partial decryptions D_i and proofs π_i | not available: threshold decryption is specified but not implemented |
| iv-b | the decoded option counts match the published result | runs when the decrypted total is supplied |
| v | the quorum certificate QC_TVP over h_TVP | not available: result authorisation is specified but not implemented |
| vi-a | the revealed K_day matches the commitment cK_day published before the poll | runs |
| vi-b | cast-or-audit reconciliation: no recorded ballot was selected by the coin, every opening was selected, the two sets are disjoint, and every query-log entry is accounted for | runs |
| vi-c | every published opening re-derives to its commitment | runs |

A check that cannot be run is reported as "not available" rather than as a pass.
The summary line states that the properties behind those checks are not
established by the run.

## The fixture

`fixtures/snapshot.json` is not written by hand. It is produced by
`fabric/chaincode/accumvote/fixture_export_test.go`, which runs a small election
against the contract itself: eleven ballots recorded under the packed encoding,
one of them excluded with a published reason, three ballots that the cast-or-audit
coin selected for opening, the tally, and the publication of the ciphertext set
through `ApplyBallotStatuses`. Regenerate it with

```bash
cd fabric/chaincode/accumvote
FIXTURE_OUT=../../tools/fixtures go test -run TestFixture_ExportVerificationPack -count=1
```

The test skips unless `FIXTURE_OUT` is set, so it does not affect an ordinary
`go test ./...` run. Because the fixture comes from the contract, a change in the
contract's export format breaks the tool tests instead of surfacing in a
deployment.

## Running the tests

```bash
node --test fabric/tools/
python3 -m unittest discover -s fabric/tools -p 'test_*.py' -v
```

## Privacy rules the exporter enforces

The booth identifier, the device identifier and the time of an opening are
dropped from the published audit log. An opening shows the option a voter had
just chosen, so a booth and a time would come close to naming that voter. Those
fields stay in the restricted audit record that Tier B uses for fault
attribution. The tests assert that they never reach the pack.
