# Chaincode sources

This directory contains the Go chaincode implementations used by the Fabric
prototype.

## Contents

- `accumvote/`: the main voting and tally contract, its Go module, its generated
  mocks and its test suite. This is the contract the thesis evaluates.
- `boothpdc/`: booth metadata preload and query contract.
- `evote-preload/`: candidate and voter-roll preload contract.

## Module status

`accumvote/` carries `go.mod`, `go.sum`, `main.go` and a `fakes` package of
generated mocks, so its suite of 56 tests runs from a clean clone once
`go mod vendor` has been run. See `accumvote/README.md` for the commands and for
the reason `vendor/` is not committed.

`boothpdc/` and `evote-preload/` are still source-only. They are compiled as part
of a deployment rather than tested in isolation, and they would each need their
own module file before that changed.

## Notes

- Keep the mocks in `accumvote/fakes` committed. Regenerating them on every run
  would make the suite depend on `mockgen` being installed, which defeats the
  purpose of a self-contained test tree.
