# Fabric testing notes

This directory contains test-oriented runbooks for the Go chaincode and related
validation flows.

## Contents

- `hyperledger-go-testing.txt`: test commands and test-environment notes.

## Where the authoritative commands live

The runnable procedure for the AccumVote suite is in
`fabric/chaincode/accumvote/README.md`. That file records the Go version the
module declares, the one-off `go mod vendor` step needed because the vendor tree
is not committed, the inventory of the 56 tests, and the environment-guarded
fixture generator. The notes in this directory are background material and
should not be treated as the current command set where the two differ.

The public verification tools have their own procedure in
`fabric/tools/README.md`. Those tests are independent of the chaincode module and
need only Node and Python.

## Notes

- Keep this directory aligned with the actual module and test layout of the
  chaincode directories.
