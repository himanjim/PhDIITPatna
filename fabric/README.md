# Fabric module

This directory contains the Hyperledger Fabric side of the prototype, including
chaincode, network configurations for Raft and SmartBFT, benchmark harnesses,
the public verification tools, setup and testing runbooks, generated data, and
captured results.

## Subdirectories

- `benchmarks/`: Caliper and standalone Node.js benchmark code.
- `chaincode/`: AccumVote, booth metadata, and preload contracts.
- `data/`: synthetic and compact dataset-generation assets.
- `docs/`: setup and testing runbooks.
- `network/`: Raft and SmartBFT network configuration files.
- `results/`: result artefacts used in reporting.
- `tools/`: the public verification tools, `export_freeze.js` and
  `verify_public.py`, with their tests and a fixture produced by the contract.

## Reproducing the chaincode tests

The AccumVote contract carries its own Go module, its generated mocks and its
full test suite, so the suite runs from a fresh checkout without a peer, a
network or Docker. The `vendor/` directory is not committed, so restore it once
with `go mod vendor` before the first run. The procedure, the Go version
requirement and the test inventory are in
`chaincode/accumvote/README.md`.

## Notes

- The benchmark harnesses in `benchmarks/` still need a Node dependency
  manifest if they are to be reused from a fresh checkout.
- `chaincode/boothpdc/` and `chaincode/evote-preload/` do not yet carry module
  files of their own. Only `chaincode/accumvote/` is currently reproducible from
  a clean clone.
