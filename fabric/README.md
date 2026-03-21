# Fabric module

This directory contains the Hyperledger Fabric side of the prototype, including chaincode, network configurations for Raft and SmartBFT, benchmark harnesses, setup/testing runbooks, generated data, and captured results.

## Subdirectories
- `benchmarks/`: Caliper and standalone Node.js benchmark code.
- `chaincode/`: AccumVote, booth metadata, and preload contracts.
- `data/`: synthetic and compact dataset-generation assets.
- `docs/`: setup and testing runbooks.
- `network/`: Raft and SmartBFT network configuration files.
- `results/`: result artefacts used in reporting.

## Notes
- Add Go module files to each chaincode directory for reproducible builds and tests.
- Add a Node dependency manifest for the benchmark harnesses.