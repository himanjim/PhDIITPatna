# Chaincode sources

This directory contains the Go chaincode implementations used by the Fabric prototype.

## Contents
- `accumvote/`: main voting and tally contract plus tests.
- `boothpdc/`: booth metadata preload and query contract.
- `evote-preload/`: candidate and voter-roll preload contract.

## Notes
- Each chaincode directory should carry its own `go.mod` and test instructions.