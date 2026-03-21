# SmartBFT network configuration

This directory contains the SmartBFT deployment files for the benchmark network.

## Contents
- `configtx.yaml`: channel configuration template.
- `crypto-config.yaml`: cryptogen topology.
- `docker-compose*.yaml`: runtime deployment specification.
- `connection-eci.yaml`: client gateway profile.
- `fabric-accumvote.yaml`: Caliper network profile.
- `smartbft-commands.docx`: operational runbook.

## Notes
- Keep benchmark profiles consistent with the deployed chaincode name and connection profile used in the SmartBFT stack.