# Raft network configuration

This directory contains the Raft deployment files for the benchmark network.

## Contents
- `configtx.yaml`: channel configuration template.
- `crypto-config.yaml`: cryptogen topology.
- `docker-compose*.yaml`: runtime deployment specification.
- `connection-eci.yaml`: client gateway profile.
- `fabric-accumvote.yaml`: Caliper network profile.

## Notes
- Keep chaincode names, connection-profile paths, and crypto locations aligned across these files.