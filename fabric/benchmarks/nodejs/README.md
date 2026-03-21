# Standalone Node.js benchmarks

This directory contains direct Fabric Gateway benchmark clients for RecordVote and related paths.

## Contents
- Generic and profile-specific benchmark drivers for Raft and SmartBFT.
- A runbook documenting how these scripts were executed.

## Notes
- The scripts currently rely on external Node packages but no local `package.json` is present.
- Keep connection-profile paths and chaincode names aligned with the target network configuration.