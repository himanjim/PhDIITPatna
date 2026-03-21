# Caliper benchmarks

This directory contains Caliper workload definitions, network/workload YAML files, and helper JavaScript modules used to benchmark Fabric transactions.

## Contents
- YAML workload definitions for smoke, ping, record-vote, and full-loop scenarios.
- Helper modules such as `smoke-getparams.js` and `tallyclose.js`.
- A command runbook documenting how the benchmark suite was executed.

## Notes
- Keep workload files aligned with the active chaincode name and network profile.
- Record any benchmark assumptions that are not encoded directly in the YAML files.