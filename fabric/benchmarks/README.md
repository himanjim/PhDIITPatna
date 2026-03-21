# Fabric benchmarks

This directory contains benchmark harnesses and helper tools used to measure Fabric transaction performance outside the chaincode source tree.

## Contents
- `caliper/`: Caliper network/workload definitions and helper scripts.
- `nodejs/`: standalone Fabric Gateway benchmark clients.
- `tools/`: small benchmark helpers such as ping probes.

## Notes
- This subtree needs its own Node dependency manifest if it is to be reused from a fresh checkout.