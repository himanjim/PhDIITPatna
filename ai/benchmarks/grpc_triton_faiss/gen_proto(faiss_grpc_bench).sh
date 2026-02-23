#!/usr/bin/env bash
set -euo pipefail

# Generate stubs in ./gen
mkdir -p gen
python -m grpc_tools.protoc -I. --python_out=gen --grpc_python_out=gen dedup.proto
echo "[OK] Generated stubs in $(pwd)/gen"
