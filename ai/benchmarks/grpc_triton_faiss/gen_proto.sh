#!/usr/bin/env bash
set -euo pipefail
mkdir -p gen
python -m grpc_tools.protoc -I. --python_out=gen --grpc_python_out=gen dedup.proto
touch gen/__init__.py
echo "[OK] Generated stubs in $(pwd)/gen"
