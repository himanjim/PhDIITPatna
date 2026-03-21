#!/usr/bin/env bash
# Generate Python gRPC client and server stubs from the local Protocol
# Buffers definition used by the FAISS de-duplication services. The script
# is intentionally minimal: it creates the output package directory, runs
# protoc with the Python and gRPC Python plugins, and ensures that the
# generated directory is importable as a Python package. This utility is
# primarily a build convenience for reproducible local development and
# experimental deployment.
set -euo pipefail
mkdir -p gen

# Invoke protoc through grpc_tools to generate both the base Python
# protobuf classes and the corresponding gRPC service stubs from
# dedup.proto, using the current directory as the import root.
python -m grpc_tools.protoc -I. --python_out=gen --grpc_python_out=gen dedup.proto

# Ensure that the generated output directory is recognised as a Python
# package so that the stubs can be imported using standard module syntax.
touch gen/__init__.py
echo "[OK] Generated stubs in $(pwd)/gen"
